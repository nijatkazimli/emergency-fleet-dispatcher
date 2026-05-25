"""Cross-platform Tkinter UI for the Intelligent Emergency Fleet Dispatcher.

Tkinter ships with the standard Python installer on both macOS and Windows,
so this UI runs on either OS with zero extra dependencies.

What it shows
-------------
* The mock city as a grid of intersections + roads.
* Ambulances (blue squares) and emergencies (red circles).
* The N x M cost matrix produced by Algorithm A.
* The current assignment (drawn as coloured Dijkstra paths) and its total cost.

Controls
--------
* Regenerate city  -- new random grid + closures.
* Place units      -- new random ambulance / emergency positions.
* Dispatch         -- runs Algorithm A, then the placeholder assignment
                      (random or greedy, selectable). When Partner 2 wires
                      `solve_assignment` in, just point the dropdown at it.
"""

from __future__ import annotations

import random
import time
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Tuple, cast

from routing import UNREACHABLE, build_cost_matrix, dijkstra_with_paths, reconstruct_path
from routing.city_generator import build_grid_city, node_id

from .assignment import (
    Assignment,
    greedy_assignment,
    random_assignment,
    total_cost,
)
from .triage import Emergency, triage

Coord = Tuple[int, int]

# ---- Layout constants -------------------------------------------------------
CELL_PX = 72
PADDING = 36
NODE_R = 6
UNIT_R = 14
PATH_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


class Tooltip:
    """Lightweight hover tooltip for any Tk widget (stdlib only)."""

    def __init__(self, widget: tk.Widget, text: "str | Any", delay_ms: int = 450) -> None:
        self.widget = widget
        self.text = text  # str or zero-arg callable returning str
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _evt: Any) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _evt: Any) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        msg = self.text() if callable(self.text) else self.text
        if not msg:
            return
        msg = str(msg)
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        try:
            tip.attributes("-topmost", True)
        except tk.TclError:
            pass
        tk.Label(
            tip,
            text=msg,
            background="#fffbcc",
            foreground="#111111",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=3,
            font=("TkDefaultFont", 9),
            justify=tk.LEFT,
        ).pack()
        self._tip = tip

    def _hide(self) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


class DispatcherApp(tk.Tk):
    def __init__(self, rows: int = 8, cols: int = 8) -> None:
        super().__init__()
        self.title("Emergency Fleet Dispatcher — Demo")
        self.rows = rows
        self.cols = cols

        self.n_ambulances = tk.IntVar(value=3)
        self.n_emergencies = tk.IntVar(value=3)
        self.strategy = tk.StringVar(value="random (placeholder)")

        # Resizable city dimensions, driven by the toolbar Rows/Cols spinboxes.
        self.grid_rows = tk.IntVar(value=rows)
        self.grid_cols = tk.IntVar(value=cols)

        # Animation speed multiplier: sim seconds per wall second.
        # Changes take effect on the next tick, mid-animation included.
        self.sim_speed = tk.DoubleVar(value=5.0)

        # Initialised by place_units(); declared up-front so the first
        # regenerate_city() -> _redraw() call has something to draw.
        self.ambulance_coords: List[Coord] = []
        self.emergency_coords: List[Coord] = []
        self.all_emergencies: List[Emergency] = []
        self.closures: List[Tuple[Coord, Coord]] = []

        # Animation state (populated by dispatch(), driven by play_animation()).
        self.routes: List[Dict[str, Any]] = []
        self._anim_running: bool = False
        # Sim-time accumulator (sim seconds elapsed since play_animation()).
        # Using an accumulator instead of (now - start) * speed lets the user
        # change sim_speed mid-animation without time-jumps.
        self._anim_sim_t: float = 0.0
        self._anim_last_wall: float = 0.0
        # Active arrival pulses: {x, y, color, start_wall}. Each one is
        # drawn for ~600 ms wall time and then pruned.
        self._pulses: List[Dict[str, Any]] = []

        # Last-known totals for the side-by-side strategy comparison
        # footer. Filled in by dispatch().
        self._random_total: float | None = None
        self._greedy_total: float | None = None

        # Per-dispatch Dijkstra cache: ambulance node -> prev-map. Filled
        # by dispatch() (one Dijkstra per ambulance, with early
        # termination on the emergency set). _build_routes() and _redraw()
        # reuse it instead of each re-running N Dijkstras.
        self._prev_cache: Dict[int, Dict[Any, Any]] = {}

        # O(1) edge-weight lookup for _redraw, rebuilt on regenerate_city.
        # Replaces the O(deg) linear scan over `city.neighbors(u)`.
        self._edge_w: Dict[Tuple[int, int], float] = {}

        self._build_widgets()
        self.regenerate_city()
        self.place_units()
        self.dispatch()

        # Bring the window to the foreground on launch (especially on macOS,
        # where Tk apps otherwise open behind the active app).
        self.after(50, self._raise_to_front)

    def _raise_to_front(self) -> None:
        try:
            self.lift()
            self.attributes("-topmost", True)
            self.focus_force()
            # Drop the always-on-top flag a moment later so the window
            # behaves normally afterwards.
            self.after(400, lambda: self.attributes("-topmost", False))
        except tk.TclError:
            pass

    # ---------------------------------------------------------------- UI ----
    def _build_widgets(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        regen_btn = ttk.Button(top, text="Regenerate city", command=self.regenerate_city)
        regen_btn.pack(side=tk.LEFT, padx=4)
        Tooltip(regen_btn, "Rebuild the road network with new random weights and closures.")

        place_btn = ttk.Button(top, text="Place units", command=self.place_units)
        place_btn.pack(side=tk.LEFT, padx=4)
        Tooltip(place_btn, "Randomly position ambulances and emergencies on the grid.")

        dispatch_btn = ttk.Button(top, text="Dispatch", command=self.dispatch)
        dispatch_btn.pack(side=tk.LEFT, padx=4)
        Tooltip(dispatch_btn, "Run triage, build the cost matrix, and assign ambulances to emergencies.")

        self.play_btn = ttk.Button(top, text="Play \u25B6", command=self.play_animation)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        self.play_btn.state(["disabled"])
        Tooltip(
            self.play_btn,
            lambda: (
                "Animate ambulances along their assigned routes."
                if "disabled" not in self.play_btn.state()
                else "Disabled \u2014 click Dispatch first to assign routes."
            ),
        )

        ttk.Label(top, text="  Ambulances:").pack(side=tk.LEFT)
        amb_spin = ttk.Spinbox(top, from_=1, to=12, width=3, textvariable=self.n_ambulances)
        amb_spin.pack(side=tk.LEFT)
        Tooltip(amb_spin, "Number of ambulances. Takes effect on the next Place units.")

        ttk.Label(top, text="  Emergencies:").pack(side=tk.LEFT)
        emg_spin = ttk.Spinbox(top, from_=1, to=12, width=3, textvariable=self.n_emergencies)
        emg_spin.pack(side=tk.LEFT)
        Tooltip(emg_spin, "Number of emergency calls. If it exceeds ambulances, triage queues the rest.")

        ttk.Label(top, text="  Rows:").pack(side=tk.LEFT)
        rows_spin = ttk.Spinbox(
            top, from_=4, to=16, width=3,
            textvariable=self.grid_rows, command=self._on_size_change,
        )
        rows_spin.pack(side=tk.LEFT)
        Tooltip(rows_spin, "City height (rows). Changing this rebuilds the city.")

        ttk.Label(top, text="  Cols:").pack(side=tk.LEFT)
        cols_spin = ttk.Spinbox(
            top, from_=4, to=16, width=3,
            textvariable=self.grid_cols, command=self._on_size_change,
        )
        cols_spin.pack(side=tk.LEFT)
        Tooltip(cols_spin, "City width (columns). Changing this rebuilds the city.")

        ttk.Label(top, text="  Strategy:").pack(side=tk.LEFT)
        strat_box = ttk.Combobox(
            top,
            textvariable=self.strategy,
            values=["random (placeholder)", "greedy (baseline)"],
            state="readonly",
            width=22,
        )
        strat_box.pack(side=tk.LEFT)
        Tooltip(strat_box, "Assignment policy. Swap for the partner's Hungarian solver when ready.")

        ttk.Label(top, text="  Speed:").pack(side=tk.LEFT)
        self.speed_label = ttk.Label(top, text="5.0x", width=5)
        speed_scale = ttk.Scale(
            top, from_=1.0, to=20.0, length=120,
            variable=self.sim_speed, orient=tk.HORIZONTAL,
            command=lambda v: self.speed_label.config(text=f"{float(v):.1f}x"),
        )
        speed_scale.pack(side=tk.LEFT, padx=(2, 0))
        self.speed_label.pack(side=tk.LEFT, padx=(2, 4))
        Tooltip(speed_scale, "Animation speed: sim seconds per wall second.\nApplies live, even mid-playback.")

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_w = self.cols * CELL_PX + 2 * PADDING
        canvas_h = self.rows * CELL_PX + 2 * PADDING
        self.canvas = tk.Canvas(body, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack(side=tk.LEFT, padx=8, pady=8)

        side = ttk.Frame(body, padding=8)
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Helper to spawn a read-only, theme-independent Text widget.
        def make_text(height: int) -> tk.Text:
            return tk.Text(
                side,
                height=height,
                width=46,
                font=("Menlo", 11),
                state="disabled",
                cursor="arrow",
                background="#f7f7f7",
                foreground="#111111",
                insertbackground="#111111",
                relief=tk.FLAT,
                borderwidth=1,
                highlightthickness=0,
            )

        ttk.Label(side, text="Cost matrix (rows = ambulances, cols = emergencies)").pack(anchor=tk.W)
        self.matrix_box = make_text(14)
        self.matrix_box.pack(fill=tk.BOTH, expand=False, pady=(2, 8))

        ttk.Label(side, text="Assignment").pack(anchor=tk.W)
        self.assignment_box = make_text(10)
        self.assignment_box.pack(fill=tk.BOTH, expand=True, pady=(2, 8))

        ttk.Label(side, text="Queued (N > M overflow)").pack(anchor=tk.W)
        self.queue_box = make_text(5)
        self.queue_box.pack(fill=tk.BOTH, expand=False, pady=(2, 8))

        # ttk.Label uses the OS-native foreground colour, so we don't force
        # one (the previous "#444" was unreadable on dark themes).
        self.legend = ttk.Label(side, text="")
        self.legend.pack(anchor=tk.W, pady=(4, 0))

        self.status = ttk.Label(side, text="")
        self.status.pack(anchor=tk.W)

    # ------------------------------------------------------------ Model ----
    def _on_size_change(self) -> None:
        """Spinbox callback: regenerate at the new grid size and re-place units."""
        try:
            new_r = int(self.grid_rows.get())
            new_c = int(self.grid_cols.get())
        except tk.TclError:
            return  # mid-typing in the spinbox; ignore.
        if new_r == self.rows and new_c == self.cols:
            return
        self.regenerate_city()  # picks up the new dims internally
        self.place_units()
        self.dispatch()

    def regenerate_city(self) -> None:
        self._stop_animation()
        self.routes = []
        self._prev_cache = {}
        if hasattr(self, "play_btn"):
            self.play_btn.state(["disabled"])

        # Honour any change to the Rows/Cols spinboxes. If the grid
        # shrank, existing unit coords could be out of bounds, so wipe
        # them and let the caller (or _on_size_change) re-place.
        new_r = int(self.grid_rows.get())
        new_c = int(self.grid_cols.get())
        size_changed = (new_r != self.rows or new_c != self.cols)
        if size_changed:
            self.rows, self.cols = new_r, new_c
            if hasattr(self, "canvas"):
                self.canvas.config(
                    width=self.cols * CELL_PX + 2 * PADDING,
                    height=self.rows * CELL_PX + 2 * PADDING,
                )
            self.ambulance_coords = []
            self.emergency_coords = []
            self.all_emergencies = []

        seed = random.randint(0, 10_000)
        # Random road closures: aim for a visibly sparser network with a
        # handful of isolated pockets. ~22% of internal edges are removed.
        closures: List[Tuple[Coord, Coord]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if c + 1 < self.cols and random.random() < 0.22:
                    closures.append(((r, c), (r, c + 1)))
                if r + 1 < self.rows and random.random() < 0.22:
                    closures.append(((r, c), (r + 1, c)))
        self.closures = closures
        self.city = build_grid_city(self.rows, self.cols, closed_edges=closures, seed=seed)

        # Pre-compute min/max edge weight for legend + colour scaling,
        # and snapshot every edge into an (u,v) -> w dict for O(1) lookup
        # during _redraw (was an O(deg) linear scan over neighbors).
        weights: List[float] = []
        edge_w: Dict[Tuple[int, int], float] = {}
        for u in self.city.nodes():
            u_i = cast(int, u)
            for v, w in self.city.neighbors(u):
                weights.append(w)
                edge_w[(u_i, cast(int, v))] = w
        self._edge_w = edge_w
        self.w_min = min(weights) if weights else 1.0
        self.w_max = max(weights) if weights else 1.0
        self.legend.config(
            text=f"Road colour = travel time (light = {self.w_min:.2f}, "
                 f"dark red = {self.w_max:.2f}). Dashed = closed."
        )
        self._redraw()

    def place_units(self) -> None:
        self._stop_animation()
        self.routes = []
        self._prev_cache = {}
        self.play_btn.state(["disabled"])
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        random.shuffle(all_cells)
        n_amb = self.n_ambulances.get()
        n_eme = self.n_emergencies.get()
        self.ambulance_coords = all_cells[:n_amb]
        eme_coords = all_cells[n_amb:n_amb + n_eme]
        # Assign each emergency a random priority 1..5 (5 = most urgent)
        # and a monotonically increasing call_id for FIFO tie-breaking.
        self.all_emergencies: List[Emergency] = [
            Emergency(coord=c, priority=random.randint(1, 5), call_id=k)
            for k, c in enumerate(eme_coords)
        ]
        # `emergency_coords` is what the canvas/drawing path uses; the
        # triage step in dispatch() decides which subset is active.
        self.emergency_coords = eme_coords
        self._redraw()
        self._set_text(self.assignment_box, "")
        self._set_text(self.matrix_box, "")
        self._set_text(self.queue_box, "")
        self.status.config(text="Units placed. Click Dispatch.")

    def dispatch(self) -> None:
        amb_nodes = [node_id(c, self.cols) for c in self.ambulance_coords]

        # ---- Triage (pre-processing): fleet-shortage handling -----------
        capacity = len(amb_nodes)
        dispatched, queued = triage(self.all_emergencies, capacity)

        # Map the surviving emergencies back to indices into the original
        # emergency_coords list so the UI labels (E0, E1, ...) stay stable.
        coord_to_index = {e.coord: i for i, e in enumerate(self.all_emergencies)}
        dispatched_indices = [coord_to_index[e.coord] for e in dispatched]
        eme_nodes = [node_id(self.emergency_coords[i], self.cols)
                     for i in dispatched_indices]

        # ---- Algorithm A: cost matrix + path cache in one pass ---------
        # build_cost_matrix(return_paths=True) runs Dijkstra ONCE per
        # ambulance with the emergency set as targets (early termination),
        # and hands back the predecessor maps so _build_routes and
        # _redraw can reconstruct the actual shortest paths without any
        # further Dijkstra work. Previously dispatch + redraw together
        # ran 3 * N Dijkstras; now it runs N.
        cost, prev_by_source = cast(
            Tuple[List[List[float]], Dict[Any, Dict[Any, Any]]],
            build_cost_matrix(
                self.city, amb_nodes, eme_nodes, return_paths=True
            ),
        )
        self._prev_cache = {cast(int, s): p for s, p in prev_by_source.items()}

        # ---- Algorithm B: run BOTH placeholders for live comparison -----
        # Always evaluate both strategies on the same cost matrix so the
        # side panel can display random vs greedy totals side by side.
        random_local = random_assignment(cost)
        greedy_local = greedy_assignment(cost)
        self._random_total = total_cost(cost, random_local) if random_local else None
        self._greedy_total = total_cost(cost, greedy_local) if greedy_local else None
        if self.strategy.get().startswith("greedy"):
            assignment_local: Assignment = greedy_local
        else:
            assignment_local = random_local

        # Re-map column indices from the triaged sub-list back to the
        # original emergency_coords indices so the rest of the UI
        # (drawing, badges, assignment text) keeps using E0/E1/... ids.
        assignment: Assignment = [
            (i, dispatched_indices[j]) for i, j in assignment_local
        ]

        self._show_matrix(cost, dispatched_indices)
        self._show_assignment(cost, assignment_local, dispatched_indices)
        self._show_queue(queued)
        self._redraw(cost=cost, assignment=assignment)

        # Build the per-route polylines + per-segment durations so the
        # Play button can animate ambulances along them.
        self._stop_animation()
        self.routes = self._build_routes(assignment)
        if self.routes:
            self.play_btn.state(["!disabled"])
        else:
            self.play_btn.state(["disabled"])

    # ------------------------------------------------------- Animation ----
    TICK_MS = 33  # ~30 fps
    PULSE_MS = 600  # arrival pulse duration (wall time)

    def _build_routes(self, assignment: Assignment) -> List[Dict[str, Any]]:
        routes: List[Dict[str, Any]] = []
        for k, (i, j) in enumerate(assignment):
            color = PATH_COLORS[k % len(PATH_COLORS)]
            src = node_id(self.ambulance_coords[i], self.cols)
            dst = node_id(self.emergency_coords[j], self.cols)
            prev = self._prev_cache.get(src)
            if prev is None:  # defensive: cache miss => recompute.
                _, prev = dijkstra_with_paths(self.city, src, targets=[dst])
            path_nodes = reconstruct_path(prev, dst)
            if len(path_nodes) < 2:
                continue
            pts: List[Tuple[float, float]] = []
            durs: List[float] = []
            prev_n: int | None = None
            for n in path_nodes:
                n_i = cast(int, n)
                pr, pc = divmod(n_i, self.cols)
                pts.append(self._xy((pr, pc)))
                if prev_n is not None:
                    w = self._edge_weight(prev_n, n_i) or 1.0
                    durs.append(w)
                prev_n = n_i
            routes.append({
                "label": f"A{i}\u2192E{j}",
                "color": color,
                "pts": pts,
                "durs": durs,
                "total": sum(durs),
                "arrived": None,  # filled in when the unit reaches its target
            })
        return routes

    def play_animation(self) -> None:
        if not self.routes:
            return
        self._stop_animation()
        for route in self.routes:
            route["arrived"] = None
        self._pulses = []
        self._anim_sim_t = 0.0
        self._anim_last_wall = time.perf_counter()
        self._anim_running = True
        self._tick()

    def _stop_animation(self) -> None:
        self._anim_running = False
        self._pulses = []
        self.canvas.delete("moving")

    def _tick(self) -> None:
        if not self._anim_running:
            return
        # Sim-time accumulator: integrate (wall_dt * current speed) so the
        # user can move the Speed slider mid-animation without time jumps.
        now = time.perf_counter()
        dt_wall = now - self._anim_last_wall
        self._anim_last_wall = now
        try:
            speed = float(self.sim_speed.get())
        except tk.TclError:
            speed = 5.0
        self._anim_sim_t += dt_wall * speed
        sim_t = self._anim_sim_t

        self.canvas.delete("moving")
        all_done = True
        for route in self.routes:
            total = route["total"]
            if sim_t >= total:
                x, y = route["pts"][-1]
                if route["arrived"] is None:
                    route["arrived"] = total
                    self._pulses.append({
                        "x": x, "y": y,
                        "color": route["color"],
                        "start_wall": now,
                    })
                # Faded, smaller marker once the unit has arrived.
                col = self._fade_color(route["color"], 0.55)
                self.canvas.create_oval(
                    x - 6, y - 6, x + 6, y + 6,
                    fill=col, outline="#ffffff", width=1, tags="moving",
                )
            else:
                all_done = False
                x, y = self._position_on_route(route, sim_t)
                self.canvas.create_oval(
                    x - 8, y - 8, x + 8, y + 8,
                    fill=route["color"], outline="white", width=2,
                    tags="moving",
                )

        # Arrival pulses: expanding outlined ring that fades over PULSE_MS.
        still: List[Dict[str, Any]] = []
        for p in self._pulses:
            age_ms = (now - p["start_wall"]) * 1000.0
            if age_ms >= self.PULSE_MS:
                continue
            frac = age_ms / self.PULSE_MS
            r = 8.0 + frac * 26.0
            lw = max(1, int(round(3 * (1 - frac))))
            self.canvas.create_oval(
                p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r,
                outline=p["color"], width=lw, tags="moving",
            )
            still.append(p)
        self._pulses = still

        max_total = max((r["total"] for r in self.routes), default=0.0)
        if all_done:
            arrivals = ", ".join(
                f"{r['label']}={r['arrived']:.2f}" for r in self.routes
            )
            self.status.config(
                text=f"All units arrived at sim T = {max_total:.2f}  ({arrivals})"
            )
        else:
            shown = min(sim_t, max_total)
            self.status.config(text=f"Simulated time: {shown:.2f} / {max_total:.2f}")

        # Keep ticking until the last pulse has faded out too.
        if all_done and not self._pulses:
            self._anim_running = False
            return
        self.after(self.TICK_MS, self._tick)

    @staticmethod
    def _fade_color(hex_color: str, factor: float) -> str:
        """Blend `hex_color` toward white by (1 - factor). factor=1 keeps it."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r + (255 - r) * (1 - factor))
        g = int(g + (255 - g) * (1 - factor))
        b = int(b + (255 - b) * (1 - factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _position_on_route(
        self, route: Dict[str, Any], t: float
    ) -> Tuple[float, float]:
        """Linear interpolation along the route's polyline at sim-time `t`."""
        acc = 0.0
        for i, d in enumerate(route["durs"]):
            if t < acc + d:
                frac = 0.0 if d <= 0 else (t - acc) / d
                x1, y1 = route["pts"][i]
                x2, y2 = route["pts"][i + 1]
                return (x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac)
            acc += d
        return route["pts"][-1]

    # ------------------------------------------------------------- Draw ----
    def _redraw(
        self,
        cost: List[List[float]] | None = None,
        assignment: Assignment | None = None,
    ) -> None:
        c = self.canvas
        c.delete("all")

        # ---- Roads (pass 1: lines) -------------------------------------
        # Collect open-edge midpoints so we can draw labels in a second
        # pass that sits on top of every line.
        label_jobs: List[Tuple[float, float, float]] = []
        for r in range(self.rows):
            for cc in range(self.cols):
                here = (r, cc)
                for nb in ((r, cc + 1), (r + 1, cc)):
                    nr, nc = nb
                    if nr >= self.rows or nc >= self.cols:
                        continue
                    x1, y1 = self._xy(here)
                    x2, y2 = self._xy(nb)
                    closed = (here, nb) in self.closures or (nb, here) in self.closures
                    if closed:
                        c.create_line(x1, y1, x2, y2, fill="#cccccc", dash=(2, 4))
                        continue
                    w = self._edge_weight(node_id(here, self.cols),
                                          node_id(nb, self.cols))
                    if w is None:
                        continue
                    color, width = self._weight_style(w)
                    c.create_line(x1, y1, x2, y2, fill=color, width=width)
                    label_jobs.append(((x1 + x2) / 2, (y1 + y2) / 2, w))

        # ---- Roads (pass 2: weight labels on top) ----------------------
        for mx, my, w in label_jobs:
            text = f"{w:.1f}"
            # White background pill so the number is legible across any road colour.
            pad_x, pad_y = 10, 7
            c.create_rectangle(
                mx - pad_x, my - pad_y, mx + pad_x, my + pad_y,
                fill="#ffffff", outline="",
            )
            c.create_text(
                mx, my, text=text,
                fill="#222222", font=("Helvetica", 10, "bold"),
            )

        # Intersections
        for r in range(self.rows):
            for cc in range(self.cols):
                x, y = self._xy((r, cc))
                c.create_oval(x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R,
                              fill="#eeeeee", outline="#888")

        # ---- Unit base shapes (drawn UNDER ribbons so a route that
        #      transits a unit's cell isn't hidden by the marker). The
        #      identifying badge + label is re-drawn on top further down.
        for coord in self.ambulance_coords:
            x, y = self._xy(coord)
            c.create_rectangle(x - UNIT_R, y - UNIT_R, x + UNIT_R, y + UNIT_R,
                               fill="#1f77b4", outline="black")
        for coord in self.emergency_coords:
            x, y = self._xy(coord)
            c.create_oval(x - UNIT_R, y - UNIT_R, x + UNIT_R, y + UNIT_R,
                          fill="#d62728", outline="black")

        # Assignment paths -- draw each route as a ribbon offset
        # perpendicular to its segments, so overlapping routes appear as
        # parallel ribbons instead of hiding each other.
        if assignment is not None:
            K = len(assignment)
            spacing = 6  # pixels between parallel ribbons
            ribbon_w = 3
            for k, (i, j) in enumerate(assignment):
                color = PATH_COLORS[k % len(PATH_COLORS)]
                # Symmetric offset around 0: e.g. K=3 -> [-1, 0, +1] * spacing.
                offset = (k - (K - 1) / 2.0) * spacing

                src = node_id(self.ambulance_coords[i], self.cols)
                dst = node_id(self.emergency_coords[j], self.cols)
                prev = self._prev_cache.get(src)
                if prev is None:  # defensive: cache miss => recompute.
                    _, prev = dijkstra_with_paths(self.city, src, targets=[dst])
                path_nodes = reconstruct_path(prev, dst)
                if len(path_nodes) < 2:
                    continue

                # Convert node ids back to canvas coordinates.
                xy: List[Tuple[float, float]] = []
                for n in path_nodes:
                    pr, pc = divmod(cast(int, n), self.cols)
                    xy.append(self._xy((pr, pc)))

                # Draw segment-by-segment with a per-segment perpendicular
                # offset so the route hugs the road but shifts sideways.
                for (x1, y1), (x2, y2) in zip(xy, xy[1:]):
                    dx, dy = x2 - x1, y2 - y1
                    length = (dx * dx + dy * dy) ** 0.5 or 1.0
                    # Perpendicular unit vector (rotate 90 deg).
                    nx, ny = -dy / length, dx / length
                    ox, oy = nx * offset, ny * offset
                    c.create_line(
                        x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                        fill=color, width=ribbon_w,
                        capstyle=tk.ROUND,
                    )

        # ---- Unit identity badges (small disk + label) drawn ON TOP of
        #      ribbons so each ambulance / emergency stays identifiable
        #      even when routes pass through its cell.
        badge_r = 9
        for i, coord in enumerate(self.ambulance_coords):
            x, y = self._xy(coord)
            c.create_oval(x - badge_r, y - badge_r, x + badge_r, y + badge_r,
                          fill="#1f77b4", outline="white", width=2)
            c.create_text(x, y, text=f"A{i}", fill="white",
                          font=("Helvetica", 10, "bold"))
        for j, coord in enumerate(self.emergency_coords):
            x, y = self._xy(coord)
            c.create_oval(x - badge_r, y - badge_r, x + badge_r, y + badge_r,
                          fill="#d62728", outline="white", width=2)
            c.create_text(x, y, text=f"E{j}", fill="white",
                          font=("Helvetica", 10, "bold"))
            # Tiny priority chip above the badge (P5 = most urgent).
            if j < len(self.all_emergencies):
                pri = self.all_emergencies[j].priority
                c.create_text(x, y - badge_r - 9,
                              text=f"P{pri}", fill="#a02020",
                              font=("Helvetica", 9, "bold"))

    def _xy(self, coord: Coord) -> Tuple[float, float]:
        r, c = coord
        return PADDING + c * CELL_PX, PADDING + r * CELL_PX

    def _edge_weight(self, u: int, v: int) -> float | None:
        # O(1) lookup via the precomputed adjacency dict (built in
        # regenerate_city). Falls back to a linear scan if the cache
        # has not been populated yet (e.g. during very early init).
        w = self._edge_w.get((u, v))
        if w is not None:
            return w
        for nb, w in self.city.neighbors(u):
            if nb == v:
                return w
        return None

    def _weight_style(self, w: float) -> Tuple[str, int]:
        """Map an edge weight to (hex colour, line width).

        Light grey = fastest road; dark red = slowest road.
        """
        span = self.w_max - self.w_min
        t = 0.0 if span <= 1e-9 else (w - self.w_min) / span
        r1, g1, b1 = 0xD9, 0xD9, 0xD9   # fast
        r2, g2, b2 = 0x8B, 0x10, 0x10   # slow
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        width = 1 + int(round(2 * t))
        return f"#{r:02x}{g:02x}{b:02x}", width

    def _set_text(self, widget: tk.Text, content: str) -> None:
        """Write into a read-only Text widget."""
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        if content:
            widget.insert(tk.END, content)
        widget.config(state="disabled")

    # --------------------------------------------------------- Side panel ----
    def _show_matrix(
        self,
        cost: List[List[float]],
        dispatched_indices: List[int],
    ) -> None:
        if not cost or not cost[0]:
            self._set_text(self.matrix_box, "(no dispatchable emergencies)")
            return
        # Column header uses the *original* emergency ids (E0/E1/...) so
        # the user can match it against the canvas badges.
        header = "       " + " ".join(f"E{dispatched_indices[j]:>2}   "
                                      for j in range(len(cost[0])))
        lines = [header]
        for i, row in enumerate(cost):
            cells = " ".join(
                "  INF  " if v >= UNREACHABLE else f"{v:6.2f} " for v in row
            )
            lines.append(f"A{i:>2}  {cells}")
        self._set_text(self.matrix_box, "\n".join(lines) + "\n")

    def _show_assignment(
        self,
        cost: List[List[float]],
        assignment_local: Assignment,
        dispatched_indices: List[int],
    ) -> None:
        if not assignment_local:
            self._set_text(self.assignment_box, "(no assignment)")
        else:
            lines = []
            for i, j in assignment_local:
                v = cost[i][j]
                marker = "  [UNREACHABLE]" if v >= UNREACHABLE else ""
                eid = dispatched_indices[j]
                lines.append(
                    f"Ambulance A{i} -> Emergency E{eid}   cost = {v:6.2f}{marker}"
                )
            total = total_cost(cost, assignment_local)
            lines.append(f"\nTotal travel time: {total:.2f}")

            # ---- Side-by-side strategy comparison footer ----------------
            rt = self._random_total
            gt = self._greedy_total
            if rt is not None and gt is not None:
                lines.append("")
                lines.append("Strategy comparison (same cost matrix):")
                active = "greedy" if self.strategy.get().startswith("greedy") else "random"
                r_mark = "  <-- active" if active == "random" else ""
                g_mark = "  <-- active" if active == "greedy" else ""
                lines.append(f"  random : {rt:7.2f}{r_mark}")
                lines.append(f"  greedy : {gt:7.2f}{g_mark}")
                # Only show delta if neither solution touched an UNREACHABLE cell.
                if rt < UNREACHABLE and gt < UNREACHABLE and rt > 0:
                    diff = rt - gt
                    pct = diff / rt * 100
                    if diff >= 0:
                        lines.append(f"  greedy saves {diff:+.2f}  ({pct:+.1f}% vs random)")
                    else:
                        lines.append(f"  random beat greedy by {-diff:.2f}  ({-pct:.1f}%)")

            self._set_text(self.assignment_box, "\n".join(lines) + "\n")
        strat = self.strategy.get()
        self.status.config(
            text=f"Strategy: {strat}. TODO: replace with Hungarian (Algorithm B) "
                 "via dispatcher.assignment.solve_assignment()."
        )

    def _show_queue(self, queued: List[Emergency]) -> None:
        if not queued:
            self._set_text(self.queue_box, "(none — fleet covers all calls)")
            return
        lines = [f"{len(queued)} call(s) deferred to next cycle:"]
        for e in queued:
            # Look up the original E-id so it matches the canvas badges.
            try:
                eid = next(
                    i for i, c in enumerate(self.emergency_coords) if c == e.coord
                )
                label = f"E{eid}"
            except StopIteration:
                label = "?"
            lines.append(f"  {label}  priority={e.priority}  call_id={e.call_id}")
        self._set_text(self.queue_box, "\n".join(lines) + "\n")


def main() -> None:
    DispatcherApp().mainloop()


if __name__ == "__main__":
    main()
