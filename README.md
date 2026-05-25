# Algorithm A — Routing Engine

This is **Partner 1's** half of the *Intelligent Emergency Fleet Dispatcher*.
It owns the road network, runs **Multi-Source Dijkstra** with a **binary
min-heap**, and produces the **N × M cost matrix** that Partner 2's Hungarian
algorithm consumes.

## Layout

| File | Purpose |
| --- | --- |
| [routing/graph.py](routing/graph.py) | Adjacency-list weighted graph. |
| [routing/dijkstra.py](routing/dijkstra.py) | Single-source + multi-source Dijkstra (min-heap). |
| [routing/city_generator.py](routing/city_generator.py) | Mock grid city with optional road closures. |
| [routing/cost_matrix.py](routing/cost_matrix.py) | Builds the handoff matrix; defines the `UNREACHABLE` sentinel. |
| [demo.py](demo.py) | End-to-end demo printing the matrix. |
| [tests/test_routing.py](tests/test_routing.py) | Unit tests covering correctness + edge cases. |

## Handoff contract (agreed with Partner 2)

`build_cost_matrix(graph, ambulances, emergencies)` returns a
`list[list[float]]` such that:

- **Rows** = ambulances, in the order supplied.
- **Columns** = emergencies, in the order supplied.
- Every cell is a **finite, non-negative `float`** (travel time).
- Unreachable pairs are filled with `routing.UNREACHABLE = 1e9` so the
  Hungarian algorithm never has to handle `math.inf`.

```python
from routing import build_cost_matrix
from routing.city_generator import build_grid_city, node_id

city = build_grid_city(rows=6, cols=6, seed=42)
ambulances  = [node_id((0, 0), 6), node_id((5, 0), 6), node_id((0, 5), 6)]
emergencies = [node_id((5, 5), 6), node_id((3, 3), 6), node_id((2, 4), 6)]

cost = build_cost_matrix(city, ambulances, emergencies)  # 3 x 3 matrix
# -> hand `cost` directly to Partner 2's Hungarian solver.
```

## Complexity

- **Dijkstra (per source):** $O((V + E) \log V)$ — every node is pushed at most
  once per improving relaxation, each heap op is $O(\log V)$.
- **Multi-source pass:** $O(S \cdot (V + E) \log V)$ where $S$ is the number
  of ambulances. This is the dominant term in the whole system because
  $V$ (intersections) ≫ $K$ (concurrent emergencies).
- **Space:** $O(V)$ for the distance map + heap per Dijkstra run, plus
  $O(S \cdot T)$ for the output matrix.

The chosen data structure is a **binary min-heap** (`heapq`). It gives
$O(\log V)$ extract-min, avoiding the $O(V)$ scan that a naive
array-based implementation would require, which is what would otherwise
dominate the runtime on large city graphs.

## Edge cases handled

1. **Unreachable nodes / road closures** — Dijkstra simply never relaxes
   them; `build_cost_matrix` rewrites the resulting `inf` to `UNREACHABLE`.
2. **Fleet shortage (N > M)** — out of scope for Algorithm A; the matrix is
   still well-defined for whatever subset Partner 2 (or the dispatcher
   front-end) decides to pass in.
3. **Negative weights** — rejected at `add_edge` time (Dijkstra requires
   non-negative weights).

## Run it

```bash
python3 demo.py                                      # CLI: prints the cost matrix
python3 -m unittest tests/test_routing.py tests/test_triage.py -v
python3 run_ui.py                                    # Tk demo (cross-platform)
```

The Tk demo (in [dispatcher/ui.py](dispatcher/ui.py)) wires Algorithm A,
the triage layer, and the placeholder assignment together and animates
the dispatched ambulances along their actual shortest paths. The
**Strategy** dropdown lets the user switch between the random placeholder,
the greedy baseline, and — once it lands — Partner 2's Hungarian solver.

---

## For Partner 2 — Algorithm B (Hungarian / Kuhn–Munkres)

Everything above the assignment layer is already in place and consuming
a placeholder. Your job is to drop a real Hungarian solver into
[dispatcher/assignment.py](dispatcher/assignment.py) and flip one line
in the UI to expose it. **No other file needs to change.**

### What to implement

```python
# dispatcher/assignment.py
def solve_assignment(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Optimal 1:1 ambulance -> emergency matching.

    Input  : N x M matrix of finite non-negative floats.
             Unreachable pairs are marked with routing.UNREACHABLE (= 1e9),
             NOT math.inf — keep the solver purely numeric.
    Output : list of (row, col) pairs of length min(N, M).
             Sum of cost_matrix[i][j] over the returned pairs MUST be
             the global minimum. No row or col index may repeat.
    """
    ...
```

### Hard contract (the existing pipeline relies on these)

1. **Length** of the returned list is exactly `min(N, M)`.
2. **No duplicates** in row indices nor in column indices.
3. **Indices are valid** into the input matrix — the caller maps them
   back to ambulance / emergency identities by position.
4. **Numeric only.** Treat `UNREACHABLE = 1e9` like any other big finite
   cost; the solver will naturally avoid those cells. Do NOT special-case
   `math.inf`; it never appears in the input.
5. **Rectangular input.** N may differ from M. The triage layer caps M
   at N before the matrix is built, but your solver should still handle
   M < N safely (just match the M cheapest ambulances).

### Wiring it into the demo

When `solve_assignment` is ready, do exactly two edits in
[dispatcher/ui.py](dispatcher/ui.py):

```python
# 1. Import it next to the placeholders
from .assignment import (
    Assignment, greedy_assignment, random_assignment, solve_assignment, total_cost,
)

# 2. Branch on the dropdown in dispatch() — add a third arm:
if self.strategy.get().startswith("hungarian"):
    assignment_local = solve_assignment(cost)
elif self.strategy.get().startswith("greedy"):
    assignment_local = greedy_local
else:
    assignment_local = random_local
```

…and add `"hungarian (optimal)"` to the `values=[...]` list of the
Strategy combobox in `_build_widgets`. That's it — the cost matrix,
triage, animation, side-by-side comparison panel, and arrival pulse
all light up automatically for the new strategy.

### Suggested file layout

```
dispatcher/
    hungarian.py        <- your implementation goes here
    assignment.py       <- import + re-export solve_assignment from hungarian
tests/
    test_hungarian.py   <- correctness + edge-case tests
```

Keep the algorithm itself in `hungarian.py` so it's reviewable in
isolation, then in `assignment.py` add a single line:

```python
from .hungarian import solve_assignment  # noqa: F401  (re-export)
```

### Tests you should add (`tests/test_hungarian.py`)

Cover at minimum:

1. **Square case** — a 3×3 hand-crafted matrix where the optimum is
   known; assert both the returned pairs and `total_cost`.
2. **Rectangular N > M** and **N < M** — assert `len == min(N, M)`,
   no duplicate indices, and that the result still beats greedy on a
   matrix specifically designed to fool greedy.
3. **`UNREACHABLE` handling** — feed a matrix where the only optimum
   touches one `UNREACHABLE` cell; assert the solver still returns
   something valid (no crash, no `inf`, length still correct).
4. **Determinism / tie-break** — running twice on the same input gives
   the same total cost (the exact pairing may vary if you tie-break
   non-deterministically; that's fine, but the total must be invariant).
5. **Optimality vs greedy** (property test) — for, say, 20 random
   5×5 matrices, assert `total_cost(hungarian) <= total_cost(greedy)`.
   This is the single best guard against subtle bugs.

### Complexity target

The classic Kuhn–Munkres runs in **$O(n^3)$** where $n = \max(N, M)$,
which is fine for the city sizes the demo uses (up to 16×16 grids with
≤ 12 of each unit, so $n \le 12$). Any implementation that hits that
bound is acceptable. **Do not** implement the brute-force $O(n!)$
permutation search.

### Copilot prompt you can paste

If you want Copilot Chat to scaffold the solver for you, this prompt
gives it everything it needs:

> Implement the Hungarian / Kuhn–Munkres algorithm in `dispatcher/hungarian.py`
> as `solve_assignment(cost_matrix: list[list[float]]) -> list[tuple[int, int]]`.
> The input is an N×M matrix of finite non-negative floats; unreachable cells
> are filled with the sentinel `1e9` (imported as `routing.UNREACHABLE`), not
> `math.inf`. Return a list of `(row, col)` pairs of length `min(N, M)` whose
> selected cells sum to the global minimum, with no duplicate rows or columns.
> Handle rectangular inputs by padding to a square internally. Target $O(n^3)$
> time where $n = \max(N, M)$. Pure stdlib only — no numpy, no scipy. Then
> re-export it from `dispatcher/assignment.py` and add unit tests in
> `tests/test_hungarian.py` covering: a hand-checked 3×3 optimum, rectangular
> N≠M shapes, an `UNREACHABLE`-containing matrix, and a property test that
> asserts `total_cost(hungarian) <= total_cost(greedy)` on 20 random 5×5
> matrices.

### What's already done for you

You do **not** need to touch any of this — it's stable:

- `routing/` — graph, Dijkstra, cost-matrix handoff, all tested.
- `dispatcher/triage.py` — priority + FIFO triage for the N > M case.
- `dispatcher/ui.py` — full Tk UI with animation, comparison panel,
  arrival pulse, city-size control. It will pick up your strategy as
  soon as the two edits above are made.
- `dispatcher/assignment.py` already exports `random_assignment`,
  `greedy_assignment`, and `total_cost`; the comparison panel uses
  them as baselines, so your Hungarian result will be drawn against
  them automatically.

