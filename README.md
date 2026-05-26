# Intelligent Emergency Fleet Dispatcher

End-to-end demo built around two assignment-grade algorithms:

| Layer | Algorithm | Module |
| --- | --- | --- |
| **A — Routing** | Multi-source Dijkstra on a binary min-heap | [routing/](routing) |
| **B — Assignment** | Hungarian / Kuhn–Munkres ($O(n^3)$) | [hungarian/](hungarian) |
| Glue | Triage + Tk UI + greedy/random baselines | [dispatcher/](dispatcher) |

## Layout

| File | Purpose |
| --- | --- |
| [routing/graph.py](routing/graph.py) | Adjacency-list weighted graph. |
| [routing/dijkstra.py](routing/dijkstra.py) | Single-source + multi-source Dijkstra (min-heap). |
| [routing/city_generator.py](routing/city_generator.py) | Mock grid city with optional road closures. |
| [routing/cost_matrix.py](routing/cost_matrix.py) | Builds the handoff matrix; defines the `UNREACHABLE` sentinel. |
| [hungarian/hungarian.py](hungarian/hungarian.py) | Hungarian / Kuhn–Munkres optimal assignment. |
| [dispatcher/triage.py](dispatcher/triage.py) | Priority + FIFO triage for the N > M overflow case. |
| [dispatcher/assignment.py](dispatcher/assignment.py) | Re-exports `solve_assignment` and ships `random`/`greedy` baselines. |
| [dispatcher/ui.py](dispatcher/ui.py) | Tk demo with animation and a live strategy-comparison panel. |
| [demo.py](demo.py) | CLI demo printing the cost matrix. |
| [tests/](tests) | Unit tests for routing, triage, and the Hungarian solver. |

## Algorithm A — Routing

### Handoff contract

`build_cost_matrix(graph, ambulances, emergencies)` returns a
`list[list[float]]` such that:

- **Rows** = ambulances, in the order supplied.
- **Columns** = emergencies, in the order supplied.
- Every cell is a **finite, non-negative `float`** (travel time).
- Unreachable pairs are filled with `routing.UNREACHABLE = 1e9` so the
  Hungarian solver never has to handle `math.inf`.

```python
from routing import build_cost_matrix
from routing.city_generator import build_grid_city, node_id

city = build_grid_city(rows=6, cols=6, seed=42)
ambulances  = [node_id((0, 0), 6), node_id((5, 0), 6), node_id((0, 5), 6)]
emergencies = [node_id((5, 5), 6), node_id((3, 3), 6), node_id((2, 4), 6)]

cost = build_cost_matrix(city, ambulances, emergencies)  # 3 x 3 matrix
```

### Complexity

- **Dijkstra (per source):** $O((V + E) \log V)$ — every node is pushed at
  most once per improving relaxation, each heap op is $O(\log V)$.
- **Multi-source pass:** $O(S \cdot (V + E) \log V)$ where $S$ is the number
  of ambulances. This is the dominant term on large city graphs.
- **Space:** $O(V)$ for the distance map + heap per Dijkstra run, plus
  $O(S \cdot T)$ for the output matrix.

The chosen structure is a **binary min-heap** (`heapq`) — $O(\log V)$
extract-min, avoiding the $O(V)$ scan of a naive array-based version.

### Edge cases handled

1. **Unreachable nodes / road closures** — Dijkstra never relaxes them;
   `build_cost_matrix` rewrites the resulting `inf` to `UNREACHABLE`.
2. **Negative weights** — rejected at `add_edge` time.
3. **Fleet shortage (N > M)** — the triage layer in `dispatcher/triage.py`
   caps M at N before the matrix is built; the matrix itself stays
   well-defined for any subset that is passed in.

## Algorithm B — Hungarian / Kuhn–Munkres

`hungarian.solve_assignment(cost_matrix)` returns the optimal 1:1 matching:

```python
from hungarian import solve_assignment

pairs = solve_assignment(cost)   # list[(row, col)] of length min(N, M)
```

### Contract

1. **Length** of the returned list is exactly `min(N, M)`.
2. **No duplicates** in row indices nor in column indices.
3. **Sum** of `cost[i][j]` over the returned pairs is the **global minimum**.
4. **Numeric only.** `UNREACHABLE = 1e9` is treated like any other large
   finite cost; the solver naturally avoids it when possible.
5. **Rectangular input** (N ≠ M) is handled by padding to a square with
   zeros internally — safe because all real costs are non-negative, so
   the padded dummy rows/cols are filtered out of the final assignment
   without disturbing optimality.

### Complexity

$O(n^3)$ time, $O(n^2)$ space, where $n = \max(N, M)$ — the classic
potential-method (Kuhn–Munkres) implementation with a sentinel column.
For the demo's city sizes ($n \le 12$) this is instantaneous.

### Edge cases covered by [tests/test_hungarian.py](tests/test_hungarian.py)

- Empty matrix, 1×1 matrix, all-zeros and all-equal matrices.
- Hand-checked 3×3 with a known optimum.
- Rectangular shapes in both directions (N > M and N < M).
- Matrices containing `UNREACHABLE` cells — including rows that are
  entirely unreachable, and matrices where any complete matching must
  use at least one `UNREACHABLE` cell.
- Determinism (same input → same total cost).
- Property test: Hungarian total ≤ Greedy total on 20 random 5×5
  matrices, with a strict-better count guard against degenerate solvers.

## Run it

```bash
python3 demo.py                                                # CLI: prints the cost matrix
python3 -m unittest tests/test_routing.py tests/test_triage.py tests/test_hungarian.py -v
python3 run_ui.py                                              # Tk demo (cross-platform)
```

The Tk demo (in [dispatcher/ui.py](dispatcher/ui.py)) wires Algorithm A,
the triage layer, and all three assignment strategies together and
animates the dispatched ambulances along their actual shortest paths.
The **Strategy** dropdown switches between `random`, `greedy`, and
`hungarian`; the side panel shows every strategy's total cost on the
same matrix so the optimality gap is visible at a glance.
