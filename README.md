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
python demo.py
python -m unittest tests/test_routing.py
```
