"""Assignment layer — placeholder for Algorithm B (Hungarian / Kuhn-Munkres).

==============================================================================
TODO (Partner 2): replace `random_assignment` with the real Hungarian solver.
==============================================================================

Required signature
------------------
    def solve_assignment(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:

Input
-----
    cost_matrix : list[list[float]]
        Produced by `routing.build_cost_matrix(...)`.
        Shape: N x M, where
            N = number of ambulances (rows)
            M = number of emergencies (columns)
        Every cell is a finite, non-negative float.
        Unreachable pairs are filled with `routing.UNREACHABLE` (= 1e9),
        NOT with `math.inf`, so the solver can stay purely numeric.

Output
------
    assignments : list[tuple[int, int]]
        A 1:1 bipartite matching that minimizes the total travel time.
        Each tuple is (ambulance_row_index, emergency_col_index).
        len(assignments) == min(N, M).
        Indices must be valid into the original input lists, so the caller
        can map them back to ambulance / emergency identities.

Contract guarantees the caller relies on
----------------------------------------
    * Sum of `cost_matrix[i][j]` over the returned pairs is the global minimum.
    * No row index and no column index is repeated.
    * If a pairing is forced onto an UNREACHABLE cell (because N > number of
      reachable emergencies), that's fine — the caller can detect it by
      checking the cell value against `routing.UNREACHABLE`.

Tie-breaking
------------
    When several optimal matchings exist, any one of them is acceptable.

Until Algorithm B is wired in, the UI calls `random_assignment` so the
demo flow still runs end-to-end. A "Greedy" baseline is also provided so
the presentation can contrast the (eventual) Hungarian result against a
naive dispatcher.
"""

from __future__ import annotations

import random
from typing import List, Tuple

Assignment = List[Tuple[int, int]]


def random_assignment(
    cost_matrix: List[List[float]],
    seed: int | None = None,
) -> Assignment:
    """Placeholder: returns a random valid 1:1 matching.

    Replace with `solve_assignment` (Hungarian) once Partner 2 ships it.
    """
    if not cost_matrix or not cost_matrix[0]:
        return []
    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0])
    k = min(n_rows, n_cols)

    rng = random.Random(seed)
    rows = list(range(n_rows))
    cols = list(range(n_cols))
    rng.shuffle(rows)
    rng.shuffle(cols)
    return list(zip(rows[:k], cols[:k]))


def greedy_assignment(cost_matrix: List[List[float]]) -> Assignment:
    """Naive baseline: repeatedly pick the globally cheapest remaining cell.

    Useful for the demo to contrast against the Hungarian result.
    Not optimal in general.
    """
    if not cost_matrix or not cost_matrix[0]:
        return []
    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0])
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs: Assignment = []
    k = min(n_rows, n_cols)
    while len(pairs) < k:
        best = None
        for i in range(n_rows):
            if i in used_rows:
                continue
            for j in range(n_cols):
                if j in used_cols:
                    continue
                c = cost_matrix[i][j]
                if best is None or c < best[0]:
                    best = (c, i, j)
        if best is None:
            break
        _, i, j = best
        used_rows.add(i)
        used_cols.add(j)
        pairs.append((i, j))
    return pairs


def total_cost(cost_matrix: List[List[float]], assignment: Assignment) -> float:
    return sum(cost_matrix[i][j] for i, j in assignment)


# Import and re-export the Hungarian solver from the hungarian module
from .hungarian import solve_assignment  # noqa: E402, F401
