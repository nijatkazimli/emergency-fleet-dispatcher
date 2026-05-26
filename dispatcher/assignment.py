"""Assignment layer for the dispatcher.

Re-exports Algorithm B (`hungarian.solve_assignment`) under the stable
`dispatcher.assignment` namespace the UI imports from, alongside two
baselines (`random_assignment`, `greedy_assignment`) used by the live
strategy-comparison panel.

Contract of `solve_assignment(cost_matrix)` (delivered by the `hungarian`
package):

    Input  : N x M list[list[float]] of finite non-negative travel times.
             Unreachable pairs are filled with `routing.UNREACHABLE` (= 1e9),
             never `math.inf`.
    Output : list[tuple[int, int]] of length `min(N, M)` whose selected
             cells sum to the global minimum, with no duplicate rows or cols.
"""

from __future__ import annotations

import random
from typing import List, Tuple

Assignment = List[Tuple[int, int]]


def random_assignment(
    cost_matrix: List[List[float]],
    seed: int | None = None,
) -> Assignment:
    """Random valid 1:1 matching. Used as the worst-case baseline in the
    UI's strategy-comparison panel.
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


# Re-export the Hungarian solver from the top-level `hungarian` package
# (Algorithm B). Keeps the existing `dispatcher.assignment.solve_assignment`
# import path stable for the UI and tests.
from hungarian import solve_assignment  # noqa: E402, F401
