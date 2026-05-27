"""Hungarian / Kuhn–Munkres algorithm for optimal assignment.

Solves the minimum-cost bipartite matching problem in O(n³) time where
n = max(N, M). Handles rectangular matrices by padding to square internally.
"""

from __future__ import annotations

from typing import List, Tuple


Assignment = List[Tuple[int, int]]


def solve_assignment(cost_matrix: list[list[float]]) -> Assignment:
    """Optimal 1:1 ambulance -> emergency matching using the Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : list[list[float]]
        N x M matrix of finite non-negative floats.
        Unreachable pairs are marked with 1e9 (UNREACHABLE sentinel),
        not math.inf.

    Returns
    -------
    list[tuple[int, int]]
        List of (row, col) pairs of length min(N, M).
        Sum of cost_matrix[i][j] over the returned pairs is minimized.
        No row or column index is repeated.

    Complexity
    ----------
    O(n³) where n = max(N, M).
    """
    if not cost_matrix or not cost_matrix[0]:
        return []

    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0])
    original_rows = n_rows
    original_cols = n_cols

    # Pad to square
    n = max(n_rows, n_cols)
    matrix = _pad_to_square(cost_matrix, n)

    # Run the Hungarian algorithm on the square matrix
    assignment = _hungarian(matrix)

    # Filter to only the original dimensions
    assignment = [
        (i, j) for i, j in assignment
        if i < original_rows and j < original_cols
    ]

    return assignment


def _pad_to_square(
    cost_matrix: list[list[float]],
    n: int,
) -> list[list[float]]:
    """Pad a rectangular matrix to n x n with zeros (non-edges have zero cost)."""
    padded = []
    for i in range(n):
        if i < len(cost_matrix):
            row = list(cost_matrix[i]) + [0.0] * (n - len(cost_matrix[i]))
        else:
            row = [0.0] * n
        padded.append(row)
    return padded


def _hungarian(cost_matrix: list[list[float]]) -> Assignment:
    """Run the Hungarian algorithm on a square n x n cost matrix.

    This is the core Munkres algorithm using augmenting paths.
    """
    n = len(cost_matrix)

    u = [0.0] * (n + 1)  # Dual variable for rows
    v = [0.0] * (n + 1)  # Dual variable for cols
    p = [0] * (n + 1)    # Row assignment: p[j] = i means col j is matched to row i
    way = [0] * (n + 1)  # Used to reconstruct augmenting paths

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (n + 1)
        used = [False] * (n + 1)

        while p[j0] != 0:
            i0 = p[j0]
            used[j0] = True
            delta = float('inf')
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment: Assignment = []
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment.append((p[j] - 1, j - 1))

    return assignment
