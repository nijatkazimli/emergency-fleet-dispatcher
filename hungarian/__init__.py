"""Hungarian / Kuhn-Munkres assignment solver (Algorithm B).

Public API:
    solve_assignment -- optimal 1:1 minimum-cost bipartite matching.
    Assignment       -- type alias for list[tuple[int, int]].
"""

from .hungarian import Assignment, solve_assignment

__all__ = ["Assignment", "solve_assignment"]
