"""Builds the N x M cost matrix that is handed off to Algorithm B.

INTERFACE CONTRACT (agreed with Partner 2 / Hungarian algorithm):
  * Output is a plain Python `list[list[float]]`.
  * Rows correspond to AMBULANCES (sources), in the order supplied.
  * Columns correspond to EMERGENCIES (targets), in the order supplied.
  * Every cell is a finite, non-negative float.
  * Unreachable (source, target) pairs are filled with `UNREACHABLE`
    (a very large finite sentinel) so the Hungarian algorithm never has
    to reason about `math.inf` and will naturally avoid those assignments.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union, overload

from .dijkstra import dijkstra_with_paths, multi_source_costs
from .graph import Graph

Node = Hashable
PrevMap = Dict[Node, Optional[Node]]

# Large, finite sentinel: bigger than any plausible real path cost in our
# graphs, but small enough that sums of a handful of them won't overflow.
UNREACHABLE: float = 1e9


@overload
def build_cost_matrix(
    graph: Graph,
    ambulances: Sequence[Node],
    emergencies: Sequence[Node],
    unreachable: float = ...,
) -> List[List[float]]: ...
@overload
def build_cost_matrix(
    graph: Graph,
    ambulances: Sequence[Node],
    emergencies: Sequence[Node],
    unreachable: float = ...,
    *,
    return_paths: bool,
) -> Union[List[List[float]], Tuple[List[List[float]], Dict[Node, PrevMap]]]: ...


def build_cost_matrix(
    graph: Graph,
    ambulances: Sequence[Node],
    emergencies: Sequence[Node],
    unreachable: float = UNREACHABLE,
    *,
    return_paths: bool = False,
) -> Union[List[List[float]], Tuple[List[List[float]], Dict[Node, PrevMap]]]:
    """Compute the cost matrix consumed by the Hungarian algorithm.

    Parameters
    ----------
    graph : Graph
        Road network (nodes = intersections, edge weights = travel time).
    ambulances : sequence of node IDs
        Current location of each available ambulance.
    emergencies : sequence of node IDs
        Location of each active emergency.
    unreachable : float
        Sentinel used in place of +inf for unreachable pairs.
    return_paths : bool, keyword-only
        If True, also return a `prev_by_source` map so callers can
        reconstruct the actual shortest paths without re-running Dijkstra.
        Defaults to False so the partner contract (`-> list[list[float]]`)
        stays untouched.

    Complexity
    ----------
    O(N * (K + E_K) log K) where N = len(ambulances) and K is the number
    of nodes Dijkstra has to settle before reaching every emergency. With
    early termination, K is typically much smaller than V on large graphs.

    Returns
    -------
    list[list[float]]                                        # default
    (list[list[float]], dict[Node, dict[Node, Optional[Node]]])  # return_paths=True
    """
    targets = list(emergencies)
    target_set = set(targets)
    cost: List[List[float]] = []
    prev_by_source: Dict[Node, PrevMap] = {}
    for src in ambulances:
        dist, prev = dijkstra_with_paths(graph, src, targets=target_set)
        row = [
            unreachable if math.isinf(dist.get(t, math.inf)) else float(dist[t])
            for t in targets
        ]
        cost.append(row)
        if return_paths:
            prev_by_source[src] = prev
    if return_paths:
        return cost, prev_by_source
    return cost


# Re-export so legacy callers / docs that referenced multi_source_costs
# can still find it through this module.
__all__ = ["UNREACHABLE", "build_cost_matrix", "multi_source_costs"]
