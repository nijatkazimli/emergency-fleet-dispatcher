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
from typing import Hashable, List, Sequence

from .dijkstra import multi_source_costs
from .graph import Graph

Node = Hashable

# Large, finite sentinel: bigger than any plausible real path cost in our
# graphs, but small enough that sums of a handful of them won't overflow.
UNREACHABLE: float = 1e9


def build_cost_matrix(
    graph: Graph,
    ambulances: Sequence[Node],
    emergencies: Sequence[Node],
    unreachable: float = UNREACHABLE,
) -> List[List[float]]:
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

    Returns
    -------
    list[list[float]]
        Matrix of shape (len(ambulances), len(emergencies)).
    """
    raw = multi_source_costs(graph, ambulances, emergencies)
    return [
        [unreachable if math.isinf(c) else float(c) for c in row]
        for row in raw
    ]
