"""Routing Engine (Algorithm A) for the Intelligent Emergency Fleet Dispatcher.

Public API:
    Graph                -- weighted undirected/directed graph (adjacency list).
    dijkstra             -- single-source shortest paths (min-heap).
    multi_source_costs   -- shortest paths from many sources to many targets.
    build_cost_matrix    -- produces the N x M handoff matrix for Algorithm B.
    UNREACHABLE          -- sentinel used for unreachable (source, target) pairs.
"""

from .graph import Graph
from .dijkstra import dijkstra, multi_source_costs
from .cost_matrix import build_cost_matrix, UNREACHABLE

__all__ = [
    "Graph",
    "dijkstra",
    "multi_source_costs",
    "build_cost_matrix",
    "UNREACHABLE",
]
