"""Weighted graph stored as an adjacency list.

Nodes can be any hashable value (we use ints for grid intersections).
Edges carry a non-negative weight representing travel time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Tuple

Node = Hashable
Edge = Tuple[Node, float]


class Graph:
    def __init__(self, directed: bool = False) -> None:
        self.directed = directed
        self._adj: Dict[Node, List[Edge]] = defaultdict(list)

    def add_node(self, node: Node) -> None:
        # Touch the defaultdict so isolated nodes still appear.
        _ = self._adj[node]

    def add_edge(self, u: Node, v: Node, weight: float) -> None:
        if weight < 0:
            raise ValueError("Dijkstra requires non-negative edge weights.")
        self._adj[u].append((v, weight))
        if not self.directed:
            self._adj[v].append((u, weight))
        else:
            _ = self._adj[v]  # ensure v is registered

    def neighbors(self, node: Node) -> Iterable[Edge]:
        return self._adj.get(node, ())

    def nodes(self) -> Iterable[Node]:
        return self._adj.keys()

    def __len__(self) -> int:
        return len(self._adj)

    def __contains__(self, node: Node) -> bool:
        return node in self._adj
