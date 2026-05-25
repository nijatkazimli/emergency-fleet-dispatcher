"""Dijkstra's shortest-path algorithm using a binary min-heap.

Time:  O((V + E) log V) per source.
Space: O(V) for the distance map and the heap.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

from .graph import Graph

Node = Hashable


def dijkstra(graph: Graph, source: Node) -> Dict[Node, float]:
    """Return the shortest travel time from `source` to every reachable node.

    Unreachable nodes are simply absent from the returned dict
    (callers should treat missing entries as +inf).
    """
    if source not in graph:
        raise KeyError(f"Source node {source!r} is not in the graph.")

    dist: Dict[Node, float] = {source: 0.0}
    # Heap entries: (tentative_distance, node).
    heap: List[tuple] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        # Lazy-deletion: skip stale entries.
        if d > dist[u]:
            continue
        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def multi_source_costs(
    graph: Graph,
    sources: Iterable[Node],
    targets: Iterable[Node],
) -> List[List[float]]:
    """Run Dijkstra from each source and collect costs to every target.

    Returns a 2D list `C` where `C[i][j]` is the shortest travel time from
    `sources[i]` to `targets[j]`. Unreachable pairs are returned as math.inf
    here; `build_cost_matrix` is responsible for converting that to the
    sentinel value agreed with Algorithm B.

    Time:  O(S * (V + E) log V), where S = len(sources).
    Space: O(V) per Dijkstra run + O(S * T) for the output.
    """
    sources = list(sources)
    targets = list(targets)
    matrix: List[List[float]] = []
    for s in sources:
        dist = dijkstra(graph, s)
        matrix.append([dist.get(t, math.inf) for t in targets])
    return matrix


def dijkstra_with_paths(
    graph: Graph,
    source: Node,
    targets: Optional[Iterable[Node]] = None,
) -> Tuple[Dict[Node, float], Dict[Node, Optional[Node]]]:
    """Same as `dijkstra` but also returns a predecessor map for path reconstruction.

    Parameters
    ----------
    graph, source
        As in `dijkstra`.
    targets : iterable of nodes, optional
        If given, the search terminates as soon as **every reachable target**
        has been finalized. Distances/predecessors for unreachable or
        not-yet-settled targets are simply absent from the returned maps.
        This is a pure optimisation \u2014 results for any target that *is*
        reported are bit-identical to the full search.

    Returns
    -------
    (dist, prev)
        dist[node]  -- shortest travel time from `source` to `node`
        prev[node]  -- previous node on the shortest path (None for the source)

    Complexity
    ----------
    Full search:    O((V + E) log V).
    With targets:   O((K + E_K) log K), where K is the number of nodes
                    popped before all targets are settled (often << V).
    """
    if source not in graph:
        raise KeyError(f"Source node {source!r} is not in the graph.")

    remaining: Optional[set] = None
    if targets is not None:
        remaining = {t for t in targets if t != source}

    dist: Dict[Node, float] = {source: 0.0}
    prev: Dict[Node, Optional[Node]] = {source: None}
    heap: List[tuple] = [(0.0, source)]

    # Source counts as already-settled; bail out immediately if it was the
    # only target requested.
    if remaining is not None and not remaining:
        return dist, prev

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if remaining is not None and u in remaining:
            remaining.discard(u)
            if not remaining:
                return dist, prev
        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def reconstruct_path(prev: Dict[Node, Optional[Node]], target: Node) -> List[Node]:
    """Rebuild the node-by-node path from a `prev` map produced by `dijkstra_with_paths`.

    Returns an empty list if `target` is unreachable.
    """
    if target not in prev:
        return []
    path: List[Node] = []
    cur: Optional[Node] = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path
