"""Mock city generator: a 2D grid of intersections with optional road closures.

Each intersection is identified by an integer id = row * cols + col.
Edge weights model travel time (default = 1.0 per block, but you can pass
a `weight_fn` to introduce traffic).
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

from .graph import Graph

Coord = Tuple[int, int]


def node_id(coord: Coord, cols: int) -> int:
    r, c = coord
    return r * cols + c


def build_grid_city(
    rows: int,
    cols: int,
    closed_edges: Optional[List[Tuple[Coord, Coord]]] = None,
    weight_fn: Optional[Callable[[Coord, Coord], float]] = None,
    seed: Optional[int] = None,
) -> Graph:
    """Build a `rows x cols` grid graph (4-connected) with travel-time weights.

    Parameters
    ----------
    closed_edges : list of ((r1,c1),(r2,c2))
        Edges to omit (simulating road closures / isolated subgraphs).
    weight_fn : callable
        Returns the travel time between two adjacent cells. Defaults to a
        deterministic value of 1.0 (or a pseudo-random "traffic" weight in
        [1.0, 3.0] when `seed` is provided and `weight_fn` is None).
    """
    rng = random.Random(seed)
    if weight_fn is None:
        if seed is None:
            weight_fn = lambda a, b: 1.0
        else:
            weight_fn = lambda a, b: 1.0 + 2.0 * rng.random()

    closed = set()
    for a, b in closed_edges or []:
        closed.add(frozenset((a, b)))

    g = Graph(directed=False)
    for r in range(rows):
        for c in range(cols):
            g.add_node(node_id((r, c), cols))

    for r in range(rows):
        for c in range(cols):
            here = (r, c)
            for dr, dc in ((1, 0), (0, 1)):  # add each undirected edge once
                nr, nc = r + dr, c + dc
                if nr >= rows or nc >= cols:
                    continue
                there = (nr, nc)
                if frozenset((here, there)) in closed:
                    continue
                g.add_edge(
                    node_id(here, cols),
                    node_id(there, cols),
                    weight_fn(here, there),
                )
    return g
