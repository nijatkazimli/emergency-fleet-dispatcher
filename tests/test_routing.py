"""Smoke tests for the Routing Engine.

Run:  python -m unittest tests/test_routing.py
"""

from __future__ import annotations

import math
import unittest

from routing import Graph, build_cost_matrix, dijkstra, multi_source_costs, UNREACHABLE
from routing.city_generator import build_grid_city, node_id
from routing.dijkstra import dijkstra_with_paths, reconstruct_path


class DijkstraTests(unittest.TestCase):
    def test_simple_path(self):
        g = Graph()
        g.add_edge("A", "B", 1)
        g.add_edge("B", "C", 2)
        g.add_edge("A", "C", 10)
        d = dijkstra(g, "A")
        self.assertEqual(d["A"], 0)
        self.assertEqual(d["B"], 1)
        self.assertEqual(d["C"], 3)  # A->B->C beats A->C

    def test_unreachable_node_absent(self):
        g = Graph()
        g.add_edge("A", "B", 1)
        g.add_node("Z")  # isolated
        d = dijkstra(g, "A")
        self.assertNotIn("Z", d)

    def test_negative_weight_rejected(self):
        g = Graph()
        with self.assertRaises(ValueError):
            g.add_edge("A", "B", -1)


class CostMatrixTests(unittest.TestCase):
    def test_shape_and_row_col_orientation(self):
        city = build_grid_city(3, 3)
        ambulances = [node_id((0, 0), 3), node_id((2, 2), 3)]
        emergencies = [node_id((0, 2), 3), node_id((2, 0), 3), node_id((1, 1), 3)]
        m = build_cost_matrix(city, ambulances, emergencies)
        self.assertEqual(len(m), len(ambulances))            # rows = ambulances
        self.assertTrue(all(len(r) == len(emergencies) for r in m))  # cols = emergencies
        # Manhattan distance on unit-weight grid:
        self.assertEqual(m[0][0], 2)  # (0,0) -> (0,2)
        self.assertEqual(m[0][1], 2)  # (0,0) -> (2,0)
        self.assertEqual(m[0][2], 2)  # (0,0) -> (1,1)
        self.assertEqual(m[1][0], 2)  # (2,2) -> (0,2)
        self.assertEqual(m[1][1], 2)  # (2,2) -> (2,0)
        self.assertEqual(m[1][2], 2)  # (2,2) -> (1,1)

    def test_unreachable_uses_sentinel(self):
        # Two disconnected 1x2 strips.
        g = Graph()
        g.add_edge(0, 1, 1.0)
        g.add_edge(2, 3, 1.0)
        m = build_cost_matrix(g, [0], [1, 2, 3])
        self.assertEqual(m[0][0], 1.0)
        self.assertEqual(m[0][1], UNREACHABLE)
        self.assertEqual(m[0][2], UNREACHABLE)
        # Sentinel is finite (Hungarian-friendly).
        self.assertFalse(math.isinf(m[0][1]))

    def test_raw_multi_source_keeps_inf(self):
        g = Graph()
        g.add_edge(0, 1, 1.0)
        g.add_node(2)
        raw = multi_source_costs(g, [0], [1, 2])
        self.assertEqual(raw[0][0], 1.0)
        self.assertTrue(math.isinf(raw[0][1]))


class EarlyTerminationTests(unittest.TestCase):
    def test_targets_results_match_full_search(self):
        # Distances to requested targets must be bit-identical whether we
        # ask Dijkstra to run to completion or to bail out early.
        city = build_grid_city(6, 6, seed=42)
        src = node_id((0, 0), 6)
        tgts = [node_id((5, 5), 6), node_id((2, 3), 6), node_id((4, 1), 6)]
        full_d, _ = dijkstra_with_paths(city, src)
        part_d, _ = dijkstra_with_paths(city, src, targets=tgts)
        for t in tgts:
            self.assertEqual(part_d[t], full_d[t])

    def test_targets_can_skip_unreachable(self):
        # An unreachable target must not prevent early termination once
        # the reachable ones are settled.
        g = Graph()
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 2, 1.0)
        g.add_node(99)  # isolated
        dist, prev = dijkstra_with_paths(g, 0, targets=[2, 99])
        self.assertEqual(dist[2], 2.0)
        self.assertNotIn(99, dist)
        self.assertEqual(reconstruct_path(prev, 2), [0, 1, 2])

    def test_targets_explores_fewer_nodes(self):
        # On a large grid, targeted search must settle strictly fewer
        # nodes than a full traversal (the whole point of the option).
        city = build_grid_city(10, 10, seed=7)
        src = node_id((0, 0), 10)
        tgts = [node_id((1, 1), 10)]
        full_d, _ = dijkstra_with_paths(city, src)
        part_d, _ = dijkstra_with_paths(city, src, targets=tgts)
        self.assertLess(len(part_d), len(full_d))


class CostMatrixPathsTests(unittest.TestCase):
    def test_return_paths_keeps_default_matrix_intact(self):
        # The default (positional) call must still return ONLY the
        # cost matrix -- the partner's contract is unchanged.
        city = build_grid_city(3, 3)
        a = [node_id((0, 0), 3)]
        e = [node_id((2, 2), 3)]
        matrix_only = build_cost_matrix(city, a, e)
        self.assertIsInstance(matrix_only, list)
        self.assertEqual(matrix_only[0][0], 4)

    def test_return_paths_yields_reconstructable_prev_maps(self):
        city = build_grid_city(4, 4, seed=1)
        ambs = [node_id((0, 0), 4), node_id((3, 3), 4)]
        emes = [node_id((0, 3), 4), node_id((3, 0), 4)]
        cost, prev_by_source = build_cost_matrix(
            city, ambs, emes, return_paths=True
        )
        # Every requested ambulance got a prev-map.
        self.assertEqual(set(prev_by_source.keys()), set(ambs))
        # Reconstructed path cost equals the matrix cell.
        for i, src in enumerate(ambs):
            for j, dst in enumerate(emes):
                path = reconstruct_path(prev_by_source[src], dst)
                if cost[i][j] >= UNREACHABLE:
                    continue
                # Sum of weights along the reconstructed path matches.
                total = 0.0
                for u, v in zip(path, path[1:]):
                    for nb, w in city.neighbors(u):
                        if nb == v:
                            total += w
                            break
                self.assertAlmostEqual(total, cost[i][j])


if __name__ == "__main__":
    unittest.main()
