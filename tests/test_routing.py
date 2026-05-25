"""Smoke tests for the Routing Engine.

Run:  python -m unittest tests/test_routing.py
"""

from __future__ import annotations

import math
import unittest

from routing import Graph, build_cost_matrix, dijkstra, multi_source_costs, UNREACHABLE
from routing.city_generator import build_grid_city, node_id


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


if __name__ == "__main__":
    unittest.main()
