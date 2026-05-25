"""Tests for the triage / fleet-shortage policy layer."""

from __future__ import annotations

import unittest

from dispatcher.triage import Emergency, triage


def _make(coord, priority, call_id):
    return Emergency(coord=coord, priority=priority, call_id=call_id)


class TriageTests(unittest.TestCase):
    def test_no_shortage_dispatches_everyone(self):
        es = [_make((0, 0), 2, 0), _make((1, 1), 5, 1)]
        d, q = triage(es, capacity=5)
        self.assertEqual(len(d), 2)
        self.assertEqual(q, [])

    def test_shortage_keeps_highest_priority(self):
        es = [
            _make((0, 0), 1, 0),
            _make((1, 1), 5, 1),
            _make((2, 2), 3, 2),
        ]
        d, q = triage(es, capacity=2)
        self.assertEqual([e.priority for e in d], [5, 3])
        self.assertEqual([e.priority for e in q], [1])

    def test_fifo_tiebreak_on_equal_priority(self):
        es = [
            _make((0, 0), 4, 10),
            _make((1, 1), 4, 5),   # older call, equal priority
            _make((2, 2), 4, 20),
        ]
        d, q = triage(es, capacity=2)
        self.assertEqual([e.call_id for e in d], [5, 10])
        self.assertEqual([e.call_id for e in q], [20])

    def test_zero_capacity_queues_all(self):
        es = [_make((0, 0), 5, 0)]
        d, q = triage(es, capacity=0)
        self.assertEqual(d, [])
        self.assertEqual(len(q), 1)

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            triage([], capacity=-1)


if __name__ == "__main__":
    unittest.main()
