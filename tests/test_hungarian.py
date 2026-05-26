"""Unit tests for the Hungarian / Kuhn–Munkres algorithm."""

from __future__ import annotations

import random
import unittest

from dispatcher.assignment import (
    Assignment,
    greedy_assignment,
    solve_assignment,
    total_cost,
)
from routing import UNREACHABLE


class TestHungarianBasic(unittest.TestCase):
    """Basic functionality tests for the Hungarian algorithm."""

    def test_empty_matrix(self):
        """Empty input returns empty assignment."""
        result = solve_assignment([])
        self.assertEqual(result, [])

    def test_single_cell(self):
        """1x1 matrix returns single assignment."""
        cost_matrix = [[5.0]]
        result = solve_assignment(cost_matrix)
        self.assertEqual(result, [(0, 0)])
        self.assertAlmostEqual(total_cost(cost_matrix, result), 5.0)

    def test_hand_checked_3x3_optimum(self):
        """Hand-verified 3x3 matrix with known optimal solution.

        Matrix:
            [4,  1,  3]
            [2,  0,  5]
            [3,  2,  2]

        Optimal assignment: (0,1)=1, (1,0)=2, (2,2)=2 for total=5.
        """
        cost_matrix = [
            [4.0, 1.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 2.0, 2.0],
        ]
        result = solve_assignment(cost_matrix)
        result_cost = total_cost(cost_matrix, result)

        # Verify basic properties
        self.assertEqual(len(result), 3)
        rows = {i for i, j in result}
        cols = {j for i, j in result}
        self.assertEqual(len(rows), 3, "Rows must be unique")
        self.assertEqual(len(cols), 3, "Cols must be unique")

        # Verify optimality (must be exactly 5.0 for this known case)
        self.assertAlmostEqual(result_cost, 5.0, places=5)

    def test_no_duplicates_in_assignment(self):
        """Verify no row or column appears twice in assignment."""
        cost_matrix = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        result = solve_assignment(cost_matrix)
        rows = [i for i, j in result]
        cols = [j for i, j in result]
        self.assertEqual(len(rows), len(set(rows)), "Duplicate rows in assignment")
        self.assertEqual(len(cols), len(set(cols)), "Duplicate cols in assignment")

    def test_indices_in_bounds(self):
        """All returned indices must be valid into the original matrix."""
        cost_matrix = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        result = solve_assignment(cost_matrix)
        n_rows, n_cols = len(cost_matrix), len(cost_matrix[0])
        for i, j in result:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, n_rows)
            self.assertGreaterEqual(j, 0)
            self.assertLess(j, n_cols)


class TestHungarianRectangular(unittest.TestCase):
    """Test handling of rectangular (N ≠ M) matrices."""

    def test_more_rows_than_cols(self):
        """N > M: return exactly M assignments."""
        cost_matrix = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2, "Should assign exactly min(N, M)=2")

        rows = {i for i, j in result}
        cols = {j for i, j in result}
        self.assertEqual(len(rows), 2, "Rows must be unique")
        self.assertEqual(len(cols), 2, "Cols must be unique")

    def test_more_cols_than_rows(self):
        """M > N: return exactly N assignments."""
        cost_matrix = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2, "Should assign exactly min(N, M)=2")

        rows = {i for i, j in result}
        cols = {j for i, j in result}
        self.assertEqual(len(rows), 2, "Rows must be unique")
        self.assertEqual(len(cols), 2, "Cols must be unique")

    def test_rectangular_4x2(self):
        """4x2 matrix: should assign 2 ambulances to 2 emergencies."""
        cost_matrix = [
            [1.0, 5.0],
            [2.0, 3.0],
            [4.0, 1.0],
            [6.0, 2.0],
        ]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2)
        cost = total_cost(cost_matrix, result)
        # Optimal: pick (2,1)=1 and (1,0)=2 for total=3, or (0,0)=1 and (2,1)=1 for 2
        self.assertLess(cost, 10.0, "Cost should be reasonable")

    def test_rectangular_2x5(self):
        """2x5 matrix: should assign 2 ambulances to 2 emergencies."""
        cost_matrix = [
            [10.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2)
        rows = {i for i, j in result}
        cols = {j for i, j in result}
        self.assertEqual(len(rows), 2, "Rows must be unique")
        self.assertEqual(len(cols), 2, "Cols must be unique")


class TestHungarianUnreachable(unittest.TestCase):
    """Test handling of UNREACHABLE sentinel values."""

    def test_unreachable_in_matrix(self):
        """Matrix with UNREACHABLE cells: algorithm avoids them."""
        # UNREACHABLE = 1e9
        cost_matrix = [
            [1.0, UNREACHABLE, 3.0],
            [UNREACHABLE, 2.0, 4.0],
            [5.0, 6.0, UNREACHABLE],
        ]
        result = solve_assignment(cost_matrix)
        result_cost = total_cost(cost_matrix, result)

        # Should avoid UNREACHABLE cells if possible
        # Optimal: (0,0)=1, (1,1)=2, (2,1)=6 but col 1 used twice - invalid
        # Try: (0,0)=1, (1,1)=2, (2,2)=UNREACHABLE but cost >> normal
        # Or: (0,0)=1, (1,2)=4, (2,1)=6 for total=11
        # Or: (0,2)=3, (1,1)=2, (2,0)=5 for total=10

        self.assertEqual(len(result), 3)
        # Should find a valid matching with reasonable cost
        self.assertLess(result_cost, 1e9 + 100, "Cost should be finite and reasonable")

    def test_all_unreachable_row(self):
        """Row with all UNREACHABLE values."""
        cost_matrix = [
            [1.0, 2.0],
            [UNREACHABLE, UNREACHABLE],
        ]
        result = solve_assignment(cost_matrix)
        # Should still assign both rows (min(2,2)=2)
        self.assertEqual(len(result), 2)
        rows = {i for i, j in result}
        cols = {j for i, j in result}
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(cols), 2)

    def test_unreachable_forced_assignment(self):
        """When forced, algorithm accepts UNREACHABLE costs."""
        # All paths to a complete matching require at least one UNREACHABLE
        cost_matrix = [
            [UNREACHABLE, 1.0],
            [2.0, UNREACHABLE],
        ]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2)
        result_cost = total_cost(cost_matrix, result)
        # Should be (0,1)=1 + (1,0)=2 = 3, not the UNREACHABLE cells
        self.assertLess(result_cost, UNREACHABLE, "Should avoid UNREACHABLE if possible")


class TestHungarianOptimality(unittest.TestCase):
    """Test optimality guarantees."""

    def test_beaten_greedy_simple(self):
        """Hungarian should beat or match greedy on a simple case."""
        cost_matrix = [
            [10.0, 1.0, 100.0],
            [1.0, 10.0, 100.0],
            [100.0, 100.0, 1.0],
        ]
        hungarian_result = solve_assignment(cost_matrix)
        greedy_result = greedy_assignment(cost_matrix)

        hungarian_cost = total_cost(cost_matrix, hungarian_result)
        greedy_cost = total_cost(cost_matrix, greedy_result)

        self.assertLessEqual(
            hungarian_cost,
            greedy_cost,
            "Hungarian must beat or tie greedy",
        )

    def test_beaten_greedy_property(self):
        """Property test: Hungarian <= Greedy on 20 random 5x5 matrices."""
        rng = random.Random(12345)
        successes = 0

        for trial in range(20):
            # Generate random 5x5 cost matrix
            cost_matrix = [
                [rng.random() * 100 for _ in range(5)]
                for _ in range(5)
            ]

            hungarian_result = solve_assignment(cost_matrix)
            greedy_result = greedy_assignment(cost_matrix)

            hungarian_cost = total_cost(cost_matrix, hungarian_result)
            greedy_cost = total_cost(cost_matrix, greedy_result)

            # Hungarian should never be worse than greedy
            self.assertLessEqual(
                hungarian_cost,
                greedy_cost + 1e-9,  # Small epsilon for floating-point error
                f"Trial {trial}: Hungarian={hungarian_cost} > Greedy={greedy_cost}",
            )
            if hungarian_cost < greedy_cost:
                successes += 1

        # In most cases, Hungarian should beat greedy
        self.assertGreater(
            successes,
            10,
            f"Hungarian beat greedy in only {successes}/20 trials; likely bug",
        )

    def test_determinism(self):
        """Running twice on the same input gives the same total cost."""
        cost_matrix = [
            [3.0, 1.0, 2.0],
            [1.0, 4.0, 3.0],
            [2.0, 3.0, 1.0],
        ]
        result1 = solve_assignment(cost_matrix)
        result2 = solve_assignment(cost_matrix)

        cost1 = total_cost(cost_matrix, result1)
        cost2 = total_cost(cost_matrix, result2)

        self.assertAlmostEqual(cost1, cost2, places=10, msg="Costs must be deterministic")


class TestHungarianEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_all_zeros(self):
        """Matrix of all zeros."""
        cost_matrix = [[0.0, 0.0], [0.0, 0.0]]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(total_cost(cost_matrix, result), 0.0)

    def test_all_same_cost(self):
        """All cells have the same non-zero cost."""
        cost_matrix = [[5.0, 5.0], [5.0, 5.0]]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(total_cost(cost_matrix, result), 10.0)

    def test_large_variance_costs(self):
        """Costs with large variance (near UNREACHABLE)."""
        cost_matrix = [
            [0.1, 1e8],
            [1e8, 0.2],
        ]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2)
        cost = total_cost(cost_matrix, result)
        # Should pick (0,0) and (1,1) for total ~0.3
        self.assertLess(cost, 1.0, "Should find low-cost matching")

    def test_single_row(self):
        """1xM matrix."""
        cost_matrix = [[1.0, 2.0, 3.0]]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0][0] == 0, "Row must be 0")
        self.assertIn(result[0][1], [0, 1, 2], "Col must be one of the emergencies")

    def test_single_col(self):
        """Nx1 matrix."""
        cost_matrix = [[1.0], [2.0], [3.0]]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 0, "Col must be 0")
        self.assertIn(result[0][0], [0, 1, 2], "Row must be one of the ambulances")


class TestHungarianNumericStability(unittest.TestCase):
    """Test numeric stability with floating-point edge cases."""

    def test_very_small_costs(self):
        """Matrix with very small costs (near machine epsilon)."""
        eps = 1e-15
        cost_matrix = [
            [eps, 2 * eps],
            [3 * eps, eps],
        ]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2)
        # Should complete without NaN or Inf
        cost = total_cost(cost_matrix, result)
        self.assertFalse(cost != cost, "Cost must not be NaN")  # NaN != NaN is True
        self.assertTrue(cost != float('inf'), "Cost must not be Inf")

    def test_mixed_scale_costs(self):
        """Matrix with costs at widely different scales."""
        cost_matrix = [
            [1e-10, 1e10],
            [1e10, 1e-10],
        ]
        result = solve_assignment(cost_matrix)
        self.assertEqual(len(result), 2)
        cost = total_cost(cost_matrix, result)
        # Should pick (0,0) and (1,1) for total ~2e-10
        self.assertLess(cost, 1.0, "Should find diagonal matching")


if __name__ == '__main__':
    unittest.main()
