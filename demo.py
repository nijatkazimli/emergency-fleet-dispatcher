"""Demo for Algorithm A (Routing Engine).

Run:  python demo.py

Generates a small mock city, computes the N x M cost matrix from
ambulances -> emergencies, and prints it in the exact format that
Partner 2's Hungarian algorithm will consume.
"""

from __future__ import annotations

from routing import build_cost_matrix
from routing.city_generator import build_grid_city, node_id


def main() -> None:
    rows, cols = 6, 6
    # Two road closures to demonstrate edge-case handling.
    closed = [((2, 2), (2, 3)), ((3, 2), (3, 3))]
    city = build_grid_city(rows, cols, closed_edges=closed, seed=42)

    ambulances_coords = [(0, 0), (5, 0), (0, 5)]
    emergencies_coords = [(5, 5), (3, 3), (2, 4)]

    ambulances = [node_id(c, cols) for c in ambulances_coords]
    emergencies = [node_id(c, cols) for c in emergencies_coords]

    cost = build_cost_matrix(city, ambulances, emergencies)

    print("=== Algorithm A : Cost Matrix Handoff ===")
    print(f"Ambulances (rows):   {ambulances_coords}")
    print(f"Emergencies (cols):  {emergencies_coords}")
    print()
    header = "             " + "  ".join(f"E{j}={e!s:>6}" for j, e in enumerate(emergencies_coords))
    print(header)
    for i, row in enumerate(cost):
        cells = "  ".join(f"{v:>10.3f}" for v in row)
        print(f"A{i}={ambulances_coords[i]!s:>6}  {cells}")

    print("\nThis 2D list is the exact object passed to Algorithm B.")


if __name__ == "__main__":
    main()
