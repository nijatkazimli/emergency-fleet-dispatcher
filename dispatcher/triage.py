"""Triage / pre-processing layer for the dispatcher.

This sits ABOVE both algorithms:

    raw emergencies (with priorities)  +  ambulance count
                |
                v
          triage(...)                          <-- this module
                |
                +-- dispatched -> Algorithm A -> cost matrix -> Algorithm B
                |
                +-- queued     -> held for the next dispatch cycle

It exists because neither Dijkstra (Algorithm A) nor the Hungarian solver
(Algorithm B) knows how to choose WHICH emergencies to drop when there
aren't enough ambulances. That decision is policy, not math: it needs a
priority (severity, age of the call, ...). Keeping it out of both
algorithms preserves their generality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Emergency:
    """A single emergency call.

    Attributes
    ----------
    coord    : opaque location identifier (e.g. (row, col) on the grid).
    priority : integer urgency; HIGHER means MORE urgent.
    call_id  : stable monotonically-increasing id used as a FIFO tie-breaker
               so older calls beat newer ones at equal priority.
    """
    coord: Tuple[int, int]
    priority: int
    call_id: int


def triage(
    emergencies: Sequence[Emergency],
    capacity: int,
) -> Tuple[List[Emergency], List[Emergency]]:
    """Split incoming emergencies into (dispatched_now, queued_for_later).

    Rules
    -----
    1. If `capacity >= len(emergencies)` everyone is dispatched, nothing queued.
    2. Otherwise the top `capacity` calls (by priority, then by call_id)
       are dispatched and the rest are queued for the next cycle.
    3. The relative order of the returned lists is the same as if we sorted
       all calls by (-priority, call_id): the most urgent come first.

    Parameters
    ----------
    emergencies : sequence of Emergency
    capacity    : number of ambulances available right now (>= 0).
    """
    if capacity < 0:
        raise ValueError("capacity must be >= 0")
    # Higher priority first, then FIFO (lower call_id first) on ties.
    ranked = sorted(emergencies, key=lambda e: (-e.priority, e.call_id))
    return list(ranked[:capacity]), list(ranked[capacity:])
