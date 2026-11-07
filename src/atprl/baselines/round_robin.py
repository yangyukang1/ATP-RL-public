from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import ClassVar


@dataclass
class RoundRobinScheduler:
    """Cycle over all K-user combinations; advance only after a committed decision."""

    SCHEDULE_KIND: ClassVar[str] = "all_combinations_lexicographic_v1"
    STATE_SCHEMA: ClassVar[int] = 1

    n: int
    k: int
    offset: int = 0

    def __post_init__(self) -> None:
        if not all(
            isinstance(x, int) and not isinstance(x, bool) for x in (self.n, self.k, self.offset)
        ):
            raise ValueError("N, K and offset must be integers")
        if not 1 <= self.k <= self.n:
            raise ValueError("Require 1 <= K <= N")
        if not 0 <= self.offset < comb(self.n, self.k):
            raise ValueError("Require 0 <= combination offset < C(N,K)")
        self._combinations = tuple(combinations(range(self.n), self.k))
        self.reset()

    @property
    def cycle_length(self) -> int:
        return len(self._combinations)

    def peek(self) -> tuple[int, ...]:
        return self._combinations[self.pointer]

    def commit(self) -> None:
        self.decisions += 1
        self.pointer = (self.pointer + 1) % self.cycle_length

    def reset(self) -> None:
        self.pointer = self.offset
        self.decisions = 0

    def state_dict(self) -> dict:
        return dict(
            schema_version=self.STATE_SCHEMA,
            schedule_kind=self.SCHEDULE_KIND,
            n=self.n,
            k=self.k,
            offset=self.offset,
            cycle_length=self.cycle_length,
            pointer=self.pointer,
            decisions=self.decisions,
        )

    def load_state_dict(self, state: dict) -> None:
        expected = dict(
            schema_version=self.STATE_SCHEMA,
            schedule_kind=self.SCHEDULE_KIND,
            n=self.n,
            k=self.k,
            offset=self.offset,
            cycle_length=self.cycle_length,
        )
        if not isinstance(state, dict) or any(
            type(state.get(key)) is not type(value) or state.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("Round-robin schedule/configuration mismatch")
        count, pointer = state.get("decisions"), state.get("pointer")
        if (
            type(count) is not int
            or count < 0
            or type(pointer) is not int
            or pointer != (self.offset + count) % self.cycle_length
        ):
            raise ValueError("Invalid round-robin recovery state")
        self.decisions, self.pointer = count, pointer
