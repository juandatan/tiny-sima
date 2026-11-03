"""Measured odds of the executor's fights, for the goal source's fight choice.

The staged fights of :mod:`tiny_sima.evaluation.fights` show the executor
wins a zombie fight about as often as a scripted chase-and-hit reference,
and that what decides the fight is the player's health and sword: at health
2 it is always lost, and from 5 up mostly won. Whether to fight is then the
goal source's call. This table gives it the measured odds of the fight in
front of it, looked up from what the player can see: its health, its best
sword and how many of the creature are close. Declining is not free: the
creature follows, so the table also holds the odds of an errand, such as
drinking, run with the same creatures close.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

SWORDS: tuple[str, ...] = ("none", "wood", "stone")


def held_sword(state: Any) -> str:
    """Return the best sword the player holds, an iron one counted as stone."""

    inventory = state.inventory
    if int(np.asarray(inventory.stone_sword)) or int(np.asarray(inventory.iron_sword)):
        return "stone"
    return "wood" if int(np.asarray(inventory.wood_sword)) else "none"


class FightRate(NamedTuple):
    """The measured cell a fight or errand was looked up in, with its rates."""

    skill: str
    health: int
    sword: str
    mobs: int
    win_rate: float
    death_rate: float
    rollouts: int


Key = tuple[str, str, int, str, int]


@dataclass(frozen=True)
class FightOdds:
    """Success and death rates of the executor's staged skills, by setting.

    ``rates`` maps ``(skill, mob, health, sword, mobs)`` to ``(win_rate,
    death_rate, rollouts)``, pooled over the staged distances and times of
    day, which barely moved them. ``skill`` is a skill id's value: the
    hunting skill for ``mob``, whose win is the kill, or an errand run with
    ``mobs`` of them close, whose win is the errand done.
    """

    rates: Mapping[Key, tuple[float, float, int]]

    def skills(self, mob: str) -> tuple[str, ...]:
        """Return the skills staged with ``mob`` close, in sorted order."""

        return tuple(sorted({key[0] for key in self.rates if key[1] == mob}))

    def lookup(
        self, skill: str, mob: str, health: int, sword: str, mobs: int
    ) -> FightRate | None:
        """Return the rates of the nearest measured cell, or None for none.

        Health and the creature count are clamped to the measured range, so
        health 1 reads as the lowest health staged, an upper bound on its
        odds.
        """

        cells = [
            key for key in self.rates if key[:2] == (skill, mob) and key[3] == sword
        ]
        if not cells:
            return None
        healths = sorted({key[2] for key in cells})
        counts = sorted({key[4] for key in cells})
        key = (
            skill,
            mob,
            min(max(health, healths[0]), healths[-1]),
            sword,
            min(max(mobs, counts[0]), counts[-1]),
        )
        if key not in self.rates:
            return None
        return FightRate(skill, key[2], sword, key[4], *self.rates[key])

    def for_state(
        self, state: Any, skill: str, mob: str, mobs: int
    ) -> FightRate | None:
        """Return the rates for running ``skill`` with ``mobs`` ``mob`` close."""

        health = int(np.asarray(state.player_health))
        return self.lookup(skill, mob, health, held_sword(state), mobs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the table."""

        return {
            "rates": [
                {
                    "skill": skill,
                    "mob": mob,
                    "health": health,
                    "sword": sword,
                    "mobs": mobs,
                    "win_rate": win_rate,
                    "death_rate": death_rate,
                    "rollouts": rollouts,
                }
                for (skill, mob, health, sword, mobs), (
                    win_rate,
                    death_rate,
                    rollouts,
                ) in sorted(self.rates.items())
            ]
        }

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> FightOdds:
        """Build a table from rows of the shape :meth:`to_dict` writes."""

        rates: dict[Key, tuple[float, float, int]] = {}
        for row in rows:
            if row["sword"] not in SWORDS:
                raise ValueError(f"Unknown sword {row['sword']!r}.")
            key = (
                row["skill"],
                row["mob"],
                int(row["health"]),
                row["sword"],
                int(row["mobs"]),
            )
            rates[key] = (
                float(row["win_rate"]),
                float(row["death_rate"]),
                int(row["rollouts"]),
            )
        return cls(rates)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FightOdds:
        """Rebuild a table from :meth:`to_dict` output."""

        return cls.from_rows(payload["rates"])

    def save(self, path: str | Path) -> None:
        """Write the table so a later session can skip the staged fights."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FightOdds:
        """Read a table written by :meth:`save`."""

        return cls.from_dict(json.loads(Path(path).read_text()))
