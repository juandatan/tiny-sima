"""Measured odds of the executor's fights, for the goal source's fight choice.

The staged fights of :mod:`tiny_sima.evaluation.fights` show the executor
wins a zombie fight about as often as a scripted chase-and-hit reference,
and that what decides the fight is the player's health and sword: at health
2 it is always lost, and from 5 up mostly won. Whether to fight is then the
goal source's call. This table gives it the measured odds of the fight in
front of it, looked up from what the player can see: its health, its best
sword and how many of the creature are close.
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
    """The measured cell a fight was looked up in, with its rates."""

    health: int
    sword: str
    mobs: int
    win_rate: float
    death_rate: float
    rollouts: int


@dataclass(frozen=True)
class FightOdds:
    """Win and death rates of the executor's fights, by setting.

    ``rates`` maps ``(mob, health, sword, mobs)`` to ``(win_rate,
    death_rate, rollouts)``, pooled over the staged distances and times of
    day, which barely moved them.
    """

    rates: Mapping[tuple[str, int, str, int], tuple[float, float, int]]

    def lookup(self, mob: str, health: int, sword: str, mobs: int) -> FightRate | None:
        """Return the rates of the nearest measured cell, or None for none.

        Health and the creature count are clamped to the measured range, so
        health 1 reads as the lowest health staged, an upper bound on its
        odds.
        """

        cells = [key for key in self.rates if key[0] == mob and key[2] == sword]
        if not cells:
            return None
        healths = sorted({key[1] for key in cells})
        counts = sorted({key[3] for key in cells})
        key = (
            mob,
            min(max(health, healths[0]), healths[-1]),
            sword,
            min(max(mobs, counts[0]), counts[-1]),
        )
        if key not in self.rates:
            return None
        return FightRate(key[1], sword, key[3], *self.rates[key])

    def for_state(self, state: Any, mob: str, mobs: int) -> FightRate | None:
        """Return the rates for fighting ``mobs`` close ``mob`` in ``state``."""

        health = int(np.asarray(state.player_health))
        return self.lookup(mob, health, held_sword(state), mobs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the table."""

        return {
            "rates": [
                {
                    "mob": mob,
                    "health": health,
                    "sword": sword,
                    "mobs": mobs,
                    "win_rate": win_rate,
                    "death_rate": death_rate,
                    "rollouts": rollouts,
                }
                for (mob, health, sword, mobs), (
                    win_rate,
                    death_rate,
                    rollouts,
                ) in sorted(self.rates.items())
            ]
        }

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> FightOdds:
        """Build a table from rows of the shape :meth:`to_dict` writes."""

        rates: dict[tuple[str, int, str, int], tuple[float, float, int]] = {}
        for row in rows:
            if row["sword"] not in SWORDS:
                raise ValueError(f"Unknown sword {row['sword']!r}.")
            rates[(row["mob"], int(row["health"]), row["sword"], int(row["mobs"]))] = (
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
