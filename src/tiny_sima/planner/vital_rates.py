"""Measured rates of the player's vitals, for the goal source's long-term plans.

The goal source can see its status bar, but not how fast it changes: how
long food lasts, how long health takes to come back awake or asleep, or how
long a planted sapling takes to ripen. A player learns these by playing.
:mod:`tiny_sima.evaluation.vitals` measures them the same way, in staged
play, from what the status bar and the screen show, and this table holds
the result: steps per point of each change. The goal source is told the
rates once, and at each decision how long its needs last at those rates.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from tiny_sima.planner.observation import vitals

_MAX_VITAL = 9


@dataclass(frozen=True)
class VitalRates:
    """Steps per point of each vital change, as measured; ``None`` if unseen.

    The ``*_awake`` and ``*_asleep`` rates of food, drink and energy are
    steps per point lost, except ``energy_asleep``, steps per point gained.
    ``health_awake`` and ``health_asleep`` are steps per point regained while
    food, drink and energy are above 0, ``health_starving`` steps per point
    lost awake while one of them is 0, and ``plant_ripe`` the steps from
    planting a sapling until it is ripe.
    """

    food_awake: float | None
    drink_awake: float | None
    energy_awake: float | None
    food_asleep: float | None
    drink_asleep: float | None
    energy_asleep: float | None
    health_awake: float | None
    health_asleep: float | None
    health_starving: float | None
    plant_ripe: float | None

    def system_text(self) -> str:
        """The rates, in words, for the system prompt."""

        def every(rate: float | None) -> str:
            return "?" if rate is None else f"{rate:.0f}"

        parts = [
            "Measured in this player's own staged play: awake, food falls by 1 "
            f"about every {every(self.food_awake)} steps, drink every "
            f"{every(self.drink_awake)} and energy every "
            f"{every(self.energy_awake)}; asleep, food falls every "
            f"{every(self.food_asleep)} steps and drink every "
            f"{every(self.drink_asleep)}, and energy rises by 1 every "
            f"{every(self.energy_asleep)}.",
            "While food, drink and energy are all above 0, health rises by 1 "
            f"about every {every(self.health_awake)} steps awake and every "
            f"{every(self.health_asleep)} asleep; while any is 0 it falls by 1 "
            f"every {every(self.health_starving)}.",
        ]
        if self.plant_ripe is not None:
            parts.append(
                f"A planted sapling was ripe after about {self.plant_ripe:.0f} steps."
            )
        return " ".join(parts)

    def projection(self, state: Any) -> str:
        """How long each need lasts, and health takes to fill, at these rates."""

        readings = vitals(state)
        asleep = readings["sleeping"]
        rates = {
            "food": self.food_asleep if asleep else self.food_awake,
            "drink": self.drink_asleep if asleep else self.drink_awake,
        }
        if not asleep:
            rates["energy"] = self.energy_awake
        lasts = [
            f"{need} about {readings[need] * rate:.0f}"
            for need, rate in rates.items()
            if rate is not None
        ]
        parts = []
        if lasts:
            parts.append("At these rates, steps until empty: " + ", ".join(lasts))
        missing = _MAX_VITAL - readings["health"]
        regain = [
            f"{missing * rate:.0f} steps {how}"
            for how, rate in (
                ("awake", self.health_awake),
                ("asleep", self.health_asleep),
            )
            if rate is not None
        ]
        if missing > 0 and regain:
            parts.append("health is full again in about " + " or ".join(regain))
        return "; ".join(parts) + "." if parts else ""

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VitalRates:
        return cls(
            **{
                item.name: None
                if payload.get(item.name) is None
                else float(payload[item.name])
                for item in fields(cls)
            }
        )

    def save(self, path: str | Path) -> None:
        """Write the rates so a later session can skip the measurement."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> VitalRates:
        """Read rates written by :meth:`save`."""

        return cls.from_dict(json.loads(Path(path).read_text()))
