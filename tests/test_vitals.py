import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_observation import _grass_world
from tiny_sima.evaluation.vitals import _steps_per_change, measure_vital_rates
from tiny_sima.planner.vital_rates import VitalRates

RATES = VitalRates(
    food_awake=26.0,
    drink_awake=21.0,
    energy_awake=31.0,
    food_asleep=51.0,
    drink_asleep=41.0,
    energy_asleep=10.5,
    health_awake=27.0,
    health_asleep=13.0,
    health_starving=18.0,
    plant_ripe=599.0,
)


def test_steps_per_change_pools_stretches_and_stops_at_empty() -> None:
    # One world: falls every 2 steps, then sits at 0, which does not count.
    values = np.array([[3], [3], [2], [2], [1], [1], [0], [0], [0], [0]])
    keep = np.ones((9, 1), dtype=bool)
    assert _steps_per_change(values, -1, keep) == 2.0
    # Two short kept stretches of 2 steps each, with one fall in each.
    keep = np.array([[1], [1], [0], [1], [1], [0], [0], [0], [0]], dtype=bool)
    assert _steps_per_change(values, -1, keep) == 2.0
    assert _steps_per_change(values, 1, keep) is None


def test_the_rates_are_told_and_projected() -> None:
    text = RATES.system_text()
    assert "awake, food falls by 1 about every 26 steps" in text
    assert "health rises by 1 about every 27 steps awake and every 13 asleep" in text
    assert "A planted sapling was ripe after about 599 steps." in text

    state = _grass_world().replace(
        player_health=jnp.int32(5),
        player_food=jnp.int32(4),
        player_drink=jnp.int32(2),
        player_energy=jnp.int32(9),
    )
    assert RATES.projection(state) == (
        "At these rates, steps until empty: food about 104, drink about 42, "
        "energy about 279; health is full again in about 108 steps awake or "
        "52 steps asleep."
    )


def test_the_rates_round_trip(tmp_path) -> None:
    path = tmp_path / "vital_rates.json"
    RATES.save(path)
    assert VitalRates.load(path) == RATES


def test_staged_play_measures_the_game_rates() -> None:
    rates = measure_vital_rates(worlds_per_scenario=1)
    # The game's own counters: food every ~25 steps awake, drink ~20,
    # energy ~30, health back every ~25 awake and ~12.5 asleep, a sapling
    # ripe after ~600 steps.
    assert rates.food_awake == pytest.approx(26, abs=2)
    assert rates.drink_awake == pytest.approx(21, abs=2)
    assert rates.energy_awake == pytest.approx(31, abs=2)
    assert rates.food_asleep == pytest.approx(51, abs=4)
    assert rates.drink_asleep == pytest.approx(41, abs=4)
    assert rates.health_awake == pytest.approx(26, abs=3)
    assert rates.health_asleep == pytest.approx(13, abs=2)
    assert rates.health_starving is not None
    assert rates.plant_ripe == pytest.approx(600, abs=5)
