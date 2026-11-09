"""Staged play that measures how fast the player's vitals change.

A player learns from playing how long food lasts, how fast health comes
back, and how long a sapling takes to ripen. This module learns the same by
watching: it stages a few situations in the open-ended world, without
creatures, and reads off only what the status bar and the screen show.

* ``awake``: low health and full needs, doing nothing. Food, drink and
  energy fall, and health comes back.
* ``asleep``: low health and no energy, sleeping until energy is back.
* ``sleeping``: the same, but energy is held at 0 so the player sleeps on.
  Health comes back, and food and drink fall more slowly.
* ``starving``: full health and no food. Health falls.
* ``plant``: a sapling planted on the grass the player faces, its needs kept
  full, until the plant there is ripe.

:func:`vital_rates_from_readings` turns the readings into a
:class:`~tiny_sima.planner.vital_rates.VitalRates`: each rate is the number
of steps in which the situation held, pooled over worlds and stretches,
per change of the vital in those steps.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import DIRECTIONS, Action, BlockType

from tiny_sima.envs.chains import reset_player
from tiny_sima.envs.open_ended import make_open_ended_environment
from tiny_sima.planner.vital_rates import VitalRates

VITAL_SCENARIOS: tuple[str, ...] = ("awake", "asleep", "sleeping", "starving", "plant")
VITAL_RATES_SEED = 223
# Long enough for a sapling to ripen with room to spare.
DEFAULT_VITAL_STEPS = 720
_MAX_VITAL = 9


class VitalReadings(NamedTuple):
    """What the player sees at each step, shaped ``[steps + 1, worlds]``."""

    health: np.ndarray
    food: np.ndarray
    drink: np.ndarray
    energy: np.ndarray
    sleeping: np.ndarray
    # The block on the tile the player faces.
    facing: np.ndarray


def _facing(state: Any) -> jax.Array:
    return state.player_position + jnp.asarray(DIRECTIONS)[state.player_direction]


def _reading(state: Any) -> tuple[jax.Array, ...]:
    tile = _facing(state)
    return (
        state.player_health,
        state.player_food,
        state.player_drink,
        state.player_energy,
        state.is_sleeping,
        state.map[tile[0], tile[1]],
    )


def _stage(state: Any, scenario: jax.Array) -> Any:
    """Set the player up for one of :data:`VITAL_SCENARIOS`, by index."""

    full = jnp.asarray(_MAX_VITAL, dtype=jnp.int32)
    low = jnp.asarray(1, dtype=jnp.int32)
    starving = scenario == VITAL_SCENARIOS.index("starving")
    state = reset_player(
        state,
        health=jnp.where(starving, full, low),
        food=jnp.where(starving, 0, full),
        drink=full,
        energy=jnp.where(_sleeps(scenario), 0, full),
    )
    plant = scenario == VITAL_SCENARIOS.index("plant")
    tile = _facing(state)
    block = jnp.where(plant, BlockType.GRASS.value, state.map[tile[0], tile[1]])
    return state.replace(
        map=state.map.at[tile[0], tile[1]].set(block.astype(state.map.dtype)),
        inventory=state.inventory.replace(
            sapling=jnp.where(plant, 1, state.inventory.sapling)
        ),
    )


def _sleeps(scenario: jax.Array | int) -> jax.Array:
    return (scenario == VITAL_SCENARIOS.index("asleep")) | (
        scenario == VITAL_SCENARIOS.index("sleeping")
    )


def _actions(scenarios: jax.Array, steps: int) -> jax.Array:
    """Each world's action at each step, shaped ``[steps, worlds]``."""

    sleep = _sleeps(scenarios)
    plant = scenarios == VITAL_SCENARIOS.index("plant")
    actions = jnp.where(sleep, Action.SLEEP.value, Action.NOOP.value)
    first = jnp.where(plant, Action.PLACE_PLANT.value, actions)
    return jnp.concatenate(
        [first[None], jnp.broadcast_to(actions, (steps - 1, *actions.shape))]
    ).astype(jnp.int32)


def run_vital_scenarios(
    *,
    worlds_per_scenario: int = 4,
    steps: int = DEFAULT_VITAL_STEPS,
    seed: int = VITAL_RATES_SEED,
) -> tuple[np.ndarray, VitalReadings]:
    """Stage every scenario in ``worlds_per_scenario`` worlds and play it out.

    Returns each world's scenario index and its readings.
    """

    environment = make_open_ended_environment(mobs=False)
    env, params = environment.bundle.env, environment.bundle.params
    scenarios = jnp.repeat(jnp.arange(len(VITAL_SCENARIOS)), worlds_per_scenario)
    reset_rng, step_rng = jax.random.split(jax.random.PRNGKey(seed))
    keys = jax.random.split(reset_rng, scenarios.shape[0])
    states = jax.vmap(lambda key: environment.reset(key).state)(keys)
    states = jax.vmap(_stage)(states, scenarios)
    hold = scenarios == VITAL_SCENARIOS.index("plant")
    drowsy = scenarios == VITAL_SCENARIOS.index("sleeping")
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def advance(state: Any, inputs: tuple[jax.Array, jax.Array]) -> tuple[Any, Any]:
        key, action = inputs
        _, state, _, _, _ = step(
            jax.random.split(key, scenarios.shape[0]), state, action, params
        )
        # The plant's watcher is kept fed, so it outlives the ripening, and
        # the sleeper is kept tired, so it does not wake.
        energy = jnp.where(hold, _MAX_VITAL, jnp.where(drowsy, 0, state.player_energy))
        state = state.replace(
            player_food=jnp.where(hold, _MAX_VITAL, state.player_food).astype(
                state.player_food.dtype
            ),
            player_drink=jnp.where(hold, _MAX_VITAL, state.player_drink).astype(
                state.player_drink.dtype
            ),
            player_energy=energy.astype(state.player_energy.dtype),
        )
        return state, jax.vmap(_reading)(state)

    @jax.jit
    def play(states: Any) -> Any:
        first = jax.vmap(_reading)(states)
        _, rest = jax.lax.scan(
            advance,
            states,
            (jax.random.split(step_rng, steps), _actions(scenarios, steps)),
        )
        return jax.tree_util.tree_map(
            lambda start, later: jnp.concatenate([start[None], later]), first, rest
        )

    readings = VitalReadings(*(np.asarray(leaf) for leaf in play(states)))
    return np.asarray(scenarios), readings


def _steps_per_change(
    values: np.ndarray, direction: int, keep: np.ndarray
) -> float | None:
    """Kept steps per change of ``values`` by ``direction``, or ``None``.

    ``values`` is ``[steps + 1, worlds]`` and ``keep`` ``[steps, worlds]``
    says which steps count. Pooling the kept steps of every stretch lets
    short stretches, such as one night's sleep, add up. A step counts only
    while the vital has room to move: not once it is empty or full.
    """

    before = values[:-1].astype(int)
    keep = keep & ((before > 0) if direction < 0 else (before < _MAX_VITAL))
    changed = (np.diff(values.astype(int), axis=0) * direction > 0) & keep
    count = int(changed.sum())
    return float(keep.sum()) / count if count else None


def vital_rates_from_readings(
    scenarios: np.ndarray, readings: VitalReadings
) -> VitalRates:
    """Measure each rate in the scenario that shows it, from the readings."""

    def rate(
        scenario: str, vital: str, direction: int, keep: np.ndarray
    ) -> float | None:
        worlds = scenarios == VITAL_SCENARIOS.index(scenario)
        values = getattr(readings, vital)[:, worlds]
        return _steps_per_change(values, direction, keep[:, worlds])

    def both(array: np.ndarray) -> np.ndarray:
        """Whether ``array`` holds before and after each step."""

        return array[:-1] & array[1:]

    awake = both(~readings.sleeping)
    asleep = both(readings.sleeping)
    alive = both(readings.health > 0)
    fed = both(
        (readings.food > 0)
        & (readings.drink > 0)
        & ((readings.energy > 0) | readings.sleeping)
    )
    hurt = readings.health[:-1] < _MAX_VITAL

    plant = scenarios == VITAL_SCENARIOS.index("plant")
    ripening = []
    for world in np.flatnonzero(plant):
        facing = readings.facing[:, world]
        placed = np.flatnonzero(facing == BlockType.PLANT.value)
        ripe = np.flatnonzero(facing == BlockType.RIPE_PLANT.value)
        if placed.size and ripe.size:
            ripening.append(int(ripe[0] - placed[0]))

    return VitalRates(
        food_awake=rate("awake", "food", -1, awake & alive),
        drink_awake=rate("awake", "drink", -1, awake & alive),
        energy_awake=rate("awake", "energy", -1, awake & alive),
        food_asleep=rate("sleeping", "food", -1, asleep & alive),
        drink_asleep=rate("sleeping", "drink", -1, asleep & alive),
        energy_asleep=rate("asleep", "energy", 1, asleep & alive),
        health_awake=rate("awake", "health", 1, awake & fed & hurt),
        health_asleep=rate("sleeping", "health", 1, asleep & fed & hurt),
        health_starving=rate("starving", "health", -1, awake & ~fed & alive),
        plant_ripe=float(np.mean(ripening)) if ripening else None,
    )


def measure_vital_rates(
    *,
    worlds_per_scenario: int = 4,
    steps: int = DEFAULT_VITAL_STEPS,
    seed: int = VITAL_RATES_SEED,
) -> VitalRates:
    """Stage the scenarios, play them out and measure the rates."""

    scenarios, readings = run_vital_scenarios(
        worlds_per_scenario=worlds_per_scenario, steps=steps, seed=seed
    )
    return vital_rates_from_readings(scenarios, readings)
