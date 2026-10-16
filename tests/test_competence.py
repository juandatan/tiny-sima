"""Tests for the measured SayCan competence head."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from craftax.craftax_classic.constants import BlockType

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.goals import SKILL_IDS, ReachabilityConfig, SkillId
from tiny_sima.planner import (
    CompetencePrior,
    PlannerContext,
    SayCanPlanner,
    distance_cell,
    fit_competence_prior,
    num_distance_cells,
)


def _chain_state() -> Any:
    return make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state


def _state_with_distant_trees(radius: int = 12) -> Any:
    """Clear trees near the player, leaving the precondition satisfied."""

    state = _chain_state()
    world_map = jnp.asarray(state.map)
    position = jnp.asarray(state.player_position)
    rows = jnp.arange(world_map.shape[0])[:, None]
    columns = jnp.arange(world_map.shape[1])[None, :]
    nearby = (jnp.abs(rows - position[0]) <= radius) & (
        jnp.abs(columns - position[1]) <= radius
    )
    cleared = jnp.where(
        nearby & (world_map == BlockType.TREE.value),
        BlockType.GRASS.value,
        world_map,
    )
    assert bool(jnp.any(cleared == BlockType.TREE.value))
    return state.replace(map=cleared)


def _labelled(
    skill_id: SkillId,
    steps: int | None,
    successes: int,
    failures: int,
) -> tuple[list[SkillId], list[int | None], list[bool]]:
    count = successes + failures
    return (
        [skill_id] * count,
        [steps] * count,
        [True] * successes + [False] * failures,
    )


def _merge(
    *batches: tuple[list[SkillId], list[int | None], list[bool]],
) -> tuple[list[SkillId], list[int | None], list[bool]]:
    skills: list[SkillId] = []
    steps: list[int | None] = []
    labels: list[bool] = []
    for batch_skills, batch_steps, batch_labels in batches:
        skills += batch_skills
        steps += batch_steps
        labels += batch_labels
    return skills, steps, labels


def test_distance_cell_reserves_column_zero_for_unreachable() -> None:
    edges = (2, 6)

    assert distance_cell(None, edges) == 0
    assert distance_cell(1, edges) == 1
    assert distance_cell(2, edges) == 1
    assert distance_cell(3, edges) == 2
    assert distance_cell(6, edges) == 2
    assert distance_cell(7, edges) == 3
    assert num_distance_cells(edges) == 4


def test_well_observed_cells_keep_their_own_rate() -> None:
    skills, steps, labels = _merge(
        _labelled(SkillId.COLLECT_WOOD, 1, successes=180, failures=20),
        _labelled(SkillId.COLLECT_WOOD, None, successes=4, failures=196),
    )

    prior = fit_competence_prior(skills, steps, labels, prior_strength=4.0)

    # Two hundred samples against a pseudo-count of four: shrinkage moves the
    # estimate by about a percentage point, no more.
    assert prior.success_probability(SkillId.COLLECT_WOOD, 1) == pytest.approx(
        0.9,
        abs=0.02,
    )
    assert prior.success_probability(SkillId.COLLECT_WOOD, None) == pytest.approx(
        0.02,
        abs=0.02,
    )


def test_an_unobserved_cell_falls_back_to_its_parent() -> None:
    skills, steps, labels = _merge(
        # Only the nearest band is ever observed for wood.
        _labelled(SkillId.COLLECT_WOOD, 1, successes=90, failures=10),
        _labelled(SkillId.DRINK_WATER, 1, successes=10, failures=90),
        _labelled(SkillId.COLLECT_WOOD, None, successes=0, failures=50),
    )

    prior = fit_competence_prior(skills, steps, labels)

    far = prior.success_probability(SkillId.COLLECT_WOOD, 9)
    near = prior.success_probability(SkillId.COLLECT_WOOD, 1)
    assert prior.support(SkillId.COLLECT_WOOD, 9) == 0
    # With no evidence the far band inherits wood's own reachable rate rather
    # than collapsing to zero or to the pooled rate across skills.
    assert far == pytest.approx(near, abs=0.05)
    # A skill never observed at all still gets the reachability-level rate.
    unseen = prior.success_probability(SkillId.PLACE_TABLE, 1)
    assert 0.0 < unseen < 1.0


def test_the_floor_keeps_an_unreachable_skill_selectable() -> None:
    skills, steps, labels = _labelled(
        SkillId.COLLECT_WOOD,
        None,
        successes=0,
        failures=400,
    )

    prior = fit_competence_prior(skills, steps, labels, floor=0.02)

    assert prior.success_probability(SkillId.COLLECT_WOOD, None) == 0.02
    state = _state_with_distant_trees()
    context = PlannerContext(objective="Collect wood.")

    def only_wood(
        _state: Any,
        _context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        return float(skill_id is SkillId.COLLECT_WOOD)

    decision = SayCanPlanner(
        utility_scorer=only_wood,
        feasibility_scorer=prior,
    ).select(state, context)

    # Walking off to find a tree is the only plan on offer, so a measured zero
    # must not turn into the hard veto that cost whole episodes.
    assert decision is not None
    assert decision.skill_id is SkillId.COLLECT_WOOD


def test_scoring_vetoes_on_the_symbolic_precondition() -> None:
    skills, steps, labels = _merge(
        *(
            _labelled(skill_id, 1, successes=8, failures=2)
            for skill_id in SKILL_IDS
        ),
    )
    prior = fit_competence_prior(skills, steps, labels)
    state = _chain_state()
    context = PlannerContext(objective="Collect wood.")

    scores = prior.score_all_feasibility(state, context, SKILL_IDS)

    # No crafting table exists yet, and no distance makes crafting possible.
    assert scores[SkillId.CRAFT_WOOD_PICKAXE] == 0.0
    assert scores[SkillId.COLLECT_STONE] == 0.0
    assert scores[SkillId.COLLECT_WOOD] > 0.0


def test_scoring_reads_the_distance_band_from_the_state() -> None:
    skills, steps, labels = _merge(
        _labelled(SkillId.COLLECT_WOOD, 1, successes=95, failures=5),
        _labelled(SkillId.COLLECT_WOOD, None, successes=5, failures=95),
    )
    prior = fit_competence_prior(skills, steps, labels)
    context = PlannerContext(objective="Collect wood.")

    near = prior.score_all_feasibility(
        _chain_state(),
        context,
        (SkillId.COLLECT_WOOD,),
    )
    far = prior.score_all_feasibility(
        _state_with_distant_trees(),
        context,
        (SkillId.COLLECT_WOOD,),
    )

    assert near[SkillId.COLLECT_WOOD] > 0.8
    assert far[SkillId.COLLECT_WOOD] < 0.2


def test_prior_round_trips_through_disk(tmp_path: Path) -> None:
    skills, steps, labels = _merge(
        _labelled(SkillId.COLLECT_WOOD, 1, successes=9, failures=1),
        _labelled(SkillId.DRINK_WATER, 7, successes=1, failures=9),
        _labelled(SkillId.PLACE_TABLE, None, successes=0, failures=10),
    )
    prior = fit_competence_prior(skills, steps, labels)
    path = tmp_path / "nested" / "competence_prior.json"

    prior.save(path)
    loaded = CompetencePrior.load(path)

    assert loaded.rates == prior.rates
    assert loaded.counts == prior.counts
    assert loaded.distance_edges == prior.distance_edges
    assert loaded.base_rate == pytest.approx(prior.base_rate)
    for skill_id in SKILL_IDS:
        for cell in range(num_distance_cells(prior.distance_edges)):
            assert loaded.rate_by_cell(skill_id, cell) == pytest.approx(
                prior.rate_by_cell(skill_id, cell)
            )


def test_prior_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="same probes"):
        fit_competence_prior([SkillId.COLLECT_WOOD], [1, 2], [True])
    with pytest.raises(ValueError, match="zero samples"):
        fit_competence_prior([], [], [])
    with pytest.raises(ValueError, match="prior_strength"):
        fit_competence_prior(
            [SkillId.COLLECT_WOOD],
            [1],
            [True],
            prior_strength=0.0,
        )
    with pytest.raises(ValueError, match="distance_edges"):
        fit_competence_prior(
            [SkillId.COLLECT_WOOD],
            [1],
            [True],
            distance_edges=(6, 2),
        )
    with pytest.raises(ValueError, match="one row per catalog skill"):
        CompetencePrior(rates=((0.5,) * 4,), counts=((1,) * 4,), base_rate=0.5)


def test_prior_accepts_a_custom_reachability_config() -> None:
    skills, steps, labels = _labelled(
        SkillId.COLLECT_WOOD,
        1,
        successes=9,
        failures=1,
    )
    config = ReachabilityConfig(max_steps=128, view_radius=None)
    prior = fit_competence_prior(skills, steps, labels, config=config)
    context = PlannerContext(objective="Collect wood.")

    # Searching the whole map finds a tree that the observation window hides,
    # so the same state scores as reachable rather than unreachable.
    scores = prior.score_all_feasibility(
        _state_with_distant_trees(),
        context,
        (SkillId.COLLECT_WOOD,),
    )

    assert scores[SkillId.COLLECT_WOOD] > 0.0
    assert prior.config is config
