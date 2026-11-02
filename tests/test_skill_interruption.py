import jax
import jax.numpy as jnp
import pytest
from craftax.craftax_classic.constants import BlockType

from tests.test_chains import _make_random_learned_executor
from tiny_sima.envs import make_open_ended_environment
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    ChainTermination,
    ScriptedSkillExecutor,
    SkillExecutionResult,
    SkillInterruption,
    _interrupts,
    evaluate_chain_episodes,
)
from tiny_sima.goals import SkillId
from tiny_sima.memory import EpisodeMemory, FailureCause
from tiny_sima.planner.saycan import (
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillAttempt,
    SkillOutcome,
)

RULE = SkillInterruption(threat_radius=3)
ENVIRONMENT = make_open_ended_environment(mobs=True)


def _without_mobs(state):
    def cleared(mobs):
        return mobs.replace(
            mask=jnp.zeros_like(mobs.mask), health=jnp.zeros_like(mobs.health)
        )

    return state.replace(
        zombies=cleared(state.zombies),
        cows=cleared(state.cows),
        skeletons=cleared(state.skeletons),
        mob_map=jnp.zeros_like(state.mob_map),
    )


def _with_zombie(state, offset):
    position = state.player_position + jnp.asarray(offset)
    zombies = state.zombies.replace(
        mask=state.zombies.mask.at[0].set(True),
        position=state.zombies.position.at[0].set(position),
        health=state.zombies.health.at[0].set(5),
        attack_cooldown=state.zombies.attack_cooldown.at[0].set(0),
    )
    world_map = state.map.at[position[0], position[1]].set(BlockType.GRASS.value)
    # Craftax checks the mob map to keep the player and mobs apart.
    mob_map = state.mob_map.at[position[0], position[1]].set(True)
    return state.replace(zombies=zombies, map=world_map, mob_map=mob_map)


def _cornered():
    """A zombie beside the player, who is walled in on the other three sides.

    Whatever the policy does, the player stays put and is hit on the first
    step.
    """

    state = _without_mobs(ENVIRONMENT.reset(jax.random.PRNGKey(0)).state)
    row, column = (int(value) for value in state.player_position)
    world_map = state.map
    for d_row, d_column in ((-1, 0), (1, 0), (0, -1)):
        world_map = world_map.at[row + d_row, column + d_column].set(
            BlockType.STONE.value
        )
    return _with_zombie(state.replace(map=world_map), (0, 1))


def test_a_zombie_arriving_or_a_hit_interrupts() -> None:
    clear = _without_mobs(ENVIRONMENT.reset(jax.random.PRNGKey(0)).state)
    far, near = _with_zombie(clear, (0, 4)), _with_zombie(clear, (3, -3))
    assert bool(_interrupts(RULE, far, near))
    assert bool(_interrupts(RULE, clear, near))
    # A zombie that was already close does not interrupt again.
    assert not bool(_interrupts(RULE, near, _with_zombie(clear, (0, 2))))
    assert not bool(_interrupts(RULE, near, far))

    hit = clear.replace(player_health=clear.player_health - 2)
    assert bool(_interrupts(RULE, clear, hit))
    # Health lost to thirst is not a hit.
    parched = clear.replace(player_drink=jnp.asarray(0, dtype=jnp.int32))
    assert not bool(
        _interrupts(RULE, parched, parched.replace(player_health=hit.player_health))
    )


@pytest.mark.parametrize(
    "make_executor", [ScriptedSkillExecutor, _make_random_learned_executor]
)
def test_a_hit_hands_the_skill_back(make_executor) -> None:
    executor = make_executor()
    state = _cornered()
    rng = jax.random.PRNGKey(1)

    result = executor.execute(
        ENVIRONMENT, state, SkillId.COLLECT_WOOD, rng, horizon=16, interruption=RULE
    )
    assert result.outcome is SkillOutcome.INTERRUPTED
    assert result.steps == 1
    assert int(result.state.player_health) == int(state.player_health) - 2

    uninterrupted = executor.execute(
        ENVIRONMENT, state, SkillId.COLLECT_WOOD, rng, horizon=16
    )
    assert uninterrupted.outcome is not SkillOutcome.INTERRUPTED
    assert uninterrupted.steps > 1


def test_the_fight_itself_is_not_interrupted() -> None:
    result = ScriptedSkillExecutor().execute(
        ENVIRONMENT,
        _cornered(),
        SkillId.DEFEAT_ZOMBIE,
        jax.random.PRNGKey(1),
        horizon=32,
        interruption=RULE,
    )
    assert result.outcome is SkillOutcome.SUCCESS


def test_an_interrupted_attempt_is_not_a_failure() -> None:
    attempt = SkillAttempt(SkillId.COLLECT_WOOD, SkillOutcome.INTERRUPTED, 1, 0.0)
    assert not attempt.succeeded and not attempt.failed

    state = _cornered()
    memory = EpisodeMemory().observe(state)
    recorded = memory.record(
        state, SkillId.COLLECT_WOOD, SkillOutcome.INTERRUPTED, 1, state
    )
    assert recorded.transitions[-1].cause == FailureCause.INTERRUPTED.value
    assert recorded.misses == memory.misses


class _AlwaysInterrupted:
    def execute(self, environment, state, skill_id, rng, *, horizon, interruption):
        assert interruption is RULE
        return SkillExecutionResult(
            state=state,
            outcome=SkillOutcome.INTERRUPTED,
            steps=1,
            skill_return=0.0,
            native_return=0.0,
        )


def test_interruptions_do_not_reach_the_failure_limit() -> None:
    planner = SayCanPlanner(
        utility_scorer=SequentialUtilityScorer((SkillId.COLLECT_WOOD,))
    )
    (episode,) = evaluate_chain_episodes(
        planner,
        _AlwaysInterrupted(),
        make_skill_chain_environment(),
        [jax.random.PRNGKey(0)],
        max_decisions=3,
        max_failures_per_skill=1,
        interruption=RULE,
    )
    assert episode.termination is ChainTermination.DECISION_LIMIT
    assert episode.decisions == 3
    # SayCan does not penalise the interrupted skill either.
    assert all(
        score.failure_penalty == 1.0
        for step in episode.trace
        for score in step.candidate_scores
    )
