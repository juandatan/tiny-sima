import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import BlockType

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.goals import SkillId
from tiny_sima.memory import EpisodeMemory, FailureCause, classify_failure, skill_target
from tiny_sima.planner.saycan import SkillOutcome


def _grass_world(**blocks: list[tuple[int, int]]):
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    world_map = np.full(np.asarray(state.map).shape, BlockType.GRASS.value)
    row, column = np.asarray(state.player_position).tolist()
    for name, offsets in blocks.items():
        for d_row, d_column in offsets:
            world_map[row + d_row, column + d_column] = BlockType[name.upper()].value
    return state.replace(map=jnp.asarray(world_map, dtype=state.map.dtype))


def _classify(pre, outcome, post=None, skill_id=SkillId.COLLECT_WOOD, memory=None):
    return classify_failure(
        pre,
        skill_id,
        outcome,
        10,
        pre if post is None else post,
        EpisodeMemory() if memory is None else memory,
    )


def test_skill_targets_come_from_the_registry() -> None:
    assert {skill_id: skill_target(skill_id) for skill_id in SkillId} == {
        SkillId.COLLECT_WOOD: "tree",
        SkillId.COLLECT_STONE: "stone",
        SkillId.DRINK_WATER: "water",
        SkillId.PLACE_TABLE: None,
        SkillId.CRAFT_WOOD_PICKAXE: "crafting_table",
        SkillId.CRAFT_WOOD_SWORD: "crafting_table",
        SkillId.CRAFT_STONE_PICKAXE: "crafting_table",
        SkillId.CRAFT_STONE_SWORD: "crafting_table",
        SkillId.COLLECT_COAL: "coal",
        SkillId.PLACE_STONE: None,
        SkillId.PLACE_FURNACE: None,
        # Grass and cows are not remembered blocks.
        SkillId.COLLECT_SAPLING: None,
        SkillId.PLACE_PLANT: None,
        SkillId.EAT_COW: None,
        SkillId.SLEEP: None,
        SkillId.EAT_PLANT: "ripe_plant",
    }


def test_a_success_has_no_cause() -> None:
    state = _grass_world()
    assert _classify(state, SkillOutcome.SUCCESS) is None
    assert _classify(state, "success", state.replace(player_health=0)) is None


def test_timeouts_are_explained_by_the_target() -> None:
    bare = _grass_world()
    assert _classify(bare, SkillOutcome.TIMEOUT) is FailureCause.TARGET_NOT_SEEN

    # Known from an earlier sighting, now out of view.
    seen = EpisodeMemory().observe(_grass_world(tree=[(1, 1)]))
    away = bare.replace(player_position=bare.player_position + jnp.array([0, 10]))
    assert (
        _classify(away, "timeout", memory=seen) is FailureCause.TARGET_NOT_REACHED
    )
    # In view at the decision counts as known.
    visible = _grass_world(tree=[(2, 0)])
    assert _classify(visible, "timeout") is FailureCause.TARGET_NOT_REACHED

    # Crafting targets the table.
    table = _grass_world(crafting_table=[(0, 1)])
    for state, cause in (
        (bare, FailureCause.TARGET_NOT_SEEN),
        (table, FailureCause.TARGET_NOT_REACHED),
    ):
        assert _classify(state, "timeout", skill_id=SkillId.CRAFT_WOOD_SWORD) is cause
    # A skill without a target block times out plainly.
    assert _classify(bare, "timeout", skill_id=SkillId.PLACE_TABLE) is (
        FailureCause.TIMEOUT
    )


def test_outcome_causes() -> None:
    state = _grass_world()
    assert _classify(state, SkillOutcome.INFEASIBLE) is FailureCause.INFEASIBLE
    assert (
        _classify(state, SkillOutcome.PRECONDITION_LOST)
        is FailureCause.PRECONDITION_LOST
    )
    # Terminated while alive is the episode's time limit.
    assert (
        _classify(state, SkillOutcome.ENVIRONMENT_TERMINATED)
        is FailureCause.EPISODE_ENDED
    )


def test_death_and_vitals_come_before_the_target() -> None:
    pre = _grass_world()

    # Death ends the request as a lost precondition or a termination.
    for outcome in ("precondition_lost", "environment_terminated", "timeout"):
        assert _classify(pre, outcome, pre.replace(player_health=0)) is (
            FailureCause.DIED
        )
    in_lava = _grass_world(lava=[(0, 0)])
    assert _classify(pre, "environment_terminated", in_lava) is FailureCause.DIED

    for vital, cause in (
        ("food", FailureCause.FOOD_DEPLETED),
        ("drink", FailureCause.DRINK_DEPLETED),
        ("energy", FailureCause.ENERGY_DEPLETED),
    ):
        post = pre.replace(**{f"player_{vital}": 0})
        assert _classify(pre, "timeout", post) is cause
        # Already out before the request, it does not explain this failure.
        assert _classify(post, "timeout", post) is FailureCause.TARGET_NOT_SEEN

    # Food runs out first, so it explains the health it costs.
    starving = pre.replace(player_food=0, player_health=2)
    assert _classify(pre, "timeout", starving) is FailureCause.FOOD_DEPLETED

    hurt = pre.replace(player_health=3)
    assert _classify(pre, "timeout", hurt) is FailureCause.HEALTH_LOW
    # A small loss, or low health that did not fall, is not the cause.
    assert _classify(pre, "timeout", pre.replace(player_health=5)) is (
        FailureCause.TARGET_NOT_SEEN
    )
    assert _classify(hurt, "timeout", hurt) is FailureCause.TARGET_NOT_SEEN
