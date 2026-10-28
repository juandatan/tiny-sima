import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
import pytest

from tiny_sima.evaluation.chains import (
    SKILL_CHAINS,
    STONE_PICKAXE_AND_SWORD_CHAIN,
    TERSE_STONE_PICKAXE_CHAIN,
    WOOD_TO_STONE_CHAIN,
    WOOD_TO_STONE_PICKAXE_CHAIN,
    ChainTermination,
    ScriptedSkillExecutor,
    SkillChainSpec,
    chain_decision_limit,
    chain_environment,
    chain_episode_record,
    evaluate_chain_episodes,
    evaluate_skill_chain,
    json_ready_chain_result,
)
from tiny_sima.evaluation.executor_diagnostics import COLLECT_FIRST_PLANS, chain_plans
from tiny_sima.evaluation.sequencing import episode_sequence
from tiny_sima.goals import SkillId
from tiny_sima.planner.saycan import (
    PlannerContext,
    PlannerDecision,
    PlannerSelection,
    PlannerSelectionStatus,
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillOutcome,
    SkillScore,
)
from tiny_sima.planner.symbolic import shortest_plan

NEW_CHAINS = (TERSE_STONE_PICKAXE_CHAIN, STONE_PICKAXE_AND_SWORD_CHAIN)


def _state(chain: SkillChainSpec, seed: int = 0):
    return chain_environment(chain).reset(jax.random.PRNGKey(seed)).state


def _inventory(state) -> dict[str, int]:
    return {
        name: int(np.asarray(getattr(state.inventory, name)))
        for name in ("wood", "stone", "wood_pickaxe", "wood_sword", "stone_pickaxe")
    }


@dataclass(frozen=True)
class _FakeMemory:
    """Logs every update to a list shared by all its successors."""

    log: list[tuple[Any, ...]]
    updates: int = 0

    def observe(self, state: Any) -> "_FakeMemory":
        self.log.append(("observe", _inventory(state)))
        return _FakeMemory(self.log, self.updates + 1)

    def record(
        self,
        pre_state: Any,
        skill_id: SkillId,
        outcome: SkillOutcome,
        steps: int,
        post_state: Any,
    ) -> "_FakeMemory":
        self.log.append(
            (
                "record",
                _inventory(pre_state),
                skill_id,
                outcome,
                steps,
                _inventory(post_state),
            )
        )
        return _FakeMemory(self.log, self.updates + 1)


@dataclass(frozen=True)
class _Counter:
    count: int

    def to_dict(self) -> dict[str, int]:
        return {"count": self.count}


@dataclass
class _ScriptedPlanner:
    """Follows a fixed plan, counting its selections in plan_state."""

    plan: tuple[SkillId, ...]
    contexts: list[PlannerContext] = field(default_factory=list)

    def select_many(
        self,
        states: Sequence[Any],
        contexts: Sequence[PlannerContext],
    ) -> list[PlannerSelection]:
        selections = []
        for context in contexts:
            self.contexts.append(context)
            count = 0 if context.plan_state is None else context.plan_state.count
            plan_state = _Counter(count + 1)
            if count >= len(self.plan):
                selections.append(
                    PlannerSelection(
                        decision=None,
                        status=PlannerSelectionStatus.NO_VIABLE_SKILL,
                        plan_state=plan_state,
                    )
                )
                continue
            skill_id = self.plan[count]
            score = SkillScore(skill_id, 1.0, 1.0, 1.0, 1.0)
            selections.append(
                PlannerSelection(
                    decision=PlannerDecision(skill_id=skill_id, scores=(score,)),
                    status=PlannerSelectionStatus.SELECTED,
                    plan_state=plan_state,
                )
            )
        return selections


def _run(planner, chain: SkillChainSpec, memory_factory=None, episodes: int = 1):
    return evaluate_chain_episodes(
        planner,
        ScriptedSkillExecutor(),
        chain_environment(chain),
        list(jax.random.split(jax.random.PRNGKey(7), episodes)),
        chain=chain,
        memory_factory=memory_factory,
    )


def test_memory_observes_the_reset_and_records_each_skill() -> None:
    chain = WOOD_TO_STONE_CHAIN
    log: list[tuple[Any, ...]] = []
    planner = _ScriptedPlanner(chain.reference_skills)
    (episode,) = _run(planner, chain, memory_factory=lambda: _FakeMemory(log))

    assert episode.succeeded
    assert log[0][0] == "observe"
    records = log[1:]
    assert [entry[0] for entry in records] == ["record"] * len(episode.trace)
    previous = log[0][1]
    for entry, trace in zip(records, episode.trace, strict=True):
        _, pre, skill_id, outcome, steps, post = entry
        assert pre == previous
        assert (skill_id, outcome, steps) == (
            trace.requested_skill,
            trace.outcome,
            trace.steps,
        )
        previous = post
    assert previous["stone"] >= 1
    # Each context carries the memory updated through the previous boundary.
    memories = [
        memory
        for context in planner.contexts
        if isinstance(memory := context.memory, _FakeMemory)
    ]
    assert len(memories) == len(planner.contexts)
    assert [memory.updates for memory in memories] == list(range(1, len(memories) + 1))


def test_memory_observe_sees_the_reset_state() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    log: list[tuple[Any, ...]] = []
    _run(_ScriptedPlanner(()), chain, memory_factory=lambda: _FakeMemory(log))
    reset = chain_environment(chain).reset(
        jax.random.split(jax.random.PRNGKey(7), 1)[0]
    )
    assert log == [("observe", _inventory(reset.state))]


def test_each_episode_gets_its_own_memory() -> None:
    chain = WOOD_TO_STONE_CHAIN
    logs: list[list[tuple[Any, ...]]] = []

    def factory() -> _FakeMemory:
        logs.append([])
        return _FakeMemory(logs[-1])

    episodes = _run(
        _ScriptedPlanner(chain.reference_skills), chain, factory, episodes=3
    )
    assert len(logs) == 3
    for log, episode in zip(logs, episodes, strict=True):
        assert len(log) == 1 + len(episode.trace)


def test_plan_state_reaches_the_next_context_trace_and_record() -> None:
    chain = WOOD_TO_STONE_CHAIN
    # One skill short of the plan, so the last selection ends the episode.
    plan = chain.reference_skills[:-1]
    planner = _ScriptedPlanner(plan)
    (episode,) = _run(planner, chain)

    assert episode.termination is ChainTermination.NO_VIABLE_SKILL
    assert [context.plan_state for context in planner.contexts] == [
        None,
        *(_Counter(index) for index in range(1, len(plan) + 1)),
    ]
    assert [trace.plan_state for trace in episode.trace] == [
        _Counter(index) for index in range(1, len(plan) + 1)
    ]
    assert episode.final_plan_state == _Counter(len(plan) + 1)
    assert all(context.memory is None for context in planner.contexts)

    record = json.loads(json.dumps(chain_episode_record(episode)))
    assert [step["plan_state"] for step in record["trace"]] == [
        {"count": index} for index in range(1, len(plan) + 1)
    ]
    assert record["final_plan_state"] == {"count": len(plan) + 1}


def test_plan_state_without_to_dict_is_left_out_of_the_record() -> None:
    @dataclass
    class Opaque(_ScriptedPlanner):
        def select_many(self, states, contexts):
            return [
                PlannerSelection(
                    decision=selection.decision,
                    status=selection.status,
                    plan_state=object(),
                )
                for selection in super().select_many(
                    states,
                    [
                        PlannerContext(
                            objective=context.objective,
                            history=context.history,
                            plan_state=_Counter(len(context.history)),
                        )
                        for context in contexts
                    ],
                )
            ]

    (episode,) = _run(Opaque(WOOD_TO_STONE_CHAIN.reference_skills), WOOD_TO_STONE_CHAIN)
    record = chain_episode_record(episode)
    assert episode.succeeded
    assert all("plan_state" not in step for step in record["trace"])
    assert "final_plan_state" not in record


def test_context_fields_stay_out_of_equality() -> None:
    log: list[tuple[Any, ...]] = []
    plain = PlannerContext(objective="x")
    loaded = PlannerContext(objective="x", memory=_FakeMemory(log), plan_state=[1])

    assert plain == loaded
    assert hash(plain) == hash(loaded)


@pytest.mark.parametrize(
    "chain", (WOOD_TO_STONE_CHAIN, WOOD_TO_STONE_PICKAXE_CHAIN), ids=lambda c: c.name
)
def test_the_default_path_is_unchanged(chain) -> None:
    def evaluate(memory_factory=None):
        planner = SayCanPlanner(
            utility_scorer=SequentialUtilityScorer(chain.reference_skills),
            candidate_skills=chain.skills,
        )
        return json_ready_chain_result(
            evaluate_skill_chain(
                ScriptedSkillExecutor(),
                planner=planner,
                chain=chain,
                num_episodes=2,
                seed=7,
                memory_factory=memory_factory,
            )
        )

    default = evaluate()
    assert default["success_rate"] == 1.0
    for episode in default["episodes"]:
        assert "final_plan_state" not in episode
        assert all("plan_state" not in step for step in episode["trace"])
    # A memory the planner ignores changes nothing either.
    assert evaluate(lambda: _FakeMemory([])) == default


def test_new_chains_are_registered() -> None:
    for chain in NEW_CHAINS:
        assert SKILL_CHAINS[chain.name] is chain
        assert chain_plans(chain)["collect_first"] is COLLECT_FIRST_PLANS[chain.name]
    assert TERSE_STONE_PICKAXE_CHAIN.objective == "Get a stone pickaxe."
    assert (
        TERSE_STONE_PICKAXE_CHAIN._replace(
            name=WOOD_TO_STONE_PICKAXE_CHAIN.name,
            objective=WOOD_TO_STONE_PICKAXE_CHAIN.objective,
        )
        == WOOD_TO_STONE_PICKAXE_CHAIN
    )
    assert chain_decision_limit(STONE_PICKAXE_AND_SWORD_CHAIN) == 20
    assert chain_environment(STONE_PICKAXE_AND_SWORD_CHAIN).tree_count == 5


@pytest.mark.parametrize("chain", NEW_CHAINS, ids=lambda chain: chain.name)
@pytest.mark.parametrize("plan_name", ("default", "collect_first"))
def test_scripted_plans_complete_the_new_chains(chain, plan_name) -> None:
    plan = chain_plans(chain)[plan_name]
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(),
        planner=SayCanPlanner(
            utility_scorer=SequentialUtilityScorer(plan),
            candidate_skills=chain.skills,
        ),
        chain=chain,
        num_episodes=2,
        seed=7,
    )

    assert result["success_rate"] == 1.0
    for episode in result["episodes"]:
        assert [step.requested_skill for step in episode.trace] == list(plan)


@pytest.mark.parametrize("chain", NEW_CHAINS, ids=lambda chain: chain.name)
def test_reference_length_is_the_shortest_plan(chain) -> None:
    assert chain.goal is not None
    for seed in range(3):
        plan = shortest_plan(_state(chain, seed), chain.goal, chain.skills)
        assert plan == (frozenset((SkillId.COLLECT_WOOD,)), len(chain.reference_skills))


@pytest.mark.parametrize("chain", NEW_CHAINS, ids=lambda chain: chain.name)
def test_completion_predicates_agree(chain) -> None:
    assert chain.is_complete_jax is not None
    state = _state(chain)
    for counts in (
        {},
        {"stone_pickaxe": 1},
        {"wood_sword": 1},
        {"stone_pickaxe": 1, "wood_sword": 1},
    ):
        edited = state.replace(inventory=state.inventory.replace(**counts))
        assert chain.is_complete(edited) == bool(
            np.asarray(chain.is_complete_jax(edited))
        )
    sword = STONE_PICKAXE_AND_SWORD_CHAIN
    assert not sword.is_complete(
        state.replace(inventory=state.inventory.replace(stone_pickaxe=1))
    )


@pytest.mark.parametrize("chain", NEW_CHAINS, ids=lambda chain: chain.name)
def test_valid_next_skills_at_reset(chain) -> None:
    assert chain.valid_next_skills is not None
    assert chain.valid_next_skills(_state(chain)) == (SkillId.COLLECT_WOOD,)


def test_the_sword_chain_accepts_either_craft_first() -> None:
    chain = STONE_PICKAXE_AND_SWORD_CHAIN
    assert chain.valid_next_skills is not None
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(), chain=chain, num_episodes=1, seed=7
    )
    (episode,) = json_ready_chain_result(result)["episodes"]
    assert episode_sequence(episode) == (
        "wood wood table wood pickaxe wood sword wood stone stone_pickaxe"
    )
    # Every accepted set is non-empty and in catalog order.
    accepted = [tuple(step["accepted_skills"]) for step in episode["trace"]]
    assert all(accepted)
    assert all(
        list(skills) == sorted(skills, key=[s.value for s in chain.skills].index)
        for skills in accepted
    )
