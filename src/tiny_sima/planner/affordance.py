"""A measured dead end: affordances read from the goal-conditioned critic.

**This is not the default Can term.** It is retained as the ablation that
measured its own failure, and as the only honest record of why the planner uses
:class:`~tiny_sima.planner.competence.CompetencePrior` instead. Reach for that
one; read this for the argument it refutes.

The reachability gate in :mod:`tiny_sima.goals.reachability` answers whether a
skill is *possible*; it says nothing about whether this executor can actually
do it. That information looked like it already existed in the trained critic.
The skill environment pays a single sparse reward on success and terminates
there, so

    V(s, skill) = E[gamma ** k | success] * P(success)

which would rank skills by competence without training a separate SayCan
success head. The absolute scale is wrong — discounting squashes slow successes
toward zero, and at ``gamma = 0.95`` a certain success 32 steps out is worth
0.19 — so raw values are recalibrated against observed outcomes by
:mod:`tiny_sima.planner.calibration` before use.

The derivation holds only on the distribution the critic was trained on, and
that is where it broke. Probing 432 boundary states (notebook 06) gave:

* pooled AUC 0.428, and every per-skill AUC within noise of 0.5, so the critic
  separates success no better than a coin;
* mean values spanning 0.78 (``collect_wood``) to 0.995 (``place_table``) while
  their success rates ran the *other* way, 0.63 down to 0.17 — the critic
  encodes which skill it was asked for, not whether this state admits it;
* isotonic regression must fit a non-decreasing map, so on a decreasing
  relationship it pooled every bin to the global mean. The output collapsed to
  two values inside ``[0.240, 0.276]``, a Brier score matching the base rate to
  three decimals, and an in-sample calibration error of zero.

A factor varying by 1.15x cannot reorder candidates that reciprocal-square rank
utility separates by 4x, which is why the ``vlm_plus_critic_affordance``
condition scored identically to ``vlm_plus_reachability``. The cause is
off-distribution overestimation: ``local_navigation`` training always places a
reachable target, so the critic never learned to say "not from here."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.executor.model import (
    GoalConditionedActorCriticRNN,
    apply_goal_conditioned_actor_critic,
    initial_hidden,
    make_goal_conditioned_actor_critic,
)
from tiny_sima.goals import (
    SKILL_IDS,
    SkillId,
    skill_index,
)
from tiny_sima.planner.calibration import IsotonicCalibration
from tiny_sima.planner.saycan import (
    PlannerContext,
    ReachabilityFeasibilityScorer,
    reachability_feasibility,
)

if TYPE_CHECKING:
    from tiny_sima.evaluation.goal_conditioned import (
        LoadedGoalConditionedCheckpoint,
    )


@dataclass(frozen=True)
class CriticAffordanceScorer:
    """Score feasibility from calibrated goal-conditioned critic values.

    Kept as an ablation, not as a recommended Can term -- see the module
    docstring for the measurement that retired it.

    The hidden state is reset to zero for every query because the planner only
    ever acts at an option boundary, which is exactly where the executor starts
    a fresh skill with no recurrent history.

    ``reachability_gate`` multiplies the calibrated probability by a graded
    search-based factor: one when a target is usable from here, a fixed
    ``unknown_feasibility`` when it is not in view, and zero only when the
    symbolic precondition fails. Set it to ``None`` to measure the critic on
    its own, which is the Phase 4 ablation that isolates how much of the
    grounding comes from the executor rather than from search.
    """

    params: Any
    model: GoalConditionedActorCriticRNN
    hidden_size: int
    observe: Any
    calibration: IsotonicCalibration | None = None
    reachability_gate: ReachabilityFeasibilityScorer | None = (
        reachability_feasibility
    )
    candidate_skills: tuple[SkillId, ...] = SKILL_IDS
    _value_fn: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        def critic_values(
            params: Any,
            observation: jax.Array,
            categorical_skill_ids: jax.Array,
        ) -> jax.Array:
            batch = categorical_skill_ids.shape[0]
            # One recurrent step at an option boundary: time axis of length one,
            # batch axis over the candidate skills, resets everywhere.
            observations = jnp.broadcast_to(
                observation[None, None, ...],
                (1, batch, *observation.shape),
            )
            output = apply_goal_conditioned_actor_critic(
                self.model,
                params,
                initial_hidden(batch, self.hidden_size),
                (
                    observations,
                    categorical_skill_ids[None, :],
                    jnp.ones((1, batch), dtype=jnp.bool_),
                ),
            )
            return output.value[0]

        object.__setattr__(self, "_value_fn", jax.jit(critic_values))

    def critic_values(
        self,
        state: Any,
        candidate_skills: tuple[SkillId, ...] | None = None,
    ) -> dict[SkillId, float]:
        """Return raw, uncalibrated critic values for each candidate skill."""

        skills = self.candidate_skills if candidate_skills is None else candidate_skills
        if not skills:
            return {}
        if self._value_fn is None:
            raise RuntimeError("Critic affordance scorer was not initialized.")
        values = self._value_fn(
            self.params,
            jnp.asarray(self.observe(state)),
            jnp.asarray([skill_index(skill) for skill in skills], dtype=jnp.int32),
        )
        return {
            skill: float(value)
            for skill, value in zip(skills, np.asarray(values), strict=True)
        }

    def score_all_feasibility(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> dict[SkillId, float]:
        """Return calibrated success probabilities gated on reachability."""

        if not candidate_skills:
            return {}
        values = self.critic_values(state, candidate_skills)
        if self.calibration is not None:
            raw = np.asarray(
                [values[skill] for skill in candidate_skills],
                dtype=np.float64,
            )
            calibrated = self.calibration.apply(raw)
            scores = {
                skill: float(probability)
                for skill, probability in zip(
                    candidate_skills,
                    calibrated,
                    strict=True,
                )
            }
        else:
            # Without a fitted calibration the value is still monotone in
            # success probability, so clipping keeps the ranking usable.
            scores = {
                skill: float(np.clip(value, 0.0, 1.0))
                for skill, value in values.items()
            }
        if self.reachability_gate is not None:
            gates = self.reachability_gate.score_all_feasibility(
                state,
                context,
                candidate_skills,
            )
            scores = {
                skill: score * gates.get(skill, 0.0)
                for skill, score in scores.items()
            }
        return scores

    def __call__(
        self,
        state: Any,
        context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        """Score one candidate skill."""

        return self.score_all_feasibility(state, context, (skill_id,))[skill_id]


def make_critic_affordance_scorer(
    checkpoint: LoadedGoalConditionedCheckpoint,
    *,
    calibration: IsotonicCalibration | None = None,
    reachability_gate: ReachabilityFeasibilityScorer | None = (
        reachability_feasibility
    ),
) -> CriticAffordanceScorer:
    """Build an affordance scorer from an already-loaded checkpoint."""

    environment = make_skill_chain_environment()
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=checkpoint.config.hidden_size,
        skill_embedding_size=checkpoint.config.skill_embedding_size,
    )
    return CriticAffordanceScorer(
        params=checkpoint.params,
        model=model,
        hidden_size=checkpoint.config.hidden_size,
        observe=environment.bundle.env.get_obs,
        calibration=calibration,
        reachability_gate=reachability_gate,
    )


def critic_affordance_scorer_from_checkpoint(
    checkpoint_dir: str | Path,
    *,
    calibration: IsotonicCalibration | None = None,
    calibration_path: str | Path | None = None,
    reachability_gate: ReachabilityFeasibilityScorer | None = (
        reachability_feasibility
    ),
) -> CriticAffordanceScorer:
    """Load an executor checkpoint and adapt its critic to an affordance model.

    ``calibration_path`` reads a calibration saved by
    :meth:`IsotonicCalibration.save`, which is how a fit produced on the
    training machine is reused at planning time.
    """

    # Imported here rather than at module scope: checkpoint loading lives in
    # ``tiny_sima.evaluation``, which imports this package in turn.
    from tiny_sima.evaluation.goal_conditioned import (
        load_goal_conditioned_checkpoint,
    )

    if calibration is None and calibration_path is not None:
        calibration = IsotonicCalibration.load(calibration_path)
    return make_critic_affordance_scorer(
        load_goal_conditioned_checkpoint(checkpoint_dir),
        calibration=calibration,
        reachability_gate=reachability_gate,
    )
