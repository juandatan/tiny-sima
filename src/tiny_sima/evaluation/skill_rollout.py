"""Shared episode accounting for scripted, random, and learned skill rollouts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from tiny_sima.envs.skills import SkillStep


class SkillEpisodeTracker(NamedTuple):
    """Completion state and metrics accumulated during a skill rollout."""

    finished: jax.Array
    succeeded: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    timed_out: jax.Array
    precondition_lost: jax.Array
    environment_terminated: jax.Array


class SkillRolloutOutput(NamedTuple):
    """Final per-episode metrics shared by all skill evaluators."""

    succeeded: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    timed_out: jax.Array
    precondition_lost: jax.Array
    environment_terminated: jax.Array


def initialize_skill_tracker(num_episodes: int) -> SkillEpisodeTracker:
    """Create empty metrics for a batch of skill episodes."""

    return SkillEpisodeTracker(
        finished=jnp.zeros((num_episodes,), dtype=jnp.bool_),
        succeeded=jnp.zeros((num_episodes,), dtype=jnp.bool_),
        episode_return=jnp.zeros((num_episodes,), dtype=jnp.float32),
        episode_length=jnp.zeros((num_episodes,), dtype=jnp.int32),
        timed_out=jnp.zeros((num_episodes,), dtype=jnp.bool_),
        precondition_lost=jnp.zeros((num_episodes,), dtype=jnp.bool_),
        environment_terminated=jnp.zeros(
            (num_episodes,),
            dtype=jnp.bool_,
        ),
    )


def update_skill_tracker(
    tracker: SkillEpisodeTracker,
    step: SkillStep,
) -> SkillEpisodeTracker:
    """Record one environment step while ignoring already-finished episodes."""

    active = ~tracker.finished
    succeeded = active & step.info["skill_success"]
    timed_out = active & step.info["skill_timeout"]
    environment_terminated = (
        active & step.done & ~step.info["skill_success"] & ~step.info["skill_timeout"]
    )
    return SkillEpisodeTracker(
        finished=tracker.finished | step.done,
        succeeded=tracker.succeeded | succeeded,
        episode_return=tracker.episode_return + step.reward * active,
        episode_length=tracker.episode_length + active.astype(jnp.int32),
        timed_out=tracker.timed_out | timed_out,
        precondition_lost=(
            tracker.precondition_lost
            | (active & ~step.info["skill_precondition_after"])
        ),
        environment_terminated=(
            tracker.environment_terminated | environment_terminated
        ),
    )


def finish_skill_tracker(
    tracker: SkillEpisodeTracker,
) -> SkillRolloutOutput:
    """Convert mutable rollout accounting into its public result."""

    return SkillRolloutOutput(
        succeeded=tracker.succeeded,
        episode_return=tracker.episode_return,
        episode_length=tracker.episode_length,
        timed_out=tracker.timed_out,
        precondition_lost=tracker.precondition_lost,
        environment_terminated=tracker.environment_terminated,
    )
