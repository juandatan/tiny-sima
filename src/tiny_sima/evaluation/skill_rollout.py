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


class SkillRolloutOutput(NamedTuple):
    """Final per-episode metrics shared by all skill evaluators."""

    succeeded: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array


def initialize_skill_tracker(num_episodes: int) -> SkillEpisodeTracker:
    """Create empty metrics for a batch of skill episodes."""

    return SkillEpisodeTracker(
        finished=jnp.zeros((num_episodes,), dtype=jnp.bool_),
        succeeded=jnp.zeros((num_episodes,), dtype=jnp.bool_),
        episode_return=jnp.zeros((num_episodes,), dtype=jnp.float32),
        episode_length=jnp.zeros((num_episodes,), dtype=jnp.int32),
    )


def update_skill_tracker(
    tracker: SkillEpisodeTracker,
    step: SkillStep,
) -> SkillEpisodeTracker:
    """Record one environment step while ignoring already-finished episodes."""

    active = ~tracker.finished
    return SkillEpisodeTracker(
        finished=tracker.finished | step.done,
        succeeded=tracker.succeeded | (active & step.info["skill_success"]),
        episode_return=tracker.episode_return + step.reward * active,
        episode_length=tracker.episode_length + active.astype(jnp.int32),
    )


def finish_skill_tracker(
    tracker: SkillEpisodeTracker,
) -> SkillRolloutOutput:
    """Convert mutable rollout accounting into its public result."""

    return SkillRolloutOutput(
        succeeded=tracker.succeeded,
        episode_return=tracker.episode_return,
        episode_length=tracker.episode_length,
    )
