"""Unmodified Craftax-Classic worlds for the open-ended achievement task.

The chain worlds are carved around the player so a short recipe is always
solvable. The open-ended task asks for as many achievements as the agent can
unlock, so it keeps the generated world as it is: its trees, water, caves,
day and night, and, unless they are switched off, its creatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

from tiny_sima.envs.chains import SkillChainEnvironment, SkillChainReset, _clear_mobs
from tiny_sima.envs.craftax import CraftaxBundle, make_environment

# Long enough for the open-ended step budget plus one skill's overrun.
DEFAULT_OPEN_ENDED_MAX_STEPS = 10_000


@dataclass(frozen=True)
class OpenEndedEnvironment(SkillChainEnvironment):
    """A persistent Craftax world reset without the chain scenario.

    It is a :class:`SkillChainEnvironment`, so every executor and the chain
    loop accept it. It places no blocks of its own, so ``tree_count`` and
    ``distractor_count`` are zero. With ``mobs`` false the bundle's spawn
    chances are zero and the creatures generated at reset are removed, so
    the world stays empty of them.
    """

    distractor_count: int = 0
    tree_count: int = 0
    mobs: bool = True

    def reset(self, rng: jax.Array) -> SkillChainReset:
        _, state = self.bundle.env.reset(rng, self.bundle.params)
        if not self.mobs:
            state = _clear_mobs(state)
        return SkillChainReset(
            observation=self.bundle.env.get_obs(state),
            state=state,
        )


def make_open_ended_environment(
    *,
    mobs: bool = True,
    max_episode_steps: int = DEFAULT_OPEN_ENDED_MAX_STEPS,
) -> OpenEndedEnvironment:
    """Create the open-ended world, with or without creatures.

    ``mobs`` keeps the default spawn parameters. Without it cows, zombies and
    skeletons neither start in the world nor spawn later, which also removes
    eat_cow and the combat achievements from what the agent can unlock.
    """

    bundle = make_environment(
        "Craftax-Classic-Symbolic-v1",
        auto_reset=False,
        max_episode_steps=max_episode_steps,
    )
    params = bundle.params
    if not mobs:
        params = params.replace(
            spawn_cow_chance=0.0,
            spawn_zombie_base_chance=0.0,
            spawn_zombie_night_chance=0.0,
            spawn_skeleton_chance=0.0,
        )
    return OpenEndedEnvironment(
        bundle=CraftaxBundle(env=bundle.env, params=params, spec=bundle.spec),
        mobs=mobs,
    )
