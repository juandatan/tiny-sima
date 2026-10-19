"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Self, TypeVar

import yaml


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for the Phase 1 recurrent PPO baseline."""

    env_name: str = "Craftax-Classic-Symbolic-v1"
    seed: int = 0
    total_timesteps: int = 8192
    num_envs: int = 16
    num_steps: int = 32
    max_episode_steps: int = 1000
    use_optimistic_resets: bool = True
    optimistic_reset_ratio: int = 16
    learning_rate: float = 2e-4
    anneal_learning_rate: bool = True
    update_epochs: int = 2
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    hidden_size: int = 128
    log_every: int = 4
    checkpoint_every: int = 0
    monitor_gpu: bool = True
    gpu_monitor_interval_seconds: float = 0.5
    output_dir: str = "runs/ppo-debug"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size

    @property
    def envs_per_minibatch(self) -> int:
        return self.num_envs // self.num_minibatches

    def validate(self) -> Self:
        """Raise a helpful error for invalid training shapes or values."""

        if self.total_timesteps < self.batch_size:
            raise ValueError(
                "total_timesteps must be at least num_envs * num_steps "
                f"({self.batch_size})."
            )
        if self.total_timesteps % self.batch_size:
            raise ValueError(
                "total_timesteps must be divisible by num_envs * num_steps."
            )
        if self.num_envs % self.num_minibatches:
            raise ValueError("num_envs must be divisible by num_minibatches.")
        if self.use_optimistic_resets:
            if self.optimistic_reset_ratio < 1:
                raise ValueError("optimistic_reset_ratio must be positive.")
            if self.optimistic_reset_ratio > self.num_envs:
                raise ValueError("optimistic_reset_ratio cannot exceed num_envs.")
            if self.num_envs % self.optimistic_reset_ratio:
                raise ValueError(
                    "num_envs must be divisible by optimistic_reset_ratio."
                )
        if self.num_steps < 2:
            raise ValueError("num_steps must be at least 2.")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if self.update_epochs < 1:
            raise ValueError("update_epochs must be positive.")
        if self.log_every < 1:
            raise ValueError("log_every must be positive.")
        if self.gpu_monitor_interval_seconds <= 0:
            raise ValueError("gpu_monitor_interval_seconds must be positive.")
        return self

    def with_overrides(self, **overrides: Any) -> Self:
        """Return a validated copy with selected values replaced."""

        known_fields = {field.name for field in fields(self)}
        unknown = set(overrides) - known_fields
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown PPO configuration fields: {names}")
        return replace(self, **overrides).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillPPOConfig(PPOConfig):
    """Configuration for the categorical goal-conditioned executor."""

    skill_horizon: int = 8
    skill_embedding_size: int = 32
    normalize_advantages_per_skill: bool = False
    skill_scenario: str = "nearby"
    skill_min_distance: int = 2
    skill_max_distance: int = 4
    skill_distractor_count: int = 6

    def validate(self) -> Self:
        super().validate()
        if self.skill_horizon < 2:
            raise ValueError("skill_horizon must be at least 2.")
        if self.skill_embedding_size < 1:
            raise ValueError("skill_embedding_size must be positive.")
        if self.skill_scenario not in {"nearby", "local_navigation"}:
            raise ValueError(
                "skill_scenario must be 'nearby' or 'local_navigation'."
            )
        if self.skill_min_distance < 1:
            raise ValueError("skill_min_distance must be positive.")
        if self.skill_max_distance < self.skill_min_distance:
            raise ValueError(
                "skill_max_distance must be at least skill_min_distance."
            )
        if not 0 <= self.skill_distractor_count <= 32:
            raise ValueError(
                "skill_distractor_count must be between 0 and 32."
            )
        return self


@dataclass(frozen=True)
class VLMPlannerConfig:
    """Configuration for a frozen image-text planner backend."""

    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    environment_name: str = "Craftax-Classic-Symbolic-v1"
    max_new_tokens: int = 256
    dtype: str = "bfloat16"
    device_map: str = "auto"
    block_pixel_size: int = 16
    # Boundaries from independent episodes decode together, so this caps how
    # many padded sequences share one forward pass before memory becomes the
    # binding constraint rather than latency.
    max_batch_size: int = 8

    def validate(self) -> Self:
        """Raise a helpful error for unsupported inference settings."""

        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty.")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive.")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(
                "dtype must be auto, bfloat16, float16, or float32."
            )
        if self.block_pixel_size not in {7, 16, 64}:
            raise ValueError("block_pixel_size must be 7, 16, or 64.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ConfigType = TypeVar("ConfigType", bound=PPOConfig)


def _load_config(
    path: str | Path,
    config_type: type[ConfigType],
) -> ConfigType:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")

    known_fields = {field.name for field in fields(config_type)}
    unknown = set(raw) - known_fields
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown fields in {config_path}: {names}")

    return config_type(**raw).validate()


def load_config(path: str | Path) -> PPOConfig:
    """Load a baseline PPO configuration from YAML."""

    return _load_config(path, PPOConfig)


def load_skill_config(path: str | Path) -> SkillPPOConfig:
    """Load a goal-conditioned PPO configuration from YAML."""

    return _load_config(path, SkillPPOConfig)


def load_vlm_planner_config(path: str | Path) -> VLMPlannerConfig:
    """Load a frozen VLM planner configuration from YAML."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}.")

    known_fields = {field.name for field in fields(VLMPlannerConfig)}
    unknown = set(raw) - known_fields
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown fields in {config_path}: {names}")
    return VLMPlannerConfig(**raw).validate()


def save_config(
    config: PPOConfig | VLMPlannerConfig,
    path: str | Path,
) -> None:
    """Write a configuration as stable, human-readable YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False)
