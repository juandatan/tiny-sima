"""Small, dependency-light checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flax import serialization

from tiny_sima.config import PPOConfig, save_config


def save_checkpoint(
    directory: str | Path,
    params: Any,
    config: PPOConfig,
    *,
    timesteps: int,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save policy parameters and enough metadata to evaluate them."""

    checkpoint_dir = Path(directory)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    params_path = checkpoint_dir / "params.msgpack"
    params_path.write_bytes(serialization.to_bytes(params))
    save_config(config, checkpoint_dir / "config.yaml")

    metadata = {
        "timesteps": int(timesteps),
        "metrics": metrics or {},
    }
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")
    return checkpoint_dir


def load_checkpoint(
    directory: str | Path,
    target_params: Any,
) -> tuple[Any, dict[str, Any]]:
    """Load policy parameters into an initialized parameter tree."""

    checkpoint_dir = Path(directory)
    params_path = checkpoint_dir / "params.msgpack"
    metadata_path = checkpoint_dir / "metadata.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing checkpoint parameters: {params_path}")

    params = serialization.from_bytes(target_params, params_path.read_bytes())
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    else:
        metadata = {}
    return params, metadata
