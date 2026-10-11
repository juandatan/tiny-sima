"""Profile PPO throughput across environment batch sizes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tiny_sima.config import PPOConfig, load_config, save_config


def make_profile_config(
    base_config: PPOConfig,
    *,
    num_envs: int,
    num_updates: int,
    output_dir: str | Path,
) -> PPOConfig:
    """Create an equal-update profiling configuration."""

    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    if num_updates < 2:
        raise ValueError("num_updates must be at least 2.")

    total_timesteps = num_envs * base_config.num_steps * num_updates
    return base_config.with_overrides(
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        checkpoint_every=0,
        output_dir=str(output_dir),
    )


def _first_gpu(summary: dict[str, Any]) -> dict[str, Any]:
    gpus = summary.get("gpus", [])
    return gpus[0] if gpus else {}


def tail_steps_per_second(metrics: list[dict[str, Any]]) -> float | None:
    """Calculate throughput between the final two synchronized log records."""

    if len(metrics) < 2:
        return None
    previous, current = metrics[-2:]
    elapsed = current["elapsed_seconds"] - previous["elapsed_seconds"]
    if elapsed <= 0:
        return None
    return (current["timesteps"] - previous["timesteps"]) / elapsed


def load_metric_records(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited training metrics."""

    metrics_path = Path(path)
    if not metrics_path.is_file():
        return []
    with metrics_path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def profile_record(
    config: PPOConfig,
    summary: dict[str, Any],
    metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract the measurements needed to compare scaling candidates."""

    gpu_monitor = summary.get("gpu_monitor", {})
    overall_gpu = _first_gpu(gpu_monitor)
    steady_gpu = _first_gpu(gpu_monitor.get("steady_state", {}))
    tail_gpu = _first_gpu(gpu_monitor.get("tail", {})) or steady_gpu
    jax_memory = summary.get("jax_memory", {})
    tail_sps = summary.get("tail_steps_per_second")
    if tail_sps is None and metrics is not None:
        tail_sps = tail_steps_per_second(metrics)
    return {
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "batch_size": config.batch_size,
        "num_updates": config.num_updates,
        "timesteps": config.total_timesteps,
        "compile_seconds": summary.get("compile_and_first_update_seconds"),
        "total_seconds": summary.get("total_seconds"),
        "steps_per_second": summary.get("steps_per_second_including_compile"),
        "steady_steps_per_second": summary.get(
            "steps_per_second_after_compile"
        ),
        "tail_steps_per_second": tail_sps,
        "jax_peak_memory_mib": jax_memory.get("peak_bytes_in_use_mib"),
        "overall_mean_gpu_utilization_percent": overall_gpu.get(
            "mean_gpu_utilization_percent"
        ),
        "steady_mean_gpu_utilization_percent": steady_gpu.get(
            "mean_gpu_utilization_percent"
        ),
        "steady_peak_gpu_utilization_percent": steady_gpu.get(
            "peak_gpu_utilization_percent"
        ),
        "steady_mean_power_draw_watts": steady_gpu.get(
            "mean_power_draw_watts"
        ),
        "tail_mean_gpu_utilization_percent": tail_gpu.get(
            "mean_gpu_utilization_percent"
        ),
        "tail_mean_power_draw_watts": tail_gpu.get("mean_power_draw_watts"),
        "checkpoint": summary.get("checkpoint"),
    }


def write_profile_results(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Write aggregate JSON and CSV files after each completed candidate."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "scaling_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(records, file, indent=2, sort_keys=True)
        file.write("\n")

    if not records:
        return
    with (output_path / "scaling_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_scaling_sweep(
    base_config: PPOConfig,
    *,
    environment_counts: Iterable[int],
    num_updates: int,
    output_dir: str | Path,
    reuse_existing: bool = False,
) -> list[dict[str, Any]]:
    """Run each candidate in an isolated Python process."""

    output_path = Path(output_dir).resolve()
    config_dir = output_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for num_envs in environment_counts:
        run_dir = output_path / f"envs-{num_envs}"
        config = make_profile_config(
            base_config,
            num_envs=num_envs,
            num_updates=num_updates,
            output_dir=run_dir,
        )
        summary_path = run_dir / "summary.json"
        config_path = config_dir / f"envs-{num_envs}.yaml"
        save_config(config, config_path)

        if not (reuse_existing and summary_path.is_file()):
            print(
                f"\nProfiling num_envs={num_envs}, "
                f"batch_size={config.batch_size}, "
                f"timesteps={config.total_timesteps}",
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tiny_sima.executor.ppo",
                    "--config",
                    str(config_path),
                ],
                check=True,
            )

        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        metrics = load_metric_records(run_dir / "metrics.jsonl")
        records.append(profile_record(config, summary, metrics))
        write_profile_results(records, output_path)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ppo_1m.yaml",
        help="Base PPO configuration.",
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096],
        help="Environment counts to profile.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=32,
        help="Number of PPO updates per candidate.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/executor-scaling",
        help="Directory for candidate runs and aggregate results.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse candidates that already have a summary.json file.",
    )
    args = parser.parse_args()

    records = run_scaling_sweep(
        load_config(args.config),
        environment_counts=args.num_envs,
        num_updates=args.updates,
        output_dir=args.output_dir,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
