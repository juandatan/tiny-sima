"""Compare PPO learning quality across environment batch sizes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tiny_sima.config import PPOConfig, load_config, save_config


def make_comparison_config(
    base_config: PPOConfig,
    *,
    num_envs: int,
    seed: int,
    total_timesteps: int,
    output_dir: str | Path,
) -> PPOConfig:
    """Create one fixed-timestep learning-comparison configuration."""

    return base_config.with_overrides(
        num_envs=num_envs,
        seed=seed,
        total_timesteps=total_timesteps,
        checkpoint_every=0,
        output_dir=str(output_dir),
    )


def comparison_record(
    config: PPOConfig,
    training: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Combine training and evaluation measurements into one flat record."""

    jax_memory = training.get("jax_memory", {})
    return {
        "num_envs": config.num_envs,
        "training_seed": config.seed,
        "evaluation_seed": evaluation["seed"],
        "batch_size": config.batch_size,
        "num_updates": config.num_updates,
        "timesteps": config.total_timesteps,
        "training_seconds": training["total_seconds"],
        "training_steps_per_second": training[
            "steps_per_second_including_compile"
        ],
        "tail_steps_per_second": training.get("tail_steps_per_second"),
        "jax_peak_memory_mib": jax_memory.get("peak_bytes_in_use_mib"),
        "training_mean_episode_return": training[
            "mean_completed_episode_return"
        ],
        "evaluation_episodes": evaluation["num_episodes"],
        "evaluation_mean_return": evaluation["mean_return"],
        "evaluation_return_std": evaluation["std_return"],
        "evaluation_min_return": evaluation["min_return"],
        "evaluation_max_return": evaluation["max_return"],
        "evaluation_mean_episode_length": evaluation["mean_episode_length"],
        "checkpoint": evaluation["checkpoint"],
    }


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(record[key]) for record in records)


def _optional_mean(
    records: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        float(record[key]) for record in records if record.get(key) is not None
    ]
    return statistics.fmean(values) if values else None


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate completed training seeds for every environment count."""

    aggregates = []
    for num_envs in sorted({record["num_envs"] for record in records}):
        group = [record for record in records if record["num_envs"] == num_envs]
        returns = [float(record["evaluation_mean_return"]) for record in group]
        aggregates.append(
            {
                "num_envs": num_envs,
                "completed_seeds": len(group),
                "batch_size": group[0]["batch_size"],
                "num_updates": group[0]["num_updates"],
                "mean_training_seconds": _mean(group, "training_seconds"),
                "mean_training_steps_per_second": _mean(
                    group, "training_steps_per_second"
                ),
                "mean_evaluation_return": statistics.fmean(returns),
                "evaluation_return_std_across_seeds": (
                    statistics.pstdev(returns) if len(returns) > 1 else 0.0
                ),
                "mean_evaluation_episode_length": _mean(
                    group, "evaluation_mean_episode_length"
                ),
                "mean_jax_peak_memory_mib": _optional_mean(
                    group, "jax_peak_memory_mib"
                ),
            }
        )
    return aggregates


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_comparison_results(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Persist partial per-run and aggregate results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_records(records)

    for filename, values in (
        ("comparison_runs.json", records),
        ("comparison_aggregate.json", aggregates),
    ):
        with (output_path / filename).open("w", encoding="utf-8") as file:
            json.dump(values, file, indent=2, sort_keys=True)
            file.write("\n")

    _write_csv(records, output_path / "comparison_runs.csv")
    _write_csv(aggregates, output_path / "comparison_aggregate.csv")


def _evaluate_checkpoint(
    checkpoint: Path,
    *,
    num_episodes: int,
    num_envs: int,
    seed: int,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tiny_sima.evaluation.policy import main; main()",
            str(checkpoint),
            "--num-episodes",
            str(num_episodes),
            "--num-envs",
            str(num_envs),
            "--seed",
            str(seed),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    print(result.stdout, end="", flush=True)
    return json.loads(result.stdout)


def run_learning_comparison(
    base_config: PPOConfig,
    *,
    environment_counts: Iterable[int],
    training_seeds: Iterable[int],
    total_timesteps: int,
    evaluation_episodes: int,
    evaluation_num_envs: int,
    evaluation_seed: int,
    output_dir: str | Path,
    reuse_existing: bool = False,
) -> list[dict[str, Any]]:
    """Train and evaluate every environment-count and seed combination."""

    output_path = Path(output_dir).resolve()
    config_dir = output_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for num_envs in environment_counts:
        for training_seed in training_seeds:
            run_dir = output_path / f"envs-{num_envs}" / f"seed-{training_seed}"
            config = make_comparison_config(
                base_config,
                num_envs=num_envs,
                seed=training_seed,
                total_timesteps=total_timesteps,
                output_dir=run_dir,
            )
            config_path = config_dir / f"envs-{num_envs}-seed-{training_seed}.yaml"
            training_path = run_dir / "summary.json"
            evaluation_path = run_dir / "evaluation.json"
            save_config(config, config_path)

            if not (reuse_existing and training_path.is_file()):
                print(
                    f"\nTraining num_envs={num_envs}, seed={training_seed}, "
                    f"updates={config.num_updates}",
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

            with training_path.open("r", encoding="utf-8") as file:
                training = json.load(file)

            if reuse_existing and evaluation_path.is_file():
                with evaluation_path.open("r", encoding="utf-8") as file:
                    evaluation = json.load(file)
            else:
                print(
                    f"Evaluating num_envs={num_envs}, seed={training_seed}",
                    flush=True,
                )
                evaluation = _evaluate_checkpoint(
                    run_dir / "checkpoints" / "final",
                    num_episodes=evaluation_episodes,
                    num_envs=evaluation_num_envs,
                    seed=evaluation_seed,
                )
                with evaluation_path.open("w", encoding="utf-8") as file:
                    json.dump(evaluation, file, indent=2, sort_keys=True)
                    file.write("\n")

            records.append(comparison_record(config, training, evaluation))
            write_comparison_results(records, output_path)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_1m.yaml")
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=[64, 256, 512],
    )
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument("--total-timesteps", type=int, default=1_048_576)
    parser.add_argument("--evaluation-episodes", type=int, default=64)
    parser.add_argument("--evaluation-num-envs", type=int, default=8)
    parser.add_argument("--evaluation-seed", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="runs/learning-scaling-comparison",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    records = run_learning_comparison(
        load_config(args.config),
        environment_counts=args.num_envs,
        training_seeds=args.training_seeds,
        total_timesteps=args.total_timesteps,
        evaluation_episodes=args.evaluation_episodes,
        evaluation_num_envs=args.evaluation_num_envs,
        evaluation_seed=args.evaluation_seed,
        output_dir=args.output_dir,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(aggregate_records(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
