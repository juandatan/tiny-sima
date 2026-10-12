"""Shared host-side orchestration for Tiny-SIMA PPO training runs."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import jax

from tiny_sima.config import PPOConfig
from tiny_sima.evaluation.gpu_monitor import NvidiaSmiMonitor
from tiny_sima.executor.checkpoints import save_checkpoint
from tiny_sima.executor.ppo_common import UpdateMetrics, host_metrics

RunnerType = TypeVar("RunnerType")
ProgressFormatter = Callable[
    [int, int, dict[str, float], dict[str, Any]],
    str,
]


@dataclass(frozen=True)
class TrainingLoopResult(Generic[RunnerType]):
    """Runner state, metrics, and timing produced by the shared host loop."""

    runner: RunnerType
    latest_metrics: dict[str, float]
    total_seconds: float
    compile_and_first_update_seconds: float
    post_compile_seconds: float
    steps_per_second_after_compile: float | None
    tail_steps_per_second: float | None
    steady_state_start_seconds: float | None
    tail_gpu_start_seconds: float | None


def run_training_loop(
    *,
    config: PPOConfig,
    run_dir: Path,
    initial_runner: RunnerType,
    update: Callable[
        [RunnerType],
        tuple[RunnerType, UpdateMetrics],
    ],
    monitor: NvidiaSmiMonitor,
    get_params: Callable[[RunnerType], Any],
    format_progress: ProgressFormatter,
) -> TrainingLoopResult[RunnerType]:
    """Compile and execute PPO updates with shared logging/checkpoint behavior."""

    metrics_path = run_dir / "metrics.jsonl"
    start = time.perf_counter()
    compile_start = time.perf_counter()
    runner, device_metrics = update(initial_runner)
    jax.block_until_ready(device_metrics)
    compile_seconds = time.perf_counter() - compile_start
    steady_state_start_seconds = monitor.elapsed_seconds()
    post_compile_start = time.perf_counter()
    latest_metrics = host_metrics(device_metrics)
    previous_log_timesteps: int | None = None
    previous_log_elapsed: float | None = None
    previous_log_monitor_seconds: float | None = None
    tail_steps_per_second: float | None = None
    tail_gpu_start_seconds: float | None = None

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for update_index in range(config.num_updates):
            if update_index > 0:
                runner, device_metrics = update(runner)

            should_log = (
                update_index == 0
                or (update_index + 1) % config.log_every == 0
                or update_index + 1 == config.num_updates
            )
            if should_log:
                jax.block_until_ready(device_metrics)
                latest_metrics = host_metrics(device_metrics)
                timesteps = (update_index + 1) * config.batch_size
                elapsed = time.perf_counter() - start
                post_compile_elapsed = time.perf_counter() - post_compile_start
                post_compile_timesteps = update_index * config.batch_size
                interval_steps_per_second = None
                if (
                    previous_log_timesteps is not None
                    and previous_log_elapsed is not None
                ):
                    interval_steps_per_second = (timesteps - previous_log_timesteps) / (
                        elapsed - previous_log_elapsed
                    )
                    tail_steps_per_second = interval_steps_per_second

                monitor_elapsed = monitor.elapsed_seconds()
                if previous_log_monitor_seconds is not None:
                    tail_gpu_start_seconds = previous_log_monitor_seconds
                record = {
                    "update": update_index + 1,
                    "total_updates": config.num_updates,
                    "timesteps": timesteps,
                    "elapsed_seconds": elapsed,
                    "steps_per_second_including_compile": timesteps / elapsed,
                    "steps_per_second_after_compile": (
                        post_compile_timesteps / post_compile_elapsed
                        if post_compile_timesteps
                        else None
                    ),
                    "interval_steps_per_second": interval_steps_per_second,
                    **latest_metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(
                    format_progress(
                        update_index + 1,
                        timesteps,
                        latest_metrics,
                        record,
                    )
                )
                previous_log_timesteps = timesteps
                previous_log_elapsed = elapsed
                previous_log_monitor_seconds = monitor_elapsed

            if (
                config.checkpoint_every > 0
                and (update_index + 1) % config.checkpoint_every == 0
            ):
                timesteps = (update_index + 1) * config.batch_size
                save_checkpoint(
                    run_dir / "checkpoints" / f"step-{timesteps}",
                    get_params(runner),
                    config,
                    timesteps=timesteps,
                    metrics=latest_metrics,
                )

    jax.block_until_ready(get_params(runner))
    total_seconds = time.perf_counter() - start
    post_compile_seconds = time.perf_counter() - post_compile_start
    post_compile_timesteps = max(
        0,
        config.total_timesteps - config.batch_size,
    )
    return TrainingLoopResult(
        runner=runner,
        latest_metrics=latest_metrics,
        total_seconds=total_seconds,
        compile_and_first_update_seconds=compile_seconds,
        post_compile_seconds=post_compile_seconds,
        steps_per_second_after_compile=(
            post_compile_timesteps / post_compile_seconds
            if post_compile_timesteps
            else None
        ),
        tail_steps_per_second=tail_steps_per_second,
        steady_state_start_seconds=steady_state_start_seconds,
        tail_gpu_start_seconds=tail_gpu_start_seconds,
    )


def save_final_run(
    *,
    run_dir: Path,
    params: Any,
    config: PPOConfig,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Save the final checkpoint and JSON summary for a training run."""

    checkpoint = save_checkpoint(
        run_dir / "checkpoints" / "final",
        params,
        config,
        timesteps=config.total_timesteps,
        metrics=summary,
    )
    completed_summary = {**summary, "checkpoint": str(checkpoint)}
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(completed_summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return completed_summary
