from pathlib import Path

import pytest

from tiny_sima.config import PPOConfig
from tiny_sima.profiling import (
    load_metric_records,
    make_profile_config,
    profile_record,
    tail_steps_per_second,
    write_profile_results,
)


def test_make_profile_config_uses_equal_update_count(tmp_path: Path) -> None:
    base = PPOConfig(
        num_envs=64,
        num_steps=64,
        num_minibatches=8,
        total_timesteps=4096,
    ).validate()

    config = make_profile_config(
        base,
        num_envs=128,
        num_updates=32,
        output_dir=tmp_path / "envs-128",
    )

    assert config.num_envs == 128
    assert config.num_updates == 32
    assert config.total_timesteps == 262_144
    assert config.checkpoint_every == 0


def test_make_profile_config_rejects_single_update(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        make_profile_config(
            PPOConfig(),
            num_envs=64,
            num_updates=1,
            output_dir=tmp_path,
        )


def test_profile_record_prefers_steady_gpu_window() -> None:
    config = PPOConfig()
    summary = {
        "steps_per_second_after_compile": 12_000,
        "jax_memory": {"peak_bytes_in_use_mib": 172},
        "gpu_monitor": {
            "gpus": [{"mean_gpu_utilization_percent": 40}],
            "steady_state": {
                "gpus": [
                    {
                        "mean_gpu_utilization_percent": 75,
                        "peak_gpu_utilization_percent": 100,
                        "mean_power_draw_watts": 150,
                    }
                ]
            },
            "tail": {
                "gpus": [
                    {
                        "mean_gpu_utilization_percent": 85,
                        "mean_power_draw_watts": 175,
                    }
                ]
            },
        },
    }

    metrics = [
        {"timesteps": 100, "elapsed_seconds": 10},
        {"timesteps": 300, "elapsed_seconds": 12},
    ]
    record = profile_record(config, summary, metrics)

    assert record["steady_steps_per_second"] == 12_000
    assert record["jax_peak_memory_mib"] == 172
    assert record["overall_mean_gpu_utilization_percent"] == 40
    assert record["steady_mean_gpu_utilization_percent"] == 75
    assert record["tail_steps_per_second"] == 100
    assert record["tail_mean_gpu_utilization_percent"] == 85


def test_tail_steps_per_second_and_metric_loading(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"timesteps": 100, "elapsed_seconds": 5}\n'
        '{"timesteps": 500, "elapsed_seconds": 7}\n',
        encoding="utf-8",
    )

    metrics = load_metric_records(path)

    assert tail_steps_per_second(metrics) == 200


def test_write_profile_results_creates_json_and_csv(tmp_path: Path) -> None:
    records = [{"num_envs": 64, "steady_steps_per_second": 12_000}]

    write_profile_results(records, tmp_path)

    assert (tmp_path / "scaling_summary.json").is_file()
    assert (tmp_path / "scaling_summary.csv").is_file()
