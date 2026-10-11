from pathlib import Path

from tiny_sima.config import PPOConfig
from tiny_sima.learning_comparison import (
    aggregate_records,
    comparison_record,
    make_comparison_config,
    write_comparison_results,
)


def test_make_comparison_config_keeps_total_steps_fixed(tmp_path: Path) -> None:
    base = PPOConfig(
        num_envs=64,
        num_steps=64,
        num_minibatches=8,
        total_timesteps=1_048_576,
    ).validate()

    config = make_comparison_config(
        base,
        num_envs=256,
        seed=2,
        total_timesteps=1_048_576,
        output_dir=tmp_path,
    )

    assert config.num_envs == 256
    assert config.seed == 2
    assert config.num_updates == 64
    assert config.checkpoint_every == 0


def _record(num_envs: int, seed: int, evaluation_return: float):
    return {
        "num_envs": num_envs,
        "training_seed": seed,
        "batch_size": num_envs * 64,
        "num_updates": 1_048_576 // (num_envs * 64),
        "training_seconds": 100 / (num_envs / 64),
        "training_steps_per_second": 10_000 * (num_envs / 64),
        "evaluation_mean_return": evaluation_return,
        "evaluation_mean_episode_length": 150 + seed,
        "jax_peak_memory_mib": 200,
    }


def test_aggregate_records_groups_training_seeds() -> None:
    records = [
        _record(64, 0, 3.0),
        _record(64, 1, 5.0),
        _record(256, 0, 2.0),
    ]

    aggregates = aggregate_records(records)

    assert aggregates[0]["num_envs"] == 64
    assert aggregates[0]["completed_seeds"] == 2
    assert aggregates[0]["mean_evaluation_return"] == 4.0
    assert aggregates[0]["evaluation_return_std_across_seeds"] == 1.0
    assert aggregates[1]["num_updates"] == 64


def test_aggregate_records_allows_missing_hardware_metrics() -> None:
    record = _record(64, 0, 3.0)
    record["jax_peak_memory_mib"] = None

    aggregate = aggregate_records([record])[0]

    assert aggregate["mean_jax_peak_memory_mib"] is None


def test_comparison_record_flattens_training_and_evaluation() -> None:
    config = PPOConfig()
    training = {
        "total_seconds": 10,
        "steps_per_second_including_compile": 1000,
        "tail_steps_per_second": 2000,
        "mean_completed_episode_return": 2.5,
        "jax_memory": {"peak_bytes_in_use_mib": 100},
    }
    evaluation = {
        "seed": 100,
        "num_episodes": 64,
        "mean_return": 3.5,
        "std_return": 1.0,
        "min_return": 0.1,
        "max_return": 7.0,
        "mean_episode_length": 170,
        "checkpoint": "checkpoint",
    }

    record = comparison_record(config, training, evaluation)

    assert record["evaluation_mean_return"] == 3.5
    assert record["tail_steps_per_second"] == 2000
    assert record["jax_peak_memory_mib"] == 100


def test_write_comparison_results_creates_partial_artifacts(
    tmp_path: Path,
) -> None:
    records = [_record(64, 0, 3.0)]

    write_comparison_results(records, tmp_path)

    assert (tmp_path / "comparison_runs.json").is_file()
    assert (tmp_path / "comparison_runs.csv").is_file()
    assert (tmp_path / "comparison_aggregate.json").is_file()
    assert (tmp_path / "comparison_aggregate.csv").is_file()
