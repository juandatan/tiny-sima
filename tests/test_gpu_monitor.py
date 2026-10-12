from pathlib import Path

import pytest

from tiny_sima.evaluation.gpu_monitor import (
    NvidiaSmiMonitor,
    NvidiaSmiSample,
    jax_allocator_summary,
    parse_nvidia_smi_output,
    summarize_samples,
)


def test_parse_and_summarize_nvidia_smi_samples() -> None:
    first = parse_nvidia_smi_output(
        "0, NVIDIA L40, 46068, 1200, 25, 10, 85.5\n",
        elapsed_seconds=0.5,
    )
    second = parse_nvidia_smi_output(
        "0, NVIDIA L40, 46068, 8200, 90, 45, 240.0\n",
        elapsed_seconds=1.0,
    )

    summary = summarize_samples(first + second)
    gpu = summary["gpus"][0]

    assert summary["available"]
    assert gpu["sample_count"] == 2
    assert gpu["peak_memory_used_mib"] == 8200
    assert gpu["peak_memory_delta_mib"] == 7000
    assert gpu["mean_gpu_utilization_percent"] == pytest.approx(57.5)
    assert gpu["peak_power_draw_watts"] == 240.0


def test_monitor_writes_empty_artifacts_when_disabled(tmp_path: Path) -> None:
    monitor = NvidiaSmiMonitor(enabled=False)
    monitor.start()
    summary = monitor.stop(tmp_path)

    assert not summary["available"]
    assert "disabled" in summary["reason"]
    assert (tmp_path / "gpu_metrics.csv").is_file()
    assert (tmp_path / "gpu_summary.json").is_file()


def test_monitor_summarizes_steady_state_window(tmp_path: Path) -> None:
    monitor = NvidiaSmiMonitor(enabled=False)
    monitor.samples = [
        NvidiaSmiSample(0.5, 0, "GPU", 100, 10, 20, 5, 50),
        NvidiaSmiSample(1.5, 0, "GPU", 100, 20, 80, 10, 90),
    ]

    summary = monitor.stop(
        tmp_path,
        steady_state_start_seconds=1.0,
        tail_start_seconds=1.25,
    )

    steady_gpu = summary["steady_state"]["gpus"][0]
    tail_gpu = summary["tail"]["gpus"][0]
    assert steady_gpu["sample_count"] == 1
    assert steady_gpu["mean_gpu_utilization_percent"] == 80
    assert tail_gpu["sample_count"] == 1


def test_jax_allocator_summary_converts_byte_fields() -> None:
    class FakeDevice:
        def memory_stats(self) -> dict[str, int]:
            return {
                "bytes_in_use": 10 * 1024**2,
                "peak_bytes_in_use": 20 * 1024**2,
                "bytes_limit": 100 * 1024**2,
            }

    summary = jax_allocator_summary(FakeDevice())

    assert summary["available"]
    assert summary["peak_bytes_in_use_mib"] == 20
    assert summary["peak_bytes_in_use_fraction"] == pytest.approx(0.2)


def test_summarize_samples_handles_missing_power_reading() -> None:
    sample = NvidiaSmiSample(
        elapsed_seconds=0.5,
        gpu_index=0,
        gpu_name="GPU",
        memory_total_mib=100,
        memory_used_mib=10,
        gpu_utilization_percent=20,
        memory_utilization_percent=5,
        power_draw_watts=None,
    )

    gpu = summarize_samples([sample])["gpus"][0]

    assert gpu["mean_power_draw_watts"] is None
    assert gpu["peak_power_draw_watts"] is None
