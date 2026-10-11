"""Lightweight GPU utilization and memory monitoring for training runs."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NvidiaSmiSample:
    """One utilization sample reported by ``nvidia-smi``."""

    elapsed_seconds: float
    gpu_index: int
    gpu_name: str
    memory_total_mib: float
    memory_used_mib: float
    gpu_utilization_percent: float
    memory_utilization_percent: float
    power_draw_watts: float | None


def _parse_optional_float(value: str) -> float | None:
    value = value.strip()
    if value in {"", "N/A", "[N/A]"}:
        return None
    return float(value)


def parse_nvidia_smi_output(
    output: str,
    *,
    elapsed_seconds: float,
) -> list[NvidiaSmiSample]:
    """Parse the stable CSV format requested by :class:`NvidiaSmiMonitor`."""

    samples = []
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        if len(row) != 7:
            raise ValueError(f"Expected 7 nvidia-smi fields, received {len(row)}.")
        power_draw = _parse_optional_float(row[6])
        samples.append(
            NvidiaSmiSample(
                elapsed_seconds=elapsed_seconds,
                gpu_index=int(row[0].strip()),
                gpu_name=row[1].strip(),
                memory_total_mib=float(row[2]),
                memory_used_mib=float(row[3]),
                gpu_utilization_percent=float(row[4]),
                memory_utilization_percent=float(row[5]),
                power_draw_watts=power_draw,
            )
        )
    return samples


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_samples(samples: list[NvidiaSmiSample]) -> dict[str, Any]:
    """Summarize samples separately for every visible GPU."""

    gpu_indices = sorted({sample.gpu_index for sample in samples})
    summaries = []
    for gpu_index in gpu_indices:
        gpu_samples = [
            sample for sample in samples if sample.gpu_index == gpu_index
        ]
        memory_used = [sample.memory_used_mib for sample in gpu_samples]
        gpu_utilization = [
            sample.gpu_utilization_percent for sample in gpu_samples
        ]
        memory_utilization = [
            sample.memory_utilization_percent for sample in gpu_samples
        ]
        power_draw = [
            sample.power_draw_watts
            for sample in gpu_samples
            if sample.power_draw_watts is not None
        ]
        memory_total = gpu_samples[0].memory_total_mib
        baseline_memory = memory_used[0]
        peak_memory = max(memory_used)
        summaries.append(
            {
                "gpu_index": gpu_index,
                "gpu_name": gpu_samples[0].gpu_name,
                "sample_count": len(gpu_samples),
                "memory_total_mib": memory_total,
                "baseline_memory_used_mib": baseline_memory,
                "peak_memory_used_mib": peak_memory,
                "peak_memory_delta_mib": peak_memory - baseline_memory,
                "peak_memory_fraction": peak_memory / memory_total,
                "mean_gpu_utilization_percent": _mean(gpu_utilization),
                "peak_gpu_utilization_percent": max(gpu_utilization),
                "mean_memory_utilization_percent": _mean(memory_utilization),
                "peak_memory_utilization_percent": max(memory_utilization),
                "mean_power_draw_watts": _mean(power_draw),
                "peak_power_draw_watts": max(power_draw) if power_draw else None,
            }
        )
    return {
        "available": bool(samples),
        "sample_count": len(samples),
        "gpus": summaries,
    }


def jax_allocator_summary(device: Any) -> dict[str, Any]:
    """Return allocator usage when the JAX backend exposes memory statistics."""

    memory_stats = getattr(device, "memory_stats", None)
    if memory_stats is None:
        return {"available": False, "reason": "Device has no memory_stats method."}

    stats = memory_stats()
    if not stats:
        return {
            "available": False,
            "reason": "The JAX backend did not expose allocator statistics.",
        }

    summary: dict[str, Any] = {"available": True}
    for name, value in stats.items():
        if isinstance(value, (bool, int, float, str)):
            summary[name] = value

    for name in (
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_limit",
        "bytes_reserved",
        "peak_bytes_reserved",
        "pool_bytes",
        "peak_pool_bytes",
        "largest_alloc_size",
        "largest_free_block_bytes",
    ):
        value = stats.get(name)
        if isinstance(value, (int, float)):
            summary[f"{name}_mib"] = value / (1024**2)

    peak = stats.get("peak_bytes_in_use")
    limit = stats.get("bytes_limit")
    if isinstance(peak, (int, float)) and isinstance(limit, (int, float)) and limit:
        summary["peak_bytes_in_use_fraction"] = peak / limit
    return summary


class NvidiaSmiMonitor:
    """Poll ``nvidia-smi`` in a daemon thread without making it a dependency."""

    _QUERY = (
        "index,name,memory.total,memory.used,utilization.gpu,"
        "utilization.memory,power.draw"
    )

    def __init__(self, *, enabled: bool = True, interval_seconds: float = 0.5):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive.")
        self.enabled = enabled
        self.interval_seconds = interval_seconds
        self.samples: list[NvidiaSmiSample] = []
        self.reason: str | None = None
        self._start_time: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start sampling if ``nvidia-smi`` is available."""

        if not self.enabled:
            self.reason = "GPU monitoring is disabled in the configuration."
            return
        if shutil.which("nvidia-smi") is None:
            self.reason = "nvidia-smi was not found."
            return
        if self._thread is not None:
            raise RuntimeError("GPU monitor has already been started.")

        self._start_time = time.perf_counter()
        self._thread = threading.Thread(
            target=self._poll,
            name="tiny-sima-gpu-monitor",
            daemon=True,
        )
        self._thread.start()

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval_seconds)

    def _sample_once(self) -> None:
        if self._start_time is None:
            return
        command = [
            "nvidia-smi",
            f"--query-gpu={self._QUERY}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=max(2.0, self.interval_seconds * 2),
            )
            samples = parse_nvidia_smi_output(
                result.stdout,
                elapsed_seconds=time.perf_counter() - self._start_time,
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.reason = f"{type(error).__name__}: {error}"
            return

        with self._lock:
            self.samples.extend(samples)

    def stop(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Stop sampling, write artifacts, and return the aggregate summary."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))

        with self._lock:
            samples = list(self.samples)
        summary = summarize_samples(samples)
        summary["interval_seconds"] = self.interval_seconds
        if self.reason is not None:
            summary["reason"] = self.reason

        if output_dir is not None:
            run_dir = Path(output_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_samples(samples, run_dir / "gpu_metrics.csv")
            with (run_dir / "gpu_summary.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(summary, file, indent=2, sort_keys=True)
                file.write("\n")
        return summary

    @staticmethod
    def _write_samples(samples: list[NvidiaSmiSample], path: Path) -> None:
        fieldnames = list(NvidiaSmiSample.__dataclass_fields__)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(sample) for sample in samples)
