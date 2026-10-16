"""Monotone recalibration of executor value estimates into probabilities.

The goal-conditioned skill environment pays one sparse reward on the step that
completes a skill and terminates immediately afterwards, so the critic learns

    V(s, skill) = E[gamma ** k]

for the number of steps ``k`` until success, and zero when the skill is never
completed. That is already monotone in success probability, but it is squashed
by discounting: at ``gamma = 0.95`` over a 32-step horizon a certain success
32 steps away is worth only 0.19. Fitting a monotone map from ``V`` to observed
success recovers a usable probability without retraining anything, and it keeps
the critic's preference for fast skills visible in the ordering.

Isotonic regression is implemented here with pool-adjacent-violators rather
than taken from scikit-learn, which is not a project dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IsotonicCalibration:
    """A non-decreasing piecewise-linear map from scores to probabilities."""

    scores: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.scores) != len(self.probabilities):
            raise ValueError("scores and probabilities must align.")
        if not self.scores:
            raise ValueError("A calibration needs at least one point.")
        if any(later < earlier for earlier, later in pairwise(self.scores)):
            raise ValueError("scores must be non-decreasing.")

    def __call__(self, score: float) -> float:
        """Map one raw value estimate to a calibrated probability."""

        return float(self.apply(np.asarray([score], dtype=np.float64))[0])

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Map an array of value estimates to calibrated probabilities."""

        interpolated = np.interp(
            np.asarray(scores, dtype=np.float64),
            np.asarray(self.scores, dtype=np.float64),
            np.asarray(self.probabilities, dtype=np.float64),
        )
        return np.clip(interpolated, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "scores": list(self.scores),
            "probabilities": list(self.probabilities),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IsotonicCalibration:
        """Restore a calibration from :meth:`to_dict` output."""

        return cls(
            scores=tuple(float(value) for value in payload["scores"]),
            probabilities=tuple(
                float(value) for value in payload["probabilities"]
            ),
        )

    def save(self, path: str | Path) -> None:
        """Write the calibration to a JSON file."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> IsotonicCalibration:
        """Read a calibration written by :meth:`save`."""

        return cls.from_dict(json.loads(Path(path).read_text()))


def _pool_adjacent_violators(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the weighted least-squares non-decreasing fit of ``values``."""

    level_values: list[float] = []
    level_weights: list[float] = []
    level_sizes: list[int] = []
    for value, weight in zip(values, weights, strict=True):
        level_values.append(float(value))
        level_weights.append(float(weight))
        level_sizes.append(1)
        # Merge backwards while the sequence decreases.
        while (
            len(level_values) > 1
            and level_values[-2] > level_values[-1]
        ):
            total_weight = level_weights[-2] + level_weights[-1]
            merged = (
                level_values[-2] * level_weights[-2]
                + level_values[-1] * level_weights[-1]
            ) / total_weight
            merged_size = level_sizes[-2] + level_sizes[-1]
            level_values[-2:] = [merged]
            level_weights[-2:] = [total_weight]
            level_sizes[-2:] = [merged_size]
    fitted = np.empty(len(values), dtype=np.float64)
    index = 0
    for value, size in zip(level_values, level_sizes, strict=True):
        fitted[index : index + size] = value
        index += size
    return fitted


def fit_isotonic_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
) -> IsotonicCalibration:
    """Fit a monotone score-to-probability map from binary outcomes."""

    score_values = np.asarray(scores, dtype=np.float64).ravel()
    label_values = np.asarray(labels, dtype=np.float64).ravel()
    if score_values.size != label_values.size:
        raise ValueError("scores and labels must have the same length.")
    if score_values.size == 0:
        raise ValueError("Cannot fit a calibration from zero samples.")

    order = np.argsort(score_values, kind="stable")
    sorted_scores = score_values[order]
    sorted_labels = label_values[order]

    # Average duplicate scores so each distinct score becomes one weighted
    # observation; isotonic regression is otherwise ill-defined on ties.
    unique_scores, inverse = np.unique(sorted_scores, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique_scores.size)
    sums = np.bincount(
        inverse,
        weights=sorted_labels,
        minlength=unique_scores.size,
    )
    means = sums / counts
    fitted = _pool_adjacent_violators(means, counts.astype(np.float64))
    return IsotonicCalibration(
        scores=tuple(float(value) for value in unique_scores),
        probabilities=tuple(float(value) for value in fitted),
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    num_bins: int = 10,
) -> float:
    """Return the bin-weighted gap between confidence and accuracy."""

    predicted = np.asarray(probabilities, dtype=np.float64).ravel()
    observed = np.asarray(labels, dtype=np.float64).ravel()
    if predicted.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    # ``digitize`` puts 0.0 in bin 0; shift so bins are [0, 1/n), ..., [.., 1].
    bins = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, num_bins - 1)
    error = 0.0
    for index in range(num_bins):
        selected = bins == index
        if not np.any(selected):
            continue
        weight = float(np.mean(selected))
        gap = abs(float(predicted[selected].mean() - observed[selected].mean()))
        error += weight * gap
    return error


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the rank-based area under the ROC curve, or NaN if degenerate."""

    score_values = np.asarray(scores, dtype=np.float64).ravel()
    label_values = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    positives = int(np.count_nonzero(label_values))
    negatives = int(label_values.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score_values, kind="stable")
    ranks = np.empty(score_values.size, dtype=np.float64)
    ranks[order] = np.arange(1, score_values.size + 1, dtype=np.float64)
    # Average ranks within tied score groups so ties contribute 0.5.
    unique_scores, inverse = np.unique(score_values, return_inverse=True)
    for index in range(unique_scores.size):
        tied = inverse == index
        if np.count_nonzero(tied) > 1:
            ranks[tied] = ranks[tied].mean()
    positive_rank_sum = float(ranks[label_values].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def calibration_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    calibration: IsotonicCalibration | None = None,
    *,
    num_bins: int = 10,
) -> dict[str, float]:
    """Summarize how well a score predicts success before and after fitting."""

    score_values = np.asarray(scores, dtype=np.float64).ravel()
    label_values = np.asarray(labels, dtype=np.float64).ravel()
    raw = np.clip(score_values, 0.0, 1.0)
    metrics = {
        "num_samples": float(score_values.size),
        "base_rate": float(label_values.mean()) if label_values.size else 0.0,
        "auc": roc_auc(score_values, label_values),
        "raw_brier": float(np.mean((raw - label_values) ** 2))
        if score_values.size
        else 0.0,
        "raw_ece": expected_calibration_error(
            raw,
            label_values,
            num_bins=num_bins,
        ),
    }
    if calibration is not None:
        calibrated = calibration.apply(score_values)
        metrics["calibrated_brier"] = float(
            np.mean((calibrated - label_values) ** 2)
        )
        metrics["calibrated_ece"] = expected_calibration_error(
            calibrated,
            label_values,
            num_bins=num_bins,
        )
    return metrics
