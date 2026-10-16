import json
from pathlib import Path

import numpy as np
import pytest

from tiny_sima.planner import (
    IsotonicCalibration,
    calibration_metrics,
    expected_calibration_error,
    fit_isotonic_calibration,
    roc_auc,
)


def test_isotonic_fit_is_monotone_and_pools_violators() -> None:
    scores = np.linspace(0.0, 0.9, 10)
    labels = np.asarray([0, 0, 0, 1, 0, 1, 1, 1, 1, 1], dtype=np.float64)

    calibration = fit_isotonic_calibration(scores, labels)

    assert list(calibration.probabilities) == sorted(calibration.probabilities)
    # The single out-of-order pair at 0.3 and 0.4 is averaged, not reordered.
    assert calibration(0.3) == pytest.approx(0.5)
    assert calibration(0.4) == pytest.approx(0.5)
    assert calibration(0.0) == pytest.approx(0.0)
    assert calibration(0.9) == pytest.approx(1.0)


def test_isotonic_fit_recovers_a_perfectly_separable_split() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    calibration = fit_isotonic_calibration(scores, labels)

    assert calibration.apply(scores).tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert roc_auc(scores, labels) == pytest.approx(1.0)


def test_isotonic_fit_averages_duplicate_scores() -> None:
    # A discounted critic routinely returns the same value in different states.
    scores = np.asarray([0.2, 0.2, 0.2, 0.2, 0.8])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0])

    calibration = fit_isotonic_calibration(scores, labels)

    assert calibration(0.2) == pytest.approx(0.5)
    assert calibration(0.8) == pytest.approx(1.0)


def test_calibration_interpolates_and_clips_outside_the_fit_range() -> None:
    calibration = IsotonicCalibration(
        scores=(0.0, 1.0),
        probabilities=(0.0, 1.0),
    )

    assert calibration(0.5) == pytest.approx(0.5)
    assert calibration(-3.0) == pytest.approx(0.0)
    assert calibration(7.0) == pytest.approx(1.0)


def test_calibration_improves_brier_and_calibration_error() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.0, 1.0, size=400)
    labels = (rng.uniform(size=400) < probabilities).astype(np.float64)
    # Mimic discounting: the raw score is a squashed version of the truth.
    scores = 0.2 * probabilities

    calibration = fit_isotonic_calibration(scores, labels)
    metrics = calibration_metrics(scores, labels, calibration)

    assert metrics["calibrated_brier"] < metrics["raw_brier"]
    assert metrics["calibrated_ece"] < metrics["raw_ece"]
    # Recalibration never reorders skills: it only pools adjacent scores, so a
    # lower raw value can never come out above a higher one.
    order = np.argsort(scores)
    calibrated = calibration.apply(scores)[order]
    assert np.all(np.diff(calibrated) >= 0.0)


def test_expected_calibration_error_is_zero_for_a_perfect_predictor() -> None:
    probabilities = np.asarray([0.0] * 50 + [1.0] * 50)
    labels = np.asarray([0.0] * 50 + [1.0] * 50)

    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)


def test_roc_auc_is_undefined_without_both_classes() -> None:
    assert np.isnan(roc_auc(np.asarray([0.1, 0.2]), np.asarray([1.0, 1.0])))
    assert np.isnan(roc_auc(np.asarray([0.1, 0.2]), np.asarray([0.0, 0.0])))


def test_roc_auc_scores_ties_at_one_half() -> None:
    scores = np.asarray([0.5, 0.5])
    labels = np.asarray([0.0, 1.0])

    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_calibration_round_trips_through_json(tmp_path: Path) -> None:
    calibration = fit_isotonic_calibration(
        np.asarray([0.0, 0.25, 0.5, 0.75]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
    )
    destination = tmp_path / "nested" / "calibration.json"

    calibration.save(destination)
    restored = IsotonicCalibration.load(destination)

    assert restored == calibration
    assert set(json.loads(destination.read_text())) == {
        "scores",
        "probabilities",
    }


def test_calibration_rejects_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        IsotonicCalibration(scores=(0.0, 1.0), probabilities=(0.5,))
    with pytest.raises(ValueError, match="at least one"):
        IsotonicCalibration(scores=(), probabilities=())
    with pytest.raises(ValueError, match="non-decreasing"):
        IsotonicCalibration(scores=(1.0, 0.0), probabilities=(0.0, 1.0))
    with pytest.raises(ValueError, match="same length"):
        fit_isotonic_calibration(np.asarray([0.0, 1.0]), np.asarray([1.0]))
    with pytest.raises(ValueError, match="zero samples"):
        fit_isotonic_calibration(np.asarray([]), np.asarray([]))
