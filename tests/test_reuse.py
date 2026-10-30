import json
import subprocess
from pathlib import Path

from tiny_sima.evaluation import (
    fingerprint,
    git_commit,
    provenance_mismatch,
    run_provenance,
    with_provenance,
)
from tiny_sima.goals import SkillId
from tiny_sima.planner import CompetencePrior, fit_competence_prior


def _provenance(**overrides):
    fields = {
        "seed": 51,
        "num_episodes": 64,
        "executor": Path("runs/selected/checkpoints/final"),
        "vlm": {"model_id": "Qwen/Qwen3-VL-2B-Instruct", "do_sample": False},
    }
    return run_provenance(**{**fields, **overrides})


def test_a_saved_result_with_the_same_provenance_is_reused() -> None:
    saved = with_provenance({"success_rate": 0.5}, _provenance(), "abc123")
    # A round trip through JSON, as the notebooks save and reload it.
    saved = json.loads(json.dumps(saved))

    assert provenance_mismatch(saved, _provenance()) is None
    assert saved["success_rate"] == 0.5
    assert saved["commit"] == "abc123"
    assert saved["provenance"]["executor"] == "runs/selected/checkpoints/final"


def test_a_changed_missing_or_unrecorded_field_blocks_reuse() -> None:
    saved = with_provenance({}, _provenance(), None)

    assert provenance_mismatch(saved, _provenance(seed=202)) == "seed changed"
    vlm = {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "do_sample": False}
    assert provenance_mismatch(saved, _provenance(vlm=vlm)) == "vlm changed"
    assert provenance_mismatch(saved, _provenance(prior="abc")) == "prior changed"
    assert provenance_mismatch({"success_rate": 1.0}, _provenance()) == (
        "saved without provenance"
    )


def test_fingerprint_ignores_key_order() -> None:
    assert fingerprint({"a": 1, "b": [2, 3]}) == fingerprint({"b": [2, 3], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_a_reloaded_prior_has_the_fitted_prior_fingerprint(tmp_path) -> None:
    # The incremental preset reloads the prior a full run fitted, so the two
    # must fingerprint alike for its baselines to be reused.
    skills = [SkillId.COLLECT_WOOD] * 3 + [SkillId.PLACE_TABLE] * 2
    prior = fit_competence_prior(
        skills, [1, 4, None, 2, None], [True, True, False, True, False]
    )
    prior.save(tmp_path / "prior.json")
    reloaded = CompetencePrior.load(tmp_path / "prior.json")

    assert fingerprint(reloaded.to_dict()) == fingerprint(prior.to_dict())


def test_git_commit_marks_local_changes(tmp_path) -> None:
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    assert git_commit(tmp_path) is None
    git("init", "-q")
    (tmp_path / "file.txt").write_text("one")
    git("add", "file.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one")
    clean = git_commit(tmp_path)
    assert clean is not None and len(clean) == 40
    (tmp_path / "file.txt").write_text("two")
    assert git_commit(tmp_path) == f"{clean}+dirty"
