"""Opt-in integration test for the real open-weight VLM backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tiny_sima.config import load_vlm_planner_config
from tiny_sima.evaluation.vlm_smoke import run_vlm_smoke_test
from tiny_sima.goals import SKILL_IDS

RUN_VLM_SMOKE = os.environ.get("TINY_SIMA_RUN_VLM_SMOKE", "").lower() in {
    "1",
    "true",
    "yes",
}


@pytest.mark.skipif(
    not RUN_VLM_SMOKE,
    reason="Set TINY_SIMA_RUN_VLM_SMOKE=1 to load the real VLM.",
)
def test_real_vlm_backend_end_to_end() -> None:
    config = load_vlm_planner_config(Path("configs/vlm_planner_qwen3_2b.yaml"))

    result = run_vlm_smoke_test(config, seed=51)

    assert len(result.ranking) == len(SKILL_IDS)
    assert set(result.ranking) == {skill_id.value for skill_id in SKILL_IDS}
    assert result.grounded_feasibility > 0.0
