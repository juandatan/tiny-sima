"""End-to-end smoke test for the constrained multimodal planner backend."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax

from tiny_sima.config import VLMPlannerConfig, load_vlm_planner_config
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import WOOD_TO_STONE_CHAIN
from tiny_sima.goals import SKILL_IDS
from tiny_sima.planner.saycan import PlannerContext, SayCanPlanner
from tiny_sima.planner.vlm import (
    TransformersVLMBackend,
    VisionLanguageBackend,
    VLMUtilityScorer,
)


@dataclass(frozen=True)
class VLMSmokeResult:
    """Validated outputs from one real planner generation."""

    model_id: str
    elapsed_seconds: float
    ranking: tuple[str, ...]
    utility_selected_skill: str
    grounded_selected_skill: str
    grounded_feasibility: float
    rationale: str
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""

        return asdict(self)


def run_vlm_smoke_test(
    config: VLMPlannerConfig,
    *,
    backend: VisionLanguageBackend | None = None,
    scorer: VLMUtilityScorer | None = None,
    seed: int = 0,
) -> VLMSmokeResult:
    """Run one constrained VLM call and verify grounded planner integration."""

    if backend is not None and scorer is not None:
        raise ValueError("Pass either backend or scorer, not both.")
    validated_config = config.validate()
    environment = make_skill_chain_environment()
    state = environment.reset(jax.random.PRNGKey(seed)).state
    context = PlannerContext(objective=WOOD_TO_STONE_CHAIN.objective)
    utility_scorer = scorer or VLMUtilityScorer(
        backend=backend or TransformersVLMBackend(validated_config),
        config=validated_config,
    )

    started = time.perf_counter()
    utility_batch = utility_scorer.score_all(state, context, SKILL_IDS)
    elapsed_seconds = time.perf_counter() - started
    parsed = utility_scorer.last_response
    if parsed is None:
        raise RuntimeError("The VLM backend returned no parsed response.")
    if not utility_batch.valid:
        raise RuntimeError(
            "The VLM response failed validation: "
            f"{utility_batch.error}; raw={utility_batch.raw_response!r}"
        )
    if set(utility_batch.ranking) != set(SKILL_IDS):
        raise RuntimeError(
            "The VLM response was not a complete skill permutation: "
            f"{utility_batch.ranking!r}"
        )

    selection = SayCanPlanner(utility_scorer=utility_scorer).select_with_diagnostics(
        state, context
    )
    if selection.decision is None:
        raise RuntimeError(
            "The grounded planner returned no viable skill after a valid "
            f"ranking: status={selection.status.value}"
        )
    selected_score = next(
        score
        for score in selection.decision.scores
        if score.skill_id is selection.decision.skill_id
    )
    if selected_score.feasibility <= 0.0:
        raise RuntimeError(
            "The grounded planner selected a skill with zero feasibility."
        )

    utility_selected = utility_batch.ranking[0]
    return VLMSmokeResult(
        model_id=validated_config.model_id,
        elapsed_seconds=elapsed_seconds,
        ranking=tuple(skill_id.value for skill_id in utility_batch.ranking),
        utility_selected_skill=utility_selected.value,
        grounded_selected_skill=selection.decision.skill_id.value,
        grounded_feasibility=selected_score.feasibility,
        rationale=parsed.rationale,
        raw_response=parsed.raw_response,
    )


def main() -> None:
    """Run the smoke test from a configured command-line environment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vlm_planner_qwen3_2b.yaml"),
    )
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    result = run_vlm_smoke_test(
        load_vlm_planner_config(arguments.config),
        seed=arguments.seed,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
