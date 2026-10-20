"""Policy evaluation helpers."""

from tiny_sima.evaluation.affordance import (
    AffordanceSample,
    collect_affordance_samples,
    evaluate_affordance_calibration,
    evaluate_competence_prior,
    fit_affordance_calibration,
    fit_affordance_competence_prior,
)
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    BatchedSkillExecutor,
    ChainTermination,
    LearnedSkillExecutor,
    ScriptedSkillExecutor,
    SkillProbe,
    compare_paired_successes,
    evaluate_chain_episode,
    evaluate_chain_episodes,
    evaluate_learned_skill_chain_batched,
    evaluate_skill_chain,
    learned_skill_executor_from_checkpoint,
    make_learned_skill_executor,
    probe_skills_batched,
)
from tiny_sima.evaluation.goal_conditioned import (
    diagnose_goal_conditioned_checkpoint,
    diagnose_goal_conditioned_policy,
    diagnose_goal_conditioned_skill,
    evaluate_goal_conditioned_checkpoint,
    evaluate_goal_conditioned_policy,
    evaluate_goal_conditioned_skill,
    load_goal_conditioned_checkpoint,
)
from tiny_sima.evaluation.gpu_monitor import (
    NvidiaSmiMonitor,
    jax_allocator_summary,
)
from tiny_sima.evaluation.planner import (
    PlannerBoundaryTrace,
    PrefixBoundary,
    PrefixTrajectory,
    collect_prefix_trajectory,
    evaluate_planner_prefixes,
)
from tiny_sima.evaluation.policy import evaluate_checkpoint, evaluate_policy
from tiny_sima.evaluation.skills import (
    evaluate_skill_policy,
    evaluate_skill_suite,
    scripted_skill_action,
)
from tiny_sima.evaluation.vlm_smoke import VLMSmokeResult, run_vlm_smoke_test

__all__ = [
    "AffordanceSample",
    "BatchedSkillExecutor",
    "NvidiaSmiMonitor",
    "PlannerBoundaryTrace",
    "PrefixBoundary",
    "PrefixTrajectory",
    "VLMSmokeResult",
    "LearnedSkillExecutor",
    "ScriptedSkillExecutor",
    "SkillProbe",
    "ChainTermination",
    "WOOD_TO_STONE_CHAIN",
    "collect_affordance_samples",
    "collect_prefix_trajectory",
    "compare_paired_successes",
    "diagnose_goal_conditioned_checkpoint",
    "diagnose_goal_conditioned_policy",
    "diagnose_goal_conditioned_skill",
    "evaluate_affordance_calibration",
    "evaluate_checkpoint",
    "evaluate_chain_episode",
    "evaluate_chain_episodes",
    "evaluate_competence_prior",
    "evaluate_learned_skill_chain_batched",
    "evaluate_goal_conditioned_checkpoint",
    "evaluate_goal_conditioned_policy",
    "evaluate_goal_conditioned_skill",
    "evaluate_policy",
    "evaluate_planner_prefixes",
    "evaluate_skill_policy",
    "evaluate_skill_chain",
    "evaluate_skill_suite",
    "fit_affordance_calibration",
    "fit_affordance_competence_prior",
    "jax_allocator_summary",
    "learned_skill_executor_from_checkpoint",
    "load_goal_conditioned_checkpoint",
    "make_learned_skill_executor",
    "probe_skills_batched",
    "run_vlm_smoke_test",
    "scripted_skill_action",
]
