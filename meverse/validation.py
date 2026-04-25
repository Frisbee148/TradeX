"""Validation helpers for tasks and graders."""

from __future__ import annotations

from typing import Dict

from .env import load_repo_env
from .models import SurveillanceAction
from .policy import build_llm_client, load_policy_config, select_action
from .server.meverse_environment import MarketSurveillanceEnvironment
from .tasks import list_task_names

load_repo_env()


def run_task(task_name: str) -> Dict[str, float]:
    config = load_policy_config()
    client = build_llm_client(config)
    if client is None:
        raise RuntimeError("HF_TOKEN is required. Set it in .env or as an environment variable.")
    env = MarketSurveillanceEnvironment(task=task_name)
    observation = env.reset(task=task_name)
    while not observation.done:
        action = select_action(observation, client=client, config=config, allow_fallback=False)
        observation = env.step(SurveillanceAction(action_type=action))
    return env.grade()


def run_validation_suite() -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    print("Running task validation (LLM policy)...")
    for task_name in list_task_names():
        grade = run_task(task_name)
        score = grade["score"]
        steps = grade.get("steps_run", "?")
        print(f"{task_name}: score={score:.4f} steps={steps}")
        assert 0.0 <= score <= 1.0, f"Score out of range for {task_name}: {score}"
        results[task_name] = grade
    return results


if __name__ == "__main__":
    run_validation_suite()
