"""
inference.py - Baseline agent for BBMP Road Repair Environment.
Judges run this file to verify the environment works end to end.
Must print [START], [STEP], [END] logs in exact format.
"""

import sys
import os
import json
import time
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))
sys.path.insert(0, os.path.dirname(__file__))

from environment import BBMPEnvironment
from models import BBMPAction

# ── Config — judges check these variables exist ────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.anthropic.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "claude-sonnet-4-20250514")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key",
    base_url=API_BASE_URL,
)


def get_llm_action(obs_dict: dict, task_id: str) -> dict:
    """
    Ask the LLM to decide which road to repair next.
    Returns a dict with action_type, complaint_id, repair_type.
    """
    complaints = obs_dict.get("complaints", [])
    if not complaints:
        return {"action_type": "wait"}

    # Build prompt
    complaints_text = "\n".join([
        f"- ID: {c['complaint_id']} | Road: {c['road_name']} | Ward: {c['ward']} | "
        f"Severity: {c['severity']} | Zone: {c['zone_type']} | "
        f"Days pending: {c['days_pending']} | Traffic impact: {c['traffic_impact']} | "
        f"Patch cost: Rs{c['patch_cost']} | Full repair cost: Rs{c['full_repair_cost']}"
        for c in complaints
    ])

    prompt = f"""You are a BBMP municipal officer in Bengaluru deciding which road to repair.

Current situation:
- Day: {obs_dict['day']}
- Budget remaining: Rs{obs_dict['budget_remaining']}
- Crews available: {obs_dict['crews_available']}
- Weather: {obs_dict['weather']}

Active road complaints:
{complaints_text}

Choose ONE complaint to repair. Prioritize:
1. Critical severity first
2. High traffic impact roads
3. Schools and hospitals
4. Roads pending longest

Respond with ONLY a JSON object like this:
{{"action_type": "repair", "complaint_id": "KOR-001", "repair_type": "patch", "reason": "critical road"}}

repair_type must be "patch" for low/medium severity, "full_repair" for high/critical severity.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content.strip()
        # Clean up response
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception:
        # Fallback to greedy if LLM fails
        return greedy_action(complaints, obs_dict["budget_remaining"])


def greedy_action(complaints: list, budget: float) -> dict:
    """Fallback greedy policy — always fix highest severity road."""
    priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    affordable = [
        c for c in complaints
        if c["patch_cost"] <= budget or c["full_repair_cost"] <= budget
    ]
    if not affordable:
        return {"action_type": "wait"}

    best = max(affordable, key=lambda c: (
        priority[c["severity"]],
        c["traffic_impact"],
        c["days_pending"]
    ))
    repair_type = "full_repair" if best["severity"] in ["critical", "high"] else "patch"
    return {
        "action_type": "repair",
        "complaint_id": best["complaint_id"],
        "repair_type": repair_type,
        "reason": f"Greedy: fixing {best['severity']} severity road"
    }


def run_task(task_id: str) -> float:
    """Run one full episode for a task and return final score."""
    env = BBMPEnvironment(task_id)
    reset_result = env.reset()
    obs = reset_result.observation

    # ── [START] log — required by judges ──────────────────────────────────────
    print(json.dumps({
        "event":       "START",
        "task_id":     task_id,
        "description": reset_result.task_description,
        "max_steps":   reset_result.max_steps,
        "budget":      obs.budget_remaining,
        "complaints":  len(obs.complaints),
    }))

    step_num    = 0
    total_reward = 0.0

    while not env.done:
        obs_dict = obs.model_dump()

        # Get action from LLM or greedy fallback
        action_dict = get_llm_action(obs_dict, task_id)

        # Build action object
        try:
            action = BBMPAction(**action_dict)
        except Exception:
            action = BBMPAction(action_type="wait")

        # Step the environment
        step_result  = env.step(action)
        total_reward += step_result.reward
        obs           = step_result.observation
        step_num     += 1

        # ── [STEP] log — required by judges ───────────────────────────────────
        print(json.dumps({
            "event":             "STEP",
            "task_id":           task_id,
            "step":              step_num,
            "action_type":       action.action_type,
            "complaint_id":      action.complaint_id,
            "repair_type":       action.repair_type,
            "reward":            step_result.reward,
            "done":              step_result.done,
            "budget_remaining":  obs.budget_remaining,
            "complaints_left":   len(obs.complaints),
            "total_resolved":    obs.total_resolved,
        }))

        if step_result.done:
            break

    # Compute final grade
    from graders import grade_task1, grade_task2, grade_task3
    graders = {"task1": grade_task1, "task2": grade_task2, "task3": grade_task3}
    final_score = graders[task_id]()

    # ── [END] log — required by judges ────────────────────────────────────────
    print(json.dumps({
        "event":          "END",
        "task_id":        task_id,
        "total_steps":    step_num,
        "total_reward":   round(total_reward, 3),
        "final_score":    final_score,
        "total_resolved": env.total_resolved,
        "metrics":        env._get_metrics(),
    }))

    return final_score


def main():
    """Run all 3 tasks and print final scores."""
    print(json.dumps({"event": "START", "task_id": "all", "description": "Running all tasks"}))

    scores = {}
    for task_id in ["task1", "task2", "task3"]:
        print(f"\n--- Running {task_id} ---")
        score = run_task(task_id)
        scores[task_id] = score
        time.sleep(0.5)

    print(json.dumps({
        "event":  "END",
        "task_id": "all",
        "scores": scores,
        "pass":   all(0.0 <= s <= 1.0 for s in scores.values())
    }))


if __name__ == "__main__":
    main()