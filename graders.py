"""
graders.py - Converts episode performance into a score 0.0 to 1.0
Judges run this to verify your environment works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

from environment import BBMPEnvironment
from models import BBMPAction


def run_episode(task_id: str, policy: str = "greedy") -> dict:
    """
    Run one full episode and return performance metrics.
    policy = greedy means always fix the highest severity road first.
    """
    env = BBMPEnvironment(task_id)
    result = env.reset()
    total_reward = 0.0
    steps = 0

    while not env.done:
        obs = result.observation if hasattr(result, 'observation') else result
        complaints = obs.complaints if hasattr(obs, 'complaints') else []

        if not complaints:
            action = BBMPAction(action_type="wait")
        else:
            # Greedy policy — pick highest severity complaint
            priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            best = max(complaints, key=lambda c: (
                priority[c.severity],
                c.traffic_impact,
                c.days_pending
            ))
            # Use patch for low/medium, full_repair for high/critical
            repair_type = "full_repair" if best.severity in ["critical", "high"] else "patch"
            action = BBMPAction(
                action_type="repair",
                complaint_id=best.complaint_id,
                repair_type=repair_type,
                reason=f"Fixing {best.severity} severity road"
            )

        result = env.step(action)
        total_reward += result.reward
        steps += 1

    return {
        "task_id": task_id,
        "total_reward": total_reward,
        "steps": steps,
        "metrics": env._get_metrics(),
        "total_resolved": env.total_resolved,
    }


def grade_task1() -> float:
    """
    Task 1 grader — single ward, 5 complaints.
    Score based on resolution rate and whether critical roads were fixed.
    """
    result = run_episode("task1")
    metrics = result["metrics"]

    # Score components
    resolution_score  = metrics["resolution_rate"]          # 0.0 to 1.0
    critical_score    = 1.0 if metrics["critical_pending"] == 0 else 0.3
    budget_score      = min(1.0, metrics["budget_used_pct"] * 1.5)

    # Weighted final score
    score = (
        resolution_score * 0.5 +
        critical_score   * 0.3 +
        budget_score     * 0.2
    )
    return round(min(1.0, max(0.0, score)), 4)


def grade_task2() -> float:
    """
    Task 2 grader — 3 wards, 15 complaints, weather changes.
    Score based on resolution rate, critical roads, and budget efficiency.
    """
    result = run_episode("task2")
    metrics = result["metrics"]

    resolution_score = metrics["resolution_rate"]
    critical_score   = 1.0 if metrics["critical_pending"] == 0 else max(0.0, 1.0 - metrics["critical_pending"] * 0.15)
    efficiency_score = min(1.0, result["total_resolved"] / 10.0)

    score = (
        resolution_score * 0.4 +
        critical_score   * 0.4 +
        efficiency_score * 0.2
    )
    return round(min(1.0, max(0.0, score)), 4)


def grade_task3() -> float:
    """
    Task 3 grader — full city, 30 complaints, rain events.
    Hardest task — score heavily penalises unresolved critical roads.
    """
    result = run_episode("task3")
    metrics = result["metrics"]

    resolution_score = metrics["resolution_rate"]
    critical_score   = max(0.0, 1.0 - metrics["critical_pending"] * 0.1)
    speed_score      = min(1.0, result["total_resolved"] / 20.0)
    avg_pending      = metrics["avg_days_pending"]
    urgency_score    = max(0.0, 1.0 - avg_pending / 20.0)

    score = (
        resolution_score * 0.35 +
        critical_score   * 0.35 +
        speed_score      * 0.15 +
        urgency_score    * 0.15
    )
    return round(min(1.0, max(0.0, score)), 4)


def run_all_graders():
    """Run all 3 graders and print scores."""
    print("\n" + "="*50)
    print("BBMP Road Repair Environment — Grader Results")
    print("="*50)

    task1_score = grade_task1()
    print(f"Task 1 (Easy)   — Score: {task1_score}")

    task2_score = grade_task2()
    print(f"Task 2 (Medium) — Score: {task2_score}")

    task3_score = grade_task3()
    print(f"Task 3 (Hard)   — Score: {task3_score}")

    print("="*50)
    print(f"All scores between 0.0 and 1.0: {all(0.0 <= s <= 1.0 for s in [task1_score, task2_score, task3_score])}")
    print("="*50 + "\n")

    return {
        "task1": task1_score,
        "task2": task2_score,
        "task3": task3_score,
    }


if __name__ == "__main__":
    run_all_graders()