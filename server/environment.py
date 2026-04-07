"""
environment.py - The BBMP office simulation.
"""

import random
from typing import Dict, Any
from models import (
    BBMPObservation, BBMPAction, StepResult,
    ResetResult, StateResult, RoadComplaint
)

WARDS = {
    "Koramangala": ["80 Feet Road", "1st Main Road", "School Road", "Sony World Junction"],
    "HSR Layout":  ["Sector 1 Main", "27th Main", "BDA Complex Road", "Agara Lake Road"],
    "Whitefield":  ["ITPL Main Road", "Varthur Road", "Hope Farm Junction", "Kadugodi Road"],
    "Indiranagar": ["100 Feet Road", "CMH Road", "12th Main", "Defence Colony Road"],
    "BTM Layout":  ["Silkboard Junction", "15th Cross", "Arekere Road", "JP Nagar Road"],
}

ZONE_TYPES = {
    "80 Feet Road": "highway", "100 Feet Road": "highway",
    "ITPL Main Road": "highway", "Silkboard Junction": "highway",
    "School Road": "school", "Sector 1 Main": "school",
    "Sony World Junction": "commercial", "BDA Complex Road": "commercial",
    "Agara Lake Road": "residential", "Defence Colony Road": "residential",
}
# ── Contractors ────────────────────────────────────────────────────────────────
CONTRACTORS = {
    "QuickFix Pvt Ltd": {
        "rate_multiplier": 0.8,    # 20% cheaper
        "quality_score":   0.55,   # low quality
        "repair_longevity": 0.4,   # road breaks again soon
    },
    "RoadPro Solutions": {
        "rate_multiplier": 1.2,    # 20% more expensive
        "quality_score":   0.92,   # high quality
        "repair_longevity": 0.95,  # lasts long
    },
    "CheapBuild Co": {
        "rate_multiplier": 0.6,    # 40% cheaper
        "quality_score":   0.30,   # very low quality
        "repair_longevity": 0.2,   # breaks again quickly
    },
    "CityRepair BBMP": {
        "rate_multiplier": 1.0,    # standard rate
        "quality_score":   0.75,   # decent quality
        "repair_longevity": 0.80,  # good longevity
    },
}
SEVERITY_CONFIG = {
    "critical": {"traffic_impact": (0.7, 1.0), "pop": (3000, 8000), "patch": (12000, 20000), "full": (40000, 80000)},
    "high":     {"traffic_impact": (0.4, 0.7), "pop": (1000, 3000), "patch": (7000, 12000),  "full": (20000, 40000)},
    "medium":   {"traffic_impact": (0.2, 0.4), "pop": (400,  1000), "patch": (3000, 7000),   "full": (10000, 20000)},
    "low":      {"traffic_impact": (0.0, 0.2), "pop": (50,   400),  "patch": (1000, 3000),   "full": (5000,  10000)},
}

class BBMPEnvironment:
    def __init__(self, task_id: str = "task1"):
        self.task_id = task_id
        self.rng = random.Random(42)
        self._configure_task()
        self.reset()

    def _configure_task(self):
        configs = {
            "task1": {
                "description": "Single ward, 5 complaints. Fix critical roads first.",
                "wards": ["Koramangala"],
                "num_complaints": 5,
                "daily_budget": 100000,
                "crews": 2,
                "max_steps": 10,
                "rain_chance": 0.0,
            },
            "task2": {
                "description": "3 wards, 15 complaints. Balance budget across wards.",
                "wards": ["Koramangala", "HSR Layout", "Indiranagar"],
                "num_complaints": 15,
                "daily_budget": 300000,
                "crews": 4,
                "max_steps": 20,
                "rain_chance": 0.2,
            },
            "task3": {
                "description": "Full city. 5 wards, 30 complaints, rain creates new damage.",
                "wards": list(WARDS.keys()),
                "num_complaints": 30,
                "daily_budget": 500000,
                "crews": 6,
                "max_steps": 40,
                "rain_chance": 0.4,
            },
        }
        self.config = configs.get(self.task_id, configs["task1"])

    def reset(self) -> ResetResult:
        self.day = 1
        self.budget = self.config["daily_budget"]
        self.crews_available = self.config["crews"]
        self.steps_taken = 0
        self.total_reward = 0.0
        self.resolved_today = 0
        self.total_resolved = 0
        self.done = False
        self.weather = "sunny"
        self.complaints: Dict[str, RoadComplaint] = {}
        self._generate_complaints(self.config["num_complaints"])
        obs = self._get_observation()
        return ResetResult(
            observation=obs,
            task_id=self.task_id,
            task_description=self.config["description"],
            max_steps=self.config["max_steps"],
        )

    def step(self, action: BBMPAction) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._get_observation(),
                reward=0.0, done=True,
                info={"error": "Episode done. Call reset()."}
            )
        self.steps_taken += 1
        reward = 0.0
        info: Dict[str, Any] = {"action": action.action_type, "step": self.steps_taken}

        if action.action_type == "repair":
            reward, info = self._handle_repair(action, info)
        elif action.action_type == "inspect":
            reward, info = self._handle_inspect(action, info)
        elif action.action_type == "wait":
            reward = -2.0
            info["result"] = "Waited. -2 penalty."

        self._update_weather()
        self._age_complaints()

        if self.steps_taken >= self.config["max_steps"] or self.budget <= 0:
            self.done = True
            reward += self._end_of_episode_bonus()

        self.total_reward += reward
        return StepResult(
            observation=self._get_observation(),
            reward=round(reward, 3),
            done=self.done,
            info=info,
        )

    def state(self) -> StateResult:
        return StateResult(
            task_id=self.task_id,
            day=self.day,
            budget_remaining=round(self.budget, 2),
            total_reward=round(self.total_reward, 3),
            steps_taken=self.steps_taken,
            done=self.done,
            metrics=self._get_metrics(),
        )

    def _handle_repair(self, action, info):
        if not action.complaint_id or not action.repair_type:
            info["error"] = "Need complaint_id and repair_type"
            return -5.0, info
        complaint = self.complaints.get(action.complaint_id)
        if not complaint:
            info["error"] = f"Complaint {action.complaint_id} not found"
            return -5.0, info
        if self.crews_available <= 0:
            info["error"] = "No crews available"
            return -3.0, info
        cost = complaint.patch_cost if action.repair_type == "patch" else complaint.full_repair_cost
        if action.repair_type == "emergency":
            cost = int(complaint.full_repair_cost * 1.5)
        if cost > self.budget:
            info["error"] = f"Not enough budget. Need {cost}, have {self.budget}"
            return -3.0, info
        reward = self._calculate_reward(complaint, action.repair_type, cost)
        self.budget -= cost
        self.crews_available -= 1
        self.resolved_today += 1
        self.total_resolved += 1
        del self.complaints[action.complaint_id]
        info["result"] = f"Repaired {complaint.road_name} in {complaint.ward}"
        info["cost"] = cost
        info["reward"] = reward
        return reward, info

    def _calculate_reward(self, complaint, repair_type, cost):
        reward = {"critical": 10.0, "high": 7.0, "medium": 4.0, "low": 1.5}[complaint.severity]
        reward *= {"hospital": 2.0, "school": 1.8, "highway": 1.5, "commercial": 1.2, "residential": 1.0}.get(complaint.zone_type, 1.0)
        if complaint.days_pending > 7:
            reward += (complaint.days_pending - 7) * 0.5
        reward += complaint.traffic_impact * 5.0
        if cost / self.config["daily_budget"] < 0.05:
            reward += 2.0
        if complaint.severity in ["critical", "high"] and repair_type == "full_repair":
            reward += 3.0
        elif complaint.severity in ["low", "medium"] and repair_type == "patch":
            reward += 2.0
        elif complaint.severity == "critical" and repair_type == "patch":
            reward -= 2.0
        return round(reward, 2)

    def _handle_inspect(self, action, info):
        if not action.complaint_id:
            return -1.0, info
        complaint = self.complaints.get(action.complaint_id)
        if not complaint:
            return -1.0, info
        if self.crews_available <= 0:
            return -1.0, info
        self.crews_available -= 1
        info["result"] = f"Inspected {complaint.road_name}"
        return 0.5, info

    def _generate_complaints(self, count):
        severities = ["critical", "high", "medium", "low"]
        weights = [0.2, 0.3, 0.3, 0.2]
        for i in range(count):
            ward = self.rng.choice(self.config["wards"])
            road = self.rng.choice(WARDS[ward])
            severity = self.rng.choices(severities, weights=weights)[0]
            cfg = SEVERITY_CONFIG[severity]
            cid = f"{ward[:3].upper()}-{str(i+1).zfill(3)}"
            self.complaints[cid] = RoadComplaint(
                complaint_id=cid, ward=ward, road_name=road,
                zone_type=ZONE_TYPES.get(road, "residential"),
                severity=severity,
                days_pending=self.rng.randint(1, 15),
                traffic_impact=round(self.rng.uniform(*cfg["traffic_impact"]), 2),
                patch_cost=self.rng.randint(*cfg["patch"]),
                full_repair_cost=self.rng.randint(*cfg["full"]),
                population_affected=self.rng.randint(*cfg["pop"]),
                complaint_count=self.rng.randint(1, 50),
            )

    def _update_weather(self):
        if self.rng.random() < self.config["rain_chance"]:
            self.weather = self.rng.choice(["rain", "heavy_rain"])
            self._generate_complaints(self.rng.randint(1, 2))
        else:
            self.weather = self.rng.choice(["sunny", "cloudy"])
        self.crews_available = self.config["crews"]

    def _age_complaints(self):
        for c in self.complaints.values():
            c.days_pending += 1

    def _end_of_episode_bonus(self):
        bonus = 0.0
        for c in self.complaints.values():
            if c.severity == "critical":
                bonus -= 15.0
            elif c.severity == "high":
                bonus -= 5.0
        return bonus

    def _get_observation(self):
        return BBMPObservation(
            day=self.day,
            budget_remaining=round(self.budget, 2),
            total_budget=self.config["daily_budget"],
            crews_available=self.crews_available,
            crews_total=self.config["crews"],
            weather=self.weather,
            complaints=list(self.complaints.values()),
            resolved_today=self.resolved_today,
            total_resolved=self.total_resolved,
            task_id=self.task_id,
            task_description=self.config["description"],
            metrics=self._get_metrics(),
        )

    def _get_metrics(self):
        total = self.total_resolved + len(self.complaints)
        critical = sum(1 for c in self.complaints.values() if c.severity == "critical")
        return {
            "resolution_rate": round(self.total_resolved / max(1, total), 3),
            "budget_used_pct": round(1 - self.budget / self.config["daily_budget"], 3),
            "critical_pending": float(critical),
            "total_resolved": float(self.total_resolved),
            "avg_days_pending": round(
                sum(c.days_pending for c in self.complaints.values()) / max(1, len(self.complaints)), 1
            ),
        }