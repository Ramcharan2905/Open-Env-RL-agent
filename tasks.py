"""
Task presets and deterministic graders for the hospital environment.
FINAL VERSION — OpenEnv compliant and validator-safe
"""

from typing import Any, Dict
from models import EnvConfig, ModelBase


# ==============================
# Task Definition
# ==============================

class TaskDefinition(ModelBase):
    task_id: str
    name: str
    difficulty: str
    description: str
    max_ticks: int = 50
    config: EnvConfig


def build_task_definition(task_id: str, base_config: EnvConfig) -> TaskDefinition:

    if task_id == "easy":
        return TaskDefinition(
            task_id="easy",
            name="Low-Stakes Shift",
            difficulty="easy",
            description="Basic hospital flow without penalties.",
            config=base_config.model_copy(
                update={
                    "use_wait_penalty": False,
                    "use_post_treat_penalty": False,
                    "use_doctor_costs": False,
                    "use_wealth_multiplier": False,
                    "use_invalid_action_penalty": False,
                    "doctor_costs": {1: 0.0, 2: 0.0, 3: 0.0},
                }
            ),
        )

    if task_id == "medium":
        return TaskDefinition(
            task_id="medium",
            name="Resource Balance",
            difficulty="medium",
            description="Doctor costs + congestion.",
            config=base_config.model_copy(
                update={
                    "total_t1_docs": 4,
                    "total_t2_docs": 2,
                    "total_t3_docs": 1,
                    "total_beds": 8,
                    "mean_arrivals_per_tick": 2.6,
                    "max_arrivals_per_tick": 10,
                    "doctor_costs": {1: 3.0, 2: 8.0, 3: 18.0},
                }
            ),
        )

    if task_id == "hard":
        return TaskDefinition(
            task_id="hard",
            name="Full Hospital Control",
            difficulty="hard",
            description="Full penalties + surge.",
            config=base_config.model_copy(
                update={
                    "total_t1_docs": 3,
                    "total_t2_docs": 2,
                    "total_t3_docs": 1,
                    "total_beds": 5,
                    "mean_arrivals_per_tick": 3.2,
                    "max_arrivals_per_tick": 12,
                    "death_penalty": 300.0,
                    "doctor_costs": {1: 4.0, 2: 10.0, 3: 22.0},
                }
            ),
        )

    raise ValueError(f"Unknown task_id: {task_id}")


# ==============================
# SAFE HELPERS
# ==============================

def _safe_div(n, d):
    return n / d if d > 0 else 0.0


def _clip01(x):
    return max(0.0, min(1.0, x))


def _survival(deaths, total):
    return _clip01(1.0 - _safe_div(deaths, total))


def _throughput(discharged, total):
    return _clip01(_safe_div(discharged, total))


def _wait_score(wait):
    return _clip01(1.0 / (1.0 + max(wait, 0.0)))


def _reward_score(score, scale):
    return _clip01(score / scale)


def _reward_penalty(score, scale):
    return min(0.4, max(0.0, -score / scale))


def _death_penalty(deaths, total, cap):
    rate = _safe_div(deaths, total)
    return min(cap, (rate ** 1.5) * cap * 2.0)


def _legality_penalty(illegals, total):
    return min(0.08, _safe_div(illegals, total) * 0.5)


def _doctor_efficiency(t1, t2, t3):
    deviation = 0.5 * abs(t1 - 0.7) + 0.3 * abs(t2 - 0.45) + 0.2 * abs(t3 - 0.25)
    return _clip01(1.0 - deviation)


# ==============================
# FINAL GRADER
# ==============================

def grade_episode(task_id: str, info: Dict[str, Any], final_score: float) -> float:

    total = max(int(info.get("total_patients_seen", 0)), 1)
    deaths = max(0, int(info.get("total_deaths", 0)))
    discharged = max(0, int(info.get("total_discharged", 0)))
    illegals = max(0, int(info.get("illegal_actions_attempted", 0)))

    wait = max(0.0, float(info.get("avg_wait_time", 0.0)))
    t1 = max(0.0, float(info.get("avg_t1_doctor_utilization", 0.0)))
    t2 = max(0.0, float(info.get("avg_t2_doctor_utilization", 0.0)))
    t3 = max(0.0, float(info.get("avg_t3_doctor_utilization", 0.0)))

    # Guard against NaN in final_score
    final_score = float(final_score) if final_score == final_score else 0.0

    surv = _survival(deaths, total)
    thru = _throughput(discharged, total)
    wait_s = _wait_score(wait)
    eff = _doctor_efficiency(t1, t2, t3)
    leg = _legality_penalty(illegals, total)

    if task_id == "easy":
        rew = _reward_score(final_score, 4000)
        rew_p = _reward_penalty(final_score, 8000)  # FIX: now applied below
        d_p = _death_penalty(deaths, total, 0.2)
        score = 0.55 * surv + 0.25 * thru + 0.20 * rew

    elif task_id == "medium":
        rew = _reward_score(final_score, 2500)
        rew_p = _reward_penalty(final_score, 5000)
        d_p = _death_penalty(deaths, total, 0.25)
        score = 0.40 * surv + 0.20 * thru + 0.20 * wait_s + 0.10 * eff + 0.10 * rew

    elif task_id == "hard":
        rew = _reward_score(final_score, 1500)
        rew_p = _reward_penalty(final_score, 3000)
        d_p = _death_penalty(deaths, total, 0.35)
        score = 0.45 * surv + 0.15 * thru + 0.20 * wait_s + 0.10 * eff + 0.10 * rew

    else:
        raise ValueError(f"Invalid task_id: {task_id}")

    # FIX: apply ALL penalties (rew_p was missing from all tasks previously)
    score = score - (d_p + rew_p + leg)

    # Guard against NaN
    if not isinstance(score, float) or score != score:
        score = 0.5

    # STRICT open interval (0, 1) — validator requires score != 0.0 and score != 1.0
    EPS = 1e-6
    return max(EPS, min(score, 1.0 - EPS))