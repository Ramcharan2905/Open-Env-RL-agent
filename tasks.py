"""Task presets and deterministic graders for the hospital environment.

Grader improvements over v1:
  - Reward normalisation is calibrated per-task so a decent agent scores
    in the 0.3–0.7 range rather than being crushed by large negative rewards.
  - Each task weights sub-scores to match its actual difficulty contract:
    easy cares about survival + throughput, hard cares about survival + wait + efficiency.
  - The doctor-efficiency score penalises agents that over-use expensive tiers
    on patients that didn't need them (medium and hard only).
  - The death penalty curve is quadratic rather than linear so the grader
    distinguishes between "a few deaths" and "catastrophic failure" more cleanly.
  - All sub-scores are clipped to [0, 1] before weighting to prevent any single
    dimension from dragging the total below zero on its own.
"""

from typing import Any, Dict

from models import EnvConfig, ModelBase


class TaskDefinition(ModelBase):
    """Describes one hackathon task preset."""

    task_id: str
    name: str
    difficulty: str
    description: str
    max_ticks: int = 50
    config: EnvConfig


def build_task_definition(task_id: str, base_config: EnvConfig) -> TaskDefinition:
    """Return a task preset derived from the shared base configuration."""

    if task_id == "easy":
        return TaskDefinition(
            task_id="easy",
            name="Low-Stakes Shift",
            difficulty="easy",
            description=(
                "Only death penalties and positive discharge rewards are active. "
                "Teaches the agent the basic triage → treat → discharge pipeline "
                "without economic or congestion pressure."
            ),
            max_ticks=50,
            config=base_config.model_copy(
                update={
                    "use_wait_penalty":           False,
                    "use_post_treat_penalty":     False,
                    "use_doctor_costs":           False,
                    "use_wealth_multiplier":      False,
                    "use_invalid_action_penalty": False,
                    "doctor_costs":               {1: 0.0, 2: 0.0, 3: 0.0},
                }
            ),
        )

    if task_id == "medium":
        return TaskDefinition(
            task_id="medium",
            name="Resource Balance",
            difficulty="medium",
            description=(
                "Tighter capacity with doctor costs and busier arrivals. "
                "Tests resource allocation under congestion "
                "without wealth-based ethical tension."
            ),
            max_ticks=50,
            config=base_config.model_copy(
                update={
                    "total_t1_docs":              4,
                    "total_t2_docs":              2,
                    "total_t3_docs":              1,
                    "total_beds":                 8,
                    "mean_arrivals_per_tick":     2.6,
                    "max_arrivals_per_tick":      10,
                    "discharge_threshold":        68.0,
                    "use_wait_penalty":           False,
                    "use_post_treat_penalty":     False,
                    "use_wealth_multiplier":      False,
                    "doctor_costs":               {1: 3.0, 2: 8.0, 3: 18.0},
                }
            ),
        )

    if task_id == "hard":
        return TaskDefinition(
            task_id="hard",
            name="Full Hospital Control",
            difficulty="hard",
            description=(
                "Severe surge scenario with stronger bottlenecks, higher arrivals, "
                "full penalty suite, and the wealth/cost multi-objective trade-off."
            ),
            max_ticks=50,
            config=base_config.model_copy(
                update={
                    "total_t1_docs":              3,
                    "total_t2_docs":              2,
                    "total_t3_docs":              1,
                    "total_beds":                 5,
                    "mean_arrivals_per_tick":     3.2,
                    "max_arrivals_per_tick":      12,
                    "recovery_rate":              1.5,
                    "discharge_threshold":        75.0,
                    "wait_penalty_multiplier":    3.2,
                    "post_treat_penalty":         16.0,
                    "death_penalty":              300.0,
                    "doctor_costs":               {1: 4.0, 2: 10.0, 3: 22.0},
                }
            ),
        )

    raise ValueError(f"Unknown task_id: {task_id!r}. Must be 'easy', 'medium', or 'hard'.")


# ── Grader helpers ────────────────────────────────────────────────────────────

def _survival(deaths: int, total_seen: int) -> float:
    """Fraction of patients who did not die, clipped to [0, 1]."""
    return max(0.0, min(1.0, 1.0 - deaths / total_seen))


def _throughput(discharged: int, total_seen: int) -> float:
    """Fraction of seen patients successfully discharged, clipped to [0, 1]."""
    return max(0.0, min(1.0, discharged / total_seen))


def _wait_score(avg_wait_time: float) -> float:
    """Inversely proportional to average wait time; 0 wait → 1.0, ∞ → 0.0."""
    return max(0.0, min(1.0, 1.0 / (1.0 + max(avg_wait_time, 0.0))))


def _reward_score(final_score: float, scale: float) -> float:
    """Normalise cumulative reward to [0, 1] using a task-specific scale."""
    return max(0.0, min(1.0, final_score / scale))


def _reward_penalty(final_score: float, scale: float) -> float:
    """Convert large negative cumulative reward into a [0, 0.4] penalty."""
    return min(0.4, max(0.0, -final_score / scale))


def _death_penalty_quadratic(deaths: int, total_seen: int, cap: float) -> float:
    """Quadratic death penalty: distinguishes a few deaths from catastrophe.

    A linear death rate of d/n maps to a quadratic penalty of (d/n)^1.5 * cap
    so that going from 0% to 5% deaths costs much less than going from
    20% to 25% deaths. This gives agents partial credit for not completely failing.
    """
    rate = deaths / total_seen
    return min(cap, (rate ** 1.5) * cap * 2.0)


def _legality_penalty(illegal_actions: int, total_seen: int) -> float:
    """Small penalty for illegal actions — enough to signal quality, not dominate."""
    return min(0.08, illegal_actions / max(total_seen, 1) * 0.5)


def _doctor_efficiency_score(
    avg_t1_utilization: float,
    avg_t2_utilization: float,
    avg_t3_utilization: float,
) -> float:
    """Reward balanced doctor usage.

    A good agent uses T1 heavily and T3 sparingly.
    Score is based on a weighted utilization balance:
      ideal ≈ T1 high, T2 moderate, T3 low.
    We score by comparing the agent's utilization profile against the ideal.
    """
    # Target utilization profile: T1 busy (0.7), T2 moderate (0.45), T3 light (0.25)
    t1_diff = abs(avg_t1_utilization - 0.70)
    t2_diff = abs(avg_t2_utilization - 0.45)
    t3_diff = abs(avg_t3_utilization - 0.25)
    # Average weighted deviation (T1 matters most)
    deviation = 0.50 * t1_diff + 0.30 * t2_diff + 0.20 * t3_diff
    return max(0.0, min(1.0, 1.0 - deviation))


# ── Main grader ───────────────────────────────────────────────────────────────

def grade_episode(task_id: str, info: Dict[str, Any], final_score: float) -> float:
    """Score a completed episode from 0.0 to 1.0 using deterministic metrics.

    Sub-scores:
      survival_score       — fraction of patients who survived
      throughput_score     — fraction of patients discharged
      wait_score           — inverse of average waiting time
      reward_score         — normalised cumulative reward (task-calibrated)
      doctor_eff_score     — how well the agent matched the ideal doctor-tier mix

    Penalties applied after weighting:
      death_penalty        — quadratic, up to 0.35 on hard
      reward_penalty       — for strongly negative cumulative reward
      legality_penalty     — for illegal action rate
    """

    total_seen   = max(int(info.get("total_patients_seen", 0)), 1)
    deaths       = int(info.get("total_deaths", 0))
    discharged   = int(info.get("total_discharged", 0))
    illegals     = int(info.get("illegal_actions_attempted", 0))
    avg_wait     = float(info.get("avg_wait_time", 0.0))
    avg_t1_util  = float(info.get("avg_t1_doctor_utilization", 0.0))
    avg_t2_util  = float(info.get("avg_t2_doctor_utilization", 0.0))
    avg_t3_util  = float(info.get("avg_t3_doctor_utilization", 0.0))

    surv   = _survival(deaths, total_seen)
    thru   = _throughput(discharged, total_seen)
    wait   = _wait_score(avg_wait)
    eff    = _doctor_efficiency_score(avg_t1_util, avg_t2_util, avg_t3_util)
    leg_p  = _legality_penalty(illegals, total_seen)

    if task_id == "easy":
        # Reward scale: a decent agent on easy should reach ~4 000 cumulative
        # (50 ticks × ~20 discharge reward × ~4 patients per tick, minus a few deaths).
        rew    = _reward_score(final_score, scale=4000.0)
        rew_p  = _reward_penalty(final_score, scale=8000.0)
        d_p    = _death_penalty_quadratic(deaths, total_seen, cap=0.20)

        score = (
            0.55 * surv
            + 0.25 * thru
            + 0.20 * rew
        )
        score -= d_p + rew_p + leg_p

    elif task_id == "medium":
        # Reward scale: medium has doctor costs, so net cumulative is lower (~2 500)
        rew    = _reward_score(final_score, scale=2500.0)
        rew_p  = _reward_penalty(final_score, scale=5000.0)
        d_p    = _death_penalty_quadratic(deaths, total_seen, cap=0.25)

        score = (
            0.40 * surv
            + 0.20 * thru
            + 0.20 * wait
            + 0.10 * eff
            + 0.10 * rew
        )
        score -= d_p + rew_p + leg_p

    elif task_id == "hard":
        # Reward scale: hard has heavy penalties so net cumulative may be ~1 000
        rew    = _reward_score(final_score, scale=1500.0)
        rew_p  = _reward_penalty(final_score, scale=3000.0)
        d_p    = _death_penalty_quadratic(deaths, total_seen, cap=0.35)

        score = (
            0.45 * surv
            + 0.15 * thru
            + 0.20 * wait
            + 0.10 * eff
            + 0.10 * rew
        )
        score -= d_p + rew_p + leg_p

    else:
        raise ValueError(f"Unknown task_id: {task_id!r}")

    epsilon = 1e-6
    score = max(epsilon, min(score, 1 - epsilon))
    return score
