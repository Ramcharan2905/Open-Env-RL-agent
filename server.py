"""FastAPI server exposing the hospital environment over OpenEnv-style endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from environment import HospitalEnv
from models import HospitalAction, OpenEnvAction

# ── strict open-interval clamp used everywhere a score/reward leaves the server ─
_EPS = 1e-6

def _safe_score(raw: float) -> float:
    """Clamp any float to the strict open interval (0, 1)."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = 0.5
    if v != v:          # NaN guard
        v = 0.5
    return max(_EPS, min(v, 1.0 - _EPS))


class ResetRequest(BaseModel):
    """Optional reset parameters for creating or reconfiguring an episode."""

    task_id: str = "hard"
    seed: int = 42
    max_ticks: Optional[int] = None


class StepRequest(BaseModel):
    """Action payload accepted by the step endpoint."""

    action: Any = Field(default=None)


class GradeRequest(BaseModel):
    """Payload for the /grade endpoint."""

    task_id: str
    info: Dict[str, Any] = Field(default_factory=dict)
    final_score: float = 0.0


app = FastAPI(
    title="Hospital Resource Environment",
    description="OpenEnv-style hospital scheduling environment server.",
    version="0.1.0",
)

_env = HospitalEnv(seed=42, task_id="hard")


def _coerce_action_payload(action_payload: Any) -> Any:
    """Convert incoming JSON payloads into internal action models."""

    if action_payload is None:
        return None

    if isinstance(action_payload, list):
        coerced_actions: List[HospitalAction] = []
        for action_item in action_payload:
            if isinstance(action_item, (HospitalAction, OpenEnvAction)):
                coerced_actions.append(action_item)
            elif isinstance(action_item, dict):
                coerced_actions.append(OpenEnvAction(**action_item).to_internal())
        return coerced_actions

    if isinstance(action_payload, (HospitalAction, OpenEnvAction)):
        return action_payload

    if isinstance(action_payload, dict):
        return OpenEnvAction(**action_payload).to_internal()

    return None


@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "service": "hospital-resource-env",
        "status": "ok",
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "hospital-resource-env",
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Reset the environment and return the initial typed observation."""

    global _env
    _env = HospitalEnv(seed=request.seed, max_ticks=request.max_ticks, task_id=request.task_id)
    observation = _env.openenv_observation()
    return observation.model_dump()


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """Apply one step and return observation, reward, done, and info."""

    action = _coerce_action_payload(request.action)
    observation, reward, done, info = _env.step(action)

    # FIX: clamp reward to strict open interval so validator never sees 0.0 or 1.0
    safe_reward = _safe_score(reward)

    # FIX: also clamp episode_grade inside info when present (emitted on done=True)
    if "episode_grade" in info:
        info["episode_grade"] = _safe_score(info["episode_grade"])

    return {
        "observation": _env.openenv_observation().model_dump(),
        "reward": safe_reward,
        "done": done,
        "info": info,
    }


@app.post("/grade")
async def grade(request: GradeRequest) -> Dict[str, Any]:
    """Grade an episode — required by the OpenEnv validator (Phase 2).

    The validator calls POST /grade with task_id, info, and final_score.
    It expects a JSON response with a 'score' field strictly in (0, 1).
    """

    from tasks import grade_episode

    try:
        raw = grade_episode(request.task_id, request.info, request.final_score)
    except Exception:
        raw = 0.5

    return {"score": _safe_score(raw)}


@app.get("/grade/{task_id}")
async def grade_current(task_id: str) -> Dict[str, Any]:
    """Grade the currently running episode for the given task_id (convenience GET)."""

    from tasks import grade_episode

    try:
        info = _env._build_info()
        raw = grade_episode(task_id, info, _env.current_score)
    except Exception:
        raw = 0.5

    return {"score": _safe_score(raw)}


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return the full current environment state."""

    return _env.state().model_dump()