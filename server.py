"""FastAPI server exposing the hospital environment over OpenEnv-style endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from environment import HospitalEnv
from models import HospitalAction, OpenEnvAction, OpenEnvObservation, OpenEnvState

_EPS = 1e-6


def _safe_score(raw: float) -> float:
    """Clamp any float to the strict open interval (0, 1). For scores only, not rewards."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = 0.5
    if v != v:   # NaN guard
        v = 0.5
    return max(_EPS, min(v, 1.0 - _EPS))


class ResetRequest(BaseModel):
    task_id: str = "hard"
    seed: int = 42
    max_ticks: Optional[int] = None


class StepRequest(BaseModel):
    action: Any = Field(default=None)


class GradeRequest(BaseModel):
    task_id: str
    info: Dict[str, Any] = Field(default_factory=dict)
    final_score: float = 0.0


app = FastAPI(
    title="Hospital Resource Environment",
    description="OpenEnv-style hospital scheduling environment for RL agent training.",
    version="0.1.0",
)

_env = HospitalEnv(seed=42, task_id="hard")


def _coerce_action_payload(action_payload: Any) -> Any:
    if action_payload is None:
        return None
    if isinstance(action_payload, list):
        coerced: List[HospitalAction] = []
        for item in action_payload:
            if isinstance(item, (HospitalAction, OpenEnvAction)):
                coerced.append(item)
            elif isinstance(item, dict):
                coerced.append(OpenEnvAction(**item).to_internal())
        return coerced
    if isinstance(action_payload, (HospitalAction, OpenEnvAction)):
        return action_payload
    if isinstance(action_payload, dict):
        return OpenEnvAction(**action_payload).to_internal()
    return None


# ── Core OpenEnv endpoints ──────────────────────────────────────────────────

@app.get("/")
async def root() -> Dict[str, str]:
    return {"service": "hospital-resource-env", "status": "ok"}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "hospital-resource-env"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    return {
        "name": "hospital_resource_env",
        "description": (
            "Hospital resource management environment where an agent allocates "
            "doctors and beds to patients with varying ESI acuity levels."
        ),
        "version": "0.1.0",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    return {
        "action": OpenEnvAction.model_json_schema(),
        "observation": OpenEnvObservation.model_json_schema(),
        "state": OpenEnvState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(payload: Any = None) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "name": "hospital_resource_env",
            "description": "Hospital resource management OpenEnv environment.",
        },
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    global _env
    _env = HospitalEnv(seed=request.seed, max_ticks=request.max_ticks, task_id=request.task_id)
    return _env.openenv_observation().model_dump()


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    action = _coerce_action_payload(request.action)
    observation, reward, done, info = _env.step(action)

    # FIX: reward is NOT clamped — it can be negative (penalties, deaths, costs).
    # Only final grade/score sent to the validator needs (0, 1) clamping.
    raw_reward = float(reward)

    if "episode_grade" in info:
        info["episode_grade"] = _safe_score(info["episode_grade"])

    return {
        "observation": _env.openenv_observation().model_dump(),
        "reward": raw_reward,
        "done": done,
        "info": info,
    }


@app.post("/grade")
async def grade(request: GradeRequest) -> Dict[str, Any]:
    from tasks import grade_episode
    try:
        raw = grade_episode(request.task_id, request.info, request.final_score)
    except Exception:
        raw = 0.5
    return {"score": _safe_score(raw)}


@app.get("/grade/{task_id}")
async def grade_current(task_id: str) -> Dict[str, Any]:
    from tasks import grade_episode
    try:
        info = _env._build_info()
        raw = grade_episode(task_id, info, _env.current_score)
    except Exception:
        raw = 0.5
    return {"score": _safe_score(raw)}


@app.get("/state")
async def state() -> Dict[str, Any]:
    return _env.state().model_dump()
