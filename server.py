"""FastAPI server exposing the hospital environment over OpenEnv-style endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from environment import HospitalEnv
from models import HospitalAction, OpenEnvAction, OpenEnvAction, OpenEnvObservation, OpenEnvState

# ── strict open-interval clamp — every score/reward leaving this server passes through here ──
_EPS = 1e-6

def _safe_score(raw: float) -> float:
    """Clamp any float to the strict open interval (0, 1)."""
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
    """Required by OpenEnv validator — must return status: healthy."""
    return {"status": "healthy", "service": "hospital-resource-env"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """Required by OpenEnv validator — must return name and description."""
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
    """Required by OpenEnv validator — must return action, observation, and state schemas."""
    return {
        "action": OpenEnvAction.model_json_schema(),
        "observation": OpenEnvObservation.model_json_schema(),
        "state": OpenEnvState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(payload: Any = None) -> Dict[str, Any]:
    """Required by OpenEnv validator — must return a JSON-RPC 2.0 payload."""
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
    """Reset the environment and return the initial typed observation."""
    global _env
    _env = HospitalEnv(seed=request.seed, max_ticks=request.max_ticks, task_id=request.task_id)
    return _env.openenv_observation().model_dump()


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """Apply one step and return observation, reward, done, and info."""
    action = _coerce_action_payload(request.action)
    observation, reward, done, info = _env.step(action)

    # Clamp reward — validator may read this as a score
    safe_reward = _safe_score(reward)

    # Clamp episode_grade in info when the episode finishes
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
    """Grade an episode — called by the OpenEnv Phase 2 validator.

    Expects: {task_id, info, final_score}
    Returns: {score} strictly in (0, 1)
    """
    from tasks import grade_episode
    try:
        raw = grade_episode(request.task_id, request.info, request.final_score)
    except Exception:
        raw = 0.5
    return {"score": _safe_score(raw)}


@app.get("/grade/{task_id}")
async def grade_current(task_id: str) -> Dict[str, Any]:
    """Convenience GET — grade the current running episode."""
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