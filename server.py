"""FastAPI server exposing the hospital environment over OpenEnv-style endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from environment import HospitalEnv
from models import HospitalAction, OpenEnvAction


class ResetRequest(BaseModel):
    """Optional reset parameters for creating or reconfiguring an episode."""

    # The server can switch difficulty, random seed, and episode horizon on
    # every reset request so a single process can host many experiment setups.
    task_id: str = "hard"
    seed: int = 42
    max_ticks: Optional[int] = None


class StepRequest(BaseModel):
    """Action payload accepted by the step endpoint."""

    # `action` is intentionally typed as `Any` because callers may send:
    # - `null` for no action
    # - one action object
    # - a sparse list of actions
    # - a full per-slot action plan
    action: Any = Field(default=None)


app = FastAPI(
    title="Hospital Resource Environment",
    description="OpenEnv-style hospital scheduling environment server.",
    version="0.1.0",
)

# Keep one process-wide environment instance and replace it on `/reset`.
_env = HospitalEnv(seed=42, task_id="hard")


def _coerce_action_payload(action_payload: Any) -> Any:
    """Convert incoming JSON payloads into internal action models."""

    if action_payload is None:
        return None

    if isinstance(action_payload, list):
        # The step endpoint accepts batched actions, so lists are converted item
        # by item into the internal runtime action model.
        coerced_actions: List[HospitalAction] = []
        for action_item in action_payload:
            if isinstance(action_item, (HospitalAction, OpenEnvAction)):
                coerced_actions.append(action_item)
            elif isinstance(action_item, dict):
                coerced_actions.append(OpenEnvAction(**action_item).to_internal())
        return coerced_actions

    if isinstance(action_payload, (HospitalAction, OpenEnvAction)):
        # This path is mostly useful for in-process callers or tests that have
        # already built typed model instances.
        return action_payload

    if isinstance(action_payload, dict):
        # Single JSON actions are treated as OpenEnv-style payloads.
        return OpenEnvAction(**action_payload).to_internal()

    return None


@app.get("/")
async def root() -> Dict[str, str]:
    """Basic service metadata."""

    return {
        "service": "hospital-resource-env",
        "status": "ok",
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health-check endpoint used by deployment validators."""

    return {
        "status": "healthy",
        "service": "hospital-resource-env",
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()) -> Dict[str, Any]:
    """Reset the environment and return the initial typed observation."""

    global _env
    # Reset is also how callers switch task presets or seeds over HTTP.
    _env = HospitalEnv(seed=request.seed, max_ticks=request.max_ticks, task_id=request.task_id)
    # The reset response is only the observation, because reward/done/info are
    # meaningful only after at least one environment step.
    observation = _env.openenv_observation()
    return observation.model_dump()


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """Apply one step of action(s) and return observation, reward, done, and info."""

    action = _coerce_action_payload(request.action)
    # The internal env still returns the standard `(obs, reward, done, info)` tuple.
    # We intentionally return a fresh observation from `_env` instead of the raw
    # local `observation` variable so the API surface always goes through the
    # OpenEnv serialization path.
    observation, reward, done, info = _env.step(action)
    return {
        "observation": _env.openenv_observation().model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return the full current environment state."""

    return _env.state().model_dump()
