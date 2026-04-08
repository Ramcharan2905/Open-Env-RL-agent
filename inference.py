"""
inference.py — Root-level baseline runner for the Hospital Resource Environment.

Hackathon requirements:
- Uses OpenAI client for all LLM calls
- Uses API_BASE_URL for the LLM endpoint
- Uses ENV_BASE_URL for the environment server
- Emits strict [START], [STEP], [END] stdout logs
"""

import json
import os
from typing import Any, List

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://ramcharan2905-hospital-resource-env.hf.space",
).rstrip("/")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")


def _extract_json_array(text: str) -> List[dict[str, Any]]:
    if not text:
        return []

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []

    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def get_action_from_llm(observation: dict[str, Any]) -> List[dict[str, Any]]:
    if not HF_TOKEN:
        return []

    prompt = f"""
You are a hospital resource manager.

Rules:
- Prioritize ESI-1, then ESI-2.
- Use the correct doctor tier.
- Assign beds when needed.
- Discharge patients as soon as they are ready.
- Avoid illegal actions.

State:
{json.dumps(observation, ensure_ascii=False)}

Output ONLY a JSON array of actions like:
[
  {{"patient_id": "p1", "type": "assign_doctor", "doctor_tier": "t1"}},
  {{"patient_id": "p2", "type": "assign_bed"}},
  {{"patient_id": "p3", "type": "discharge"}}
]
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return _extract_json_array(content.strip())
    except Exception:
        return []


def run_inference(task_id: str) -> None:
    try:
        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()
    except Exception as exc:
        print("[START]")
        print(json.dumps({"task_id": task_id, "model": MODEL_NAME}))
        print("[END]")
        print(json.dumps({"task_id": task_id, "error": f"reset failed: {exc}"}))
        return

    print("[START]")
    print(json.dumps({"task_id": task_id, "model": MODEL_NAME}))

    done = False
    tick = 0
    total_score = float(obs.get("current_episode_score", 0.0))

    while not done:
        actions = get_action_from_llm(obs)

        try:
            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": actions},
                timeout=30,
            )
            step_resp.raise_for_status()
            data = step_resp.json()

            obs = data["observation"]
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            tick = int(obs.get("current_tick", tick + 1))
            total_score = float(obs.get("current_episode_score", total_score))

            print("[STEP]")
            print(
                json.dumps(
                    {
                        "tick": tick,
                        "reward": round(reward, 2),
                        "score_so_far": round(total_score, 2),
                    }
                )
            )
        except Exception as exc:
            print("[STEP]")
            print(json.dumps({"tick": tick, "error": str(exc)}))
            break

    print("[END]")
    print(
        json.dumps(
            {
                "task_id": task_id,
                "final_score": round(total_score, 4),
            }
        )
    )


if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task)