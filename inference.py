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


def run_inference(task_id: str):

    rewards = []
    steps = 0
    success = False

    try:
        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()

    except Exception as e:
        print(f"[START] task={task_id} env=hospital model={MODEL_NAME}")
        print(f"[END] success=false steps=0 score=0.00 rewards=")
        return

    print(f"[START] task={task_id} env=hospital model={MODEL_NAME}")

    done = False
    step_num = 0

    while not done:
        step_num += 1

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

            rewards.append(reward)

            print(
                f"[STEP] step={step_num} action={str(actions)} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

            if done:
                break

        except Exception as e:
            print(
                f"[STEP] step={step_num} action=null reward=0.00 done=true error={str(e)}"
            )
            break

    steps = len(rewards)

    total_reward = sum(rewards)
    score = max(1e-6, min(total_reward / 100.0, 1 - 1e-6))  # normalize

    success = score > 0.01

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}"
    )


if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task)