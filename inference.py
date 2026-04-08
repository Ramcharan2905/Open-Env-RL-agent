"""
inference.py — OpenEnv Client for Hospital Resource Environment

Complies with hackathon requirements:
- Uses OpenAI client for decisions
- Uses ENV_BASE_URL for environment
- Uses API_BASE_URL for LLM (optional override)
- Strict [START], [STEP], [END] logs
"""

import os
import json
import requests
from openai import OpenAI

# ==============================
# Environment variables
# ==============================

ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://ramcharan2905-hospital-resource-env.hf.space"
)

API_BASE_URL = os.getenv("API_BASE_URL", None)  # optional override

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN   = os.getenv("HF_TOKEN", "")

# OpenAI client (LLM ONLY)
if API_BASE_URL:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
else:
    client = OpenAI(api_key=HF_TOKEN)


# ==============================
# LLM Decision
# ==============================

def get_action_from_llm(observation):
    prompt = f"""
You are a Hospital Resource Manager.

Rules:
- Prioritize ESI-1, then ESI-2
- Assign correct doctor tier
- Assign beds when needed
- Discharge immediately when treated
- Avoid illegal actions

State:
{json.dumps(observation)}

Output ONLY a JSON array:
[
  {{"patient_id": "...", "type": "assign_doctor", "doctor_tier": "t1"}},
  {{"patient_id": "...", "type": "assign_bed"}},
  {{"patient_id": "...", "type": "discharge"}}
]
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        start = content.find('[')
        end = content.rfind(']') + 1

        if start != -1 and end != -1:
            return json.loads(content[start:end])

        return []

    except Exception as e:
        print(f"Decision Error: {e}")
        return []


# ==============================
# Main loop
# ==============================

def run_inference(task_id="hard"):

    # RESET
    try:
        r = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42}
        )
        r.raise_for_status()
        obs = r.json()
    except Exception as e:
        print(f"[ERROR] reset failed: {e}")
        return

    # STRICT FORMAT
    print("[START]")
    print(json.dumps({
        "task_id": task_id,
        "model": MODEL_NAME
    }))

    done = False
    total_score = 0.0
    tick = 0

    while not done:

        actions = get_action_from_llm(obs)

        try:
            r = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": actions}
            )
            r.raise_for_status()
            data = r.json()

            obs = data["observation"]
            reward = data["reward"]
            done = data["done"]

            tick = obs.get("current_tick", tick + 1)
            total_score = obs.get("current_episode_score", total_score)

            print("[STEP]")
            print(json.dumps({
                "tick": tick,
                "reward": round(reward, 2),
                "score": round(total_score, 2)
            }))

        except Exception as e:
            print(f"[ERROR] step failed: {e}")
            break

    print("[END]")
    print(json.dumps({
        "task_id": task_id,
        "final_score": round(total_score, 4)
    }))


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_inference(task)