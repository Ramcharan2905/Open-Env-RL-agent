"""
inference.py — Pure OpenEnv Client for the Hospital Resource Environment.
Handles environment state via HTTP and decisions via the OpenAI Client.

Complies with Meta PyTorch Hackathon requirements:
- Uses OpenAI Python Client for all decisions.
- Communicates with OpenEnv server via API_BASE_URL.
- Reads MODEL_NAME and HF_TOKEN from environment.
"""

import os
import json
import requests
from openai import OpenAI

# ==============================
# Environment Configuration
# ==============================

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://ramcharan2905-hospital-resource-env.hf.space"
)

ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://ramcharan2905-hospital-resource-env.hf.space"
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN   = os.getenv("HF_TOKEN", "")

# ✅ Correct: OpenAI client should NOT point to your env server
client = OpenAI(api_key=HF_TOKEN)


# ==============================
# LLM Decision Function
# ==============================

def get_action_from_llm(observation):
    """Uses LLM to decide next actions"""

    prompt = f"""
You are a Hospital Resource Manager.

Goal:
Maximize survival and throughput while minimizing penalties.

Rules:
- Prioritize ESI-1 and ESI-2 patients first
- Assign correct doctor tier:
  - ESI-1 → t3
  - ESI-2 → t2 or higher
  - Others → any
- Assign beds when needed
- Discharge patients as soon as treatment is complete
- Avoid illegal actions

State:
{json.dumps(observation)}

Output ONLY a JSON array of actions like:
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

        # Extract JSON array safely
        start = content.find('[')
        end = content.rfind(']') + 1

        if start != -1 and end != -1:
            return json.loads(content[start:end])

        return []

    except Exception as e:
        print(f"Decision Error: {e}")
        return []


# ==============================
# Main Inference Loop
# ==============================

def run_inference(task_id: str = "hard"):
    """Runs one full episode and prints required logs"""

    # Reset environment
    try:
        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42}
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        return

    # [START]
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "model": MODEL_NAME
    }))

    done = False
    tick = 0
    total_score = 0.0

    while not done:
        # Get action from LLM
        actions = get_action_from_llm(obs)

        # Step environment
        try:
            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": actions}
            )
            step_resp.raise_for_status()

            step_data = step_resp.json()

            obs    = step_data["observation"]
            reward = step_data["reward"]
            done   = step_data["done"]

            tick = obs.get("current_tick", tick + 1)
            total_score = obs.get("current_episode_score", total_score)

            # [STEP]
            print(json.dumps({
                "type": "STEP",
                "tick": tick,
                "reward": round(reward, 2),
                "score_so_far": round(total_score, 2),
            }))

        except Exception as e:
            print(f"Step failed at tick {tick}: {e}")
            break

    # [END]
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "final_score": round(total_score, 4),
    }))


# ==============================
# Run all tasks
# ==============================

if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task_id=task)