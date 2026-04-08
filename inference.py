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

# Required Hackathon Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o") # Evaluator's model choice
HF_TOKEN     = os.getenv("HF_TOKEN", "your_huggingface_token")

# Initialize OpenAI Client pointing to the hub or orchestrator
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_action_from_llm(observation):
    """Encapsulates the decision logic as an OpenAI chat completion call."""
    
    prompt = f"""
    You are a Hospital Resource Manager. Given the following hospital state, output a list of actions in JSON format.
    State: {json.dumps(observation)}
    
    Output exactly one JSON array of actions using this schema:
    [
      {{"patient_id": "...", "type": "assign_doctor", "doctor_tier": "t1"}},
      {{"patient_id": "...", "type": "assign_bed"}},
      {{"patient_id": "...", "type": "discharge"}}
    ]
    Refuse any other text. Output ONLY the JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"} if "gpt-4" in MODEL_NAME else None
        )
        content = response.choices[0].message.content
        # Attempt to find the JSON block if the model was chatty
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != -1:
            return json.loads(content[start:end])
        return [] # Fallback to no-op
    except Exception as e:
        print(f"Decision Error: {e}")
        return []

def run_inference(task_id: str = "hard"):
    """Run one full episode via the OpenEnv API and emit hackathon-required logs."""

    # 1. Reset Environment via API
    try:
        reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": 42})
        reset_resp.raise_for_status()
        obs = reset_resp.json()
    except Exception as e:
        print(f"Failed to reset environment at {API_BASE_URL}: {e}")
        return

    # [START] log
    print(json.dumps({"type": "START", "task_id": task_id, "model": MODEL_NAME}))

    done = False
    tick = 0
    total_score = 0.0

    while not done:
        # 2. Get Decision via OpenAI Client
        actions = get_action_from_llm(obs)

        # 3. Step Environment via API
        try:
            step_resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": actions})
            step_resp.raise_for_status()
            step_data = step_resp.json()
            
            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            info = step_data["info"]

            tick = obs.get("current_tick", tick + 1)
            total_score = obs.get("current_episode_score", total_score)

            # [STEP] log
            print(json.dumps({
                "type":         "STEP",
                "tick":         tick,
                "reward":       round(reward, 2),
                "score_so_far": round(total_score, 2),
            }))
        except Exception as e:
            print(f"Step failed at tick {tick}: {e}")
            break

    # [END] log
    final_score = round(total_score, 4)
    print(json.dumps({
        "type":         "END",
        "task_id":      task_id,
        "final_score":  final_score,
    }))

if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task_id=task)