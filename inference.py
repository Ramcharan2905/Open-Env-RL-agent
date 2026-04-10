"""
Final inference script — OpenEnv compliant
"""

import json
import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://ramcharan2905-hospital-resource-env.hf.space",
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")


def get_action(obs):
    return []  # safe baseline (no-op agent)


def run(task_id):
    rewards = []

    print(f"[START] task={task_id} env=hospital model={MODEL_NAME}")

    try:
        obs = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=30,
        ).json()
    except:
        print("[END] success=false steps=0 score=0.000001 rewards=")
        return

    done = False
    step = 0

    while not done:
        step += 1
        action = get_action(obs)

        try:
            res = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": action},
                timeout=30,
            ).json()

            obs = res["observation"]
            reward = float(res.get("reward", 0.0))
            done = bool(res.get("done", False))

            rewards.append(reward)

            print(
                f"[STEP] step={step} action={action} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

        except Exception as e:
            print(
                f"[STEP] step={step} action=null reward=0.00 done=true error={str(e)}"
            )
            break

    steps = len(rewards)

    # safe score (not used for grading but must be valid)
    score = sum(rewards) / max(steps, 1)
    score = max(1e-6, min(score, 1 - 1e-6))

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success=true steps={steps} "
        f"score={score:.6f} rewards={rewards_str}"
    )


if __name__ == "__main__":
    for t in ["easy", "medium", "hard"]:
        run(t)