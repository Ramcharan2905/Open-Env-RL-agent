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
import sys
from typing import Any, List, Dict

import requests
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ── Read config ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://ramcharan2905-hospital-resource-env.hf.space",
).rstrip("/")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── FIX 1: Detect placeholder / missing keys ─────────────────────────────────
_PLACEHOLDER_KEYS = {"your_openai_key", "your-openai-key", "", "EMPTY", "sk-..."}

def _key_is_valid(key: str) -> bool:
    if not key:
        return False
    stripped = key.strip()
    if stripped in _PLACEHOLDER_KEYS:
        return False
    if not stripped.startswith("sk-"):
        return False
    return True

_KEY_VALID = _key_is_valid(OPENAI_API_KEY)

print("MODEL:", MODEL_NAME)
print("KEY:", OPENAI_API_KEY[:12] + "..." if _KEY_VALID else "NOT SET / INVALID")

if not _KEY_VALID:
    print(
        "[WARN] OPENAI_API_KEY is missing or a placeholder. "
        "Set a real key in your .env file. Falling back to rule-based agent.",
        file=sys.stderr,
    )

client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY if _KEY_VALID else "EMPTY")


# ── FIX 2: Verify HF Space is reachable before starting ─────────────────────

def _check_env_server() -> bool:
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "healthy":
            return True
        print(
            f"[WARN] ENV server responded but status='{data.get('status')}' "
            f"(expected 'healthy'). URL: {ENV_BASE_URL}",
            file=sys.stderr,
        )
        return False
    except requests.exceptions.ConnectionError:
        print(
            f"[ERROR] Cannot reach ENV server at {ENV_BASE_URL}.\n"
            "  → Check the HuggingFace Space is running.\n"
            "  → Verify ENV_BASE_URL in your .env matches the Space URL.",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(f"[WARN] ENV health check failed: {e}", file=sys.stderr)
        return False


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _build_prompt(observation: Dict[str, Any]) -> str:
    masks = observation.get("action_masks", {})
    can_assign_doctor = masks.get("can_assign_doctor", [])
    can_assign_bed = masks.get("can_assign_bed", [])
    can_discharge = masks.get("can_discharge", [])

    active_patients = []
    for i, p in enumerate(observation.get("patients", [])):
        if not p.get("is_active"):
            continue
        active_patients.append({
            "slot_index": i,
            "esi_level": p["esi_level"],
            "phase": p["phase"],
            "hp": round(p["current_hp"], 1),
        })

    resources = {
        "free_doctors_t1": observation.get("free_doctors_t1", 0),
        "free_doctors_t2": observation.get("free_doctors_t2", 0),
        "free_doctors_t3": observation.get("free_doctors_t3", 0),
        "free_beds": observation.get("free_beds", 0),
    }

    esi_min_doc = {1: 3, 2: 2, 3: 1, 4: 1, 5: 1}

    prompt = f"""You are a hospital resource manager AI. Your job is to assign doctors and beds to patients each tick.

RESOURCES: {json.dumps(resources)}

ACTIVE PATIENTS: {json.dumps(active_patients)}

LEGAL ACTIONS THIS TICK:
- Can assign doctor to slot indices: {can_assign_doctor}
- Can assign bed to slot indices:    {can_assign_bed}  
- Can discharge from slot indices:   {can_discharge}

RULES:
- Only act on slot indices listed above — all others are illegal.
- For assign_doctor: choose doctor_tier (1/2/3) >= patient's ESI minimum: {json.dumps(esi_min_doc)}
- Only use a doctor tier if free count > 0 (check resources above).
- Prioritize ESI-1 (critical) first, then ESI-2, then others.
- Always assign beds and discharge when legal.

OUTPUT FORMAT — respond with ONLY a JSON array, no explanation:
[
  {{"action_type": "assign_doctor", "target_patient_index": <int>, "doctor_tier": <1|2|3>}},
  {{"action_type": "assign_bed",    "target_patient_index": <int>}},
  {{"action_type": "discharge",     "target_patient_index": <int>}}
]

If nothing legal to do, return: []
"""
    return prompt


def get_action_from_llm(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    rule_actions = _rule_based_actions(observation)

    if not _KEY_VALID:
        return rule_actions

    try:
        prompt = _build_prompt(observation)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        llm_actions = _extract_json_array(content.strip())

        valid_actions = []
        for a in llm_actions:
            if isinstance(a, dict) and "action_type" in a and "target_patient_index" in a:
                valid_actions.append(a)

        return valid_actions if valid_actions else rule_actions

    except Exception as e:
        print(f"[WARN] LLM call failed: {e} — using rule-based agent", file=sys.stderr)
        return rule_actions


def _rule_based_actions(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    masks = observation.get("action_masks", {})
    can_assign_doctor = masks.get("can_assign_doctor", [])
    can_assign_bed = masks.get("can_assign_bed", [])
    can_discharge = masks.get("can_discharge", [])
    patients = observation.get("patients", [])

    free_t1 = observation.get("free_doctors_t1", 0)
    free_t2 = observation.get("free_doctors_t2", 0)
    free_t3 = observation.get("free_doctors_t3", 0)

    esi_min_doc = {1: 3, 2: 2, 3: 1, 4: 1, 5: 1}
    actions = []

    assignable = sorted(
        can_assign_doctor,
        key=lambda i: patients[i]["esi_level"] if i < len(patients) else 99
    )

    for idx in assignable:
        if idx >= len(patients):
            continue
        esi = patients[idx].get("esi_level", 5)
        min_tier = esi_min_doc.get(esi, 1)

        assigned_tier = None
        if min_tier <= 1 and free_t1 > 0:
            assigned_tier = 1
            free_t1 -= 1
        elif min_tier <= 2 and free_t2 > 0:
            assigned_tier = 2
            free_t2 -= 1
        elif min_tier <= 3 and free_t3 > 0:
            assigned_tier = 3
            free_t3 -= 1

        if assigned_tier is not None:
            actions.append({
                "action_type": "assign_doctor",
                "target_patient_index": idx,
                "doctor_tier": assigned_tier,
            })

    for idx in can_assign_bed:
        actions.append({"action_type": "assign_bed", "target_patient_index": idx})

    for idx in can_discharge:
        actions.append({"action_type": "discharge", "target_patient_index": idx})

    return actions


def run_inference(task_id: str) -> None:
    rewards = []

    print(f"[START] task={task_id} env=hospital model={MODEL_NAME}")

    # FIX 2: Confirm the HF Space is alive before attempting to reset/step
    if not _check_env_server():
        print(f"[END] success=false steps=0 score=0.000001 rewards=")
        return

    try:
        reset_resp = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": 42},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()
    except Exception as e:
        print(f"[END] success=false steps=0 score=0.000001 rewards=")
        print(f"[ERROR] /reset failed: {e}", file=sys.stderr)
        return

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
                f"[STEP] step={step_num} action={json.dumps(actions)} "
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
    raw_score = sum(rewards) / max(steps, 1)
    EPS = 1e-6
    score = max(EPS, min(raw_score, 1.0 - EPS))

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success=true steps={steps} "
        f"score={score:.6f} rewards={rewards_str}"
    )


if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task)