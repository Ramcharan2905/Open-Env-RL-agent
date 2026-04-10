"""
inference.py — Root-level baseline runner for the Hospital Resource Environment.
"""

import json
import os
import sys
import time
from typing import Any, List, Dict

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://ramcharan2905-hospital-resource-env.hf.space").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Validate key ─────────────────────────────────────────────────────────────
_PLACEHOLDERS = {"your_openai_key", "your-openai-key", "", "EMPTY", "sk-..."}

def _key_is_valid(key: str) -> bool:
    s = key.strip()
    return bool(s) and s not in _PLACEHOLDERS and s.startswith("sk-")

_KEY_VALID = _key_is_valid(OPENAI_API_KEY)

print("MODEL:", MODEL_NAME)
print("KEY:", OPENAI_API_KEY[:12] + "..." if _KEY_VALID else "NOT SET / INVALID")

if not _KEY_VALID:
    print("[WARN] OPENAI_API_KEY is missing or a placeholder. Falling back to rule-based agent.", file=sys.stderr)

client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY if _KEY_VALID else "EMPTY")


# ── HF Space connectivity ─────────────────────────────────────────────────────
def _check_env_server() -> bool:
    """Ping /health — wake the HF Space if it's sleeping (up to 90s)."""
    for attempt in range(6):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=20)
            r.raise_for_status()
            if r.json().get("status") == "healthy":
                return True
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"[WARN] Health check attempt {attempt+1}/6 failed: {e}. "
                  f"Waiting {wait}s for HF Space to wake...", file=sys.stderr)
            time.sleep(wait)
    print(f"[ERROR] ENV server at {ENV_BASE_URL} unreachable after 6 attempts.", file=sys.stderr)
    return False


def _post_with_retry(url: str, payload: dict, retries: int = 4) -> requests.Response:
    """POST with exponential backoff — handles HF Space mid-episode cold drops."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            last_err = e
            wait = 5 * (2 ** attempt)   # 5, 10, 20, 40s
            print(f"[WARN] Connection dropped (attempt {attempt+1}/{retries}), "
                  f"retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except requests.exceptions.HTTPError:
            raise   # don't retry 4xx/5xx
    raise last_err


# ── LLM prompt / action ───────────────────────────────────────────────────────
def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _build_prompt(observation: Dict[str, Any]) -> str:
    masks = observation.get("action_masks", {})
    can_assign_doctor = masks.get("can_assign_doctor", [])
    can_assign_bed    = masks.get("can_assign_bed", [])
    can_discharge     = masks.get("can_discharge", [])

    active_patients = [
        {"slot_index": i, "esi_level": p["esi_level"], "phase": p["phase"], "hp": round(p["current_hp"], 1)}
        for i, p in enumerate(observation.get("patients", []))
        if p.get("is_active")
    ]

    resources = {
        "free_doctors_t1": observation.get("free_doctors_t1", 0),
        "free_doctors_t2": observation.get("free_doctors_t2", 0),
        "free_doctors_t3": observation.get("free_doctors_t3", 0),
        "free_beds":       observation.get("free_beds", 0),
    }

    esi_min_doc = {1: 3, 2: 2, 3: 1, 4: 1, 5: 1}

    return f"""You are a hospital resource manager AI. Assign doctors and beds to patients each tick.

RESOURCES: {json.dumps(resources)}
ACTIVE PATIENTS: {json.dumps(active_patients)}

LEGAL ACTIONS THIS TICK:
- assign_doctor slot indices: {can_assign_doctor}
- assign_bed slot indices:    {can_assign_bed}
- discharge slot indices:     {can_discharge}

RULES:
- Only use slot indices listed above.
- assign_doctor: doctor_tier (1/2/3) >= ESI minimum {json.dumps(esi_min_doc)}, only if free count > 0.
- Prioritize ESI-1 first, then ESI-2, then others.
- Always assign beds and discharge when legal.

Respond with ONLY a JSON array:
[
  {{"action_type": "assign_doctor", "target_patient_index": <int>, "doctor_tier": <1|2|3>}},
  {{"action_type": "assign_bed",    "target_patient_index": <int>}},
  {{"action_type": "discharge",     "target_patient_index": <int>}}
]
If nothing to do, return: []
"""


def _rule_based_actions(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    masks = observation.get("action_masks", {})
    can_assign_doctor = masks.get("can_assign_doctor", [])
    can_assign_bed    = masks.get("can_assign_bed", [])
    can_discharge     = masks.get("can_discharge", [])
    patients = observation.get("patients", [])

    free_t1 = observation.get("free_doctors_t1", 0)
    free_t2 = observation.get("free_doctors_t2", 0)
    free_t3 = observation.get("free_doctors_t3", 0)
    esi_min_doc = {1: 3, 2: 2, 3: 1, 4: 1, 5: 1}
    actions = []

    for idx in sorted(can_assign_doctor, key=lambda i: patients[i]["esi_level"] if i < len(patients) else 99):
        if idx >= len(patients):
            continue
        min_tier = esi_min_doc.get(patients[idx].get("esi_level", 5), 1)
        tier = None
        if min_tier <= 1 and free_t1 > 0:  tier = 1; free_t1 -= 1
        elif min_tier <= 2 and free_t2 > 0: tier = 2; free_t2 -= 1
        elif min_tier <= 3 and free_t3 > 0: tier = 3; free_t3 -= 1
        if tier:
            actions.append({"action_type": "assign_doctor", "target_patient_index": idx, "doctor_tier": tier})

    for idx in can_assign_bed:
        actions.append({"action_type": "assign_bed", "target_patient_index": idx})
    for idx in can_discharge:
        actions.append({"action_type": "discharge", "target_patient_index": idx})

    return actions


def get_action_from_llm(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
    rule_actions = _rule_based_actions(observation)
    if not _KEY_VALID:
        return rule_actions
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": _build_prompt(observation)}],
            temperature=0,
            max_tokens=512,
        )
        llm_actions = _extract_json_array((response.choices[0].message.content or "").strip())
        valid = [a for a in llm_actions if isinstance(a, dict) and "action_type" in a and "target_patient_index" in a]
        return valid if valid else rule_actions
    except Exception as e:
        print(f"[WARN] LLM call failed: {e} — using rule-based agent", file=sys.stderr)
        return rule_actions


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_inference(task_id: str) -> None:
    rewards = []
    print(f"[START] task={task_id} env=hospital model={MODEL_NAME}")

    if not _check_env_server():
        print(f"[END] success=false steps=0 score=0.000001 rewards=")
        return

    try:
        obs = _post_with_retry(f"{ENV_BASE_URL}/reset", {"task_id": task_id, "seed": 42}).json()
    except Exception as e:
        print(f"[END] success=false steps=0 score=0.000001 rewards=")
        print(f"[ERROR] /reset failed: {e}", file=sys.stderr)
        return

    done, step_num = False, 0
    while not done:
        step_num += 1
        actions = get_action_from_llm(obs)
        try:
            data = _post_with_retry(f"{ENV_BASE_URL}/step", {"action": actions}).json()
            obs    = data["observation"]
            reward = float(data.get("reward", 0.0))
            done   = bool(data.get("done", False))
            rewards.append(reward)
            print(f"[STEP] step={step_num} action={json.dumps(actions)} reward={reward:.2f} done={str(done).lower()} error=null")
            if done:
                break
        except Exception as e:
            print(f"[STEP] step={step_num} action=null reward=0.00 done=true error={e}")
            break

    steps = len(rewards)
    score = max(1e-6, min(sum(rewards) / max(steps, 1), 1.0 - 1e-6))
    print(f"[END] success=true steps={steps} score={score:.6f} rewards={','.join(f'{r:.2f}' for r in rewards)}")


if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task)