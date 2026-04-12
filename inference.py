"""
inference.py — LLM agent runner for the Hospital Resource Environment.

Log format (stdout only — evaluator parses these lines):
  [START] task=<id> env=hospital model=<model>
  [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<null|msg>
  [END]   success=<bool> steps=<n> score=<float> rewards=<csv>

All debug output goes to stderr so it never pollutes the evaluator's stdout parse.
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

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL   = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
ENV_BASE_URL   = os.getenv("ENV_BASE_URL", "https://ramcharan2905-hospital-resource-env.hf.space").rstrip("/")
MODEL_NAME     = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
# Dashboard requires HF_TOKEN; fall back to OPENAI_API_KEY for local dev.
# Both point to the same Groq key — the OpenAI client works with Groq's API.
OPENAI_API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")

# Startup info goes to STDERR — must NOT appear on stdout before [START]
print(f"[INFO] model={MODEL_NAME} api_base={API_BASE_URL}", file=sys.stderr)
print(f"[INFO] key={'SET (' + OPENAI_API_KEY[:8] + '...)' if OPENAI_API_KEY else 'NOT SET'}", file=sys.stderr)

client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY or "EMPTY")

# ── Grade weights per task ────────────────────────────────────────────────────
TASK_SYSTEM = {
    "easy": (
        "GRADE = 0.55*survival + 0.25*throughput + 0.20*reward_score\n"
        "No cost/death penalties. Discharge patients with high HP to maximize reward."
    ),
    "medium": (
        "GRADE = 0.40*survival + 0.20*throughput + 0.20*wait_score + 0.10*doctor_efficiency + 0.10*reward_score\n"
        "Doctor costs: T1=-3/tick T2=-8/tick T3=-18/tick. Wait penalty active.\n"
        "Use LOWEST valid tier. Assign doctors fast to cut wait penalties."
    ),
    "hard": (
        "GRADE = 0.45*survival + 0.15*throughput + 0.20*wait_score + 0.10*doctor_efficiency + 0.10*reward_score\n"
        "Death=-300, doctor costs T1=-4 T2=-10 T3=-22 per tick, only 5 beds.\n"
        "NEVER let ESI-1/2 wait. One death tanks survival (45% of grade)."
    ),
}

# System prompt explicitly states required JSON field names so the LLM never
# uses wrong keys like "action" instead of "action_type".
SYSTEM_PROMPT = """\
You are an RL agent managing a hospital for 50 ticks. Maximize the final grade score.

DOCTOR TIER MINIMUM per ESI level:
  ESI-1 → tier 3 | ESI-2 → tier 2 | ESI-3/4/5 → tier 1
  Always use the LOWEST valid tier to minimize doctor costs.

STEP REWARD = discharge bonus (hp/100 * scale) - doctor costs - wait penalties - death penalties
Use previous step reward as feedback to improve your next decision.
Always assign beds and discharge whenever legal — frees capacity for new patients.

REQUIRED JSON FIELD NAMES (use these EXACTLY):
  assign_doctor : {"action_type": "assign_doctor", "target_patient_index": <int>, "doctor_tier": <1|2|3>}
  assign_bed    : {"action_type": "assign_bed",    "target_patient_index": <int>}
  discharge     : {"action_type": "discharge",     "target_patient_index": <int>}

Reply with ONLY a raw JSON array. No markdown, no explanation, no extra text.
If nothing to do: []

Example: [{"action_type": "assign_doctor", "target_patient_index": 2, "doctor_tier": 1}, {"action_type": "assign_bed", "target_patient_index": 5}]"""


# ── Server helpers ────────────────────────────────────────────────────────────
def _check_env_server() -> bool:
    for attempt in range(6):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=20)
            r.raise_for_status()
            if r.json().get("status") == "healthy":
                return True
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"[WARN] Health check {attempt+1}/6 failed: {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    print("[ERROR] ENV server unreachable after 6 attempts.", file=sys.stderr)
    return False


def _post_with_retry(url: str, payload: dict, retries: int = 4) -> requests.Response:
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
            wait = 5 * (2 ** attempt)
            print(f"[WARN] Retry {attempt+1}/{retries} in {wait}s...", file=sys.stderr)
            time.sleep(wait)
        except requests.exceptions.HTTPError:
            raise
    raise last_err


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(obs: Dict[str, Any], last_reward: float, last_events: List[str],
                  step_num: int, task_id: str) -> str:
    masks = obs.get("action_masks", {})
    active = [
        {
            "target_patient_index": i,
            "esi_level":  p["esi_level"],
            "phase":      p["phase"],
            "hp":         round(p["current_hp"], 1),
            "wait_ticks": p.get("time_in_state", 0),
            "treat_left": p.get("treatment_ticks_left", 0),
        }
        for i, p in enumerate(obs.get("patients", []))
        if p.get("is_active")
    ]
    resources = {
        "free_t1": obs.get("free_doctors_t1", 0),
        "free_t2": obs.get("free_doctors_t2", 0),
        "free_t3": obs.get("free_doctors_t3", 0),
        "free_beds": obs.get("free_beds", 0),
    }
    grade_info = TASK_SYSTEM.get(task_id, TASK_SYSTEM["hard"])
    feedback = ""
    if step_num > 1:
        quality = "GOOD" if last_reward > 0 else ("PENALTY" if last_reward < 0 else "NEUTRAL")
        events_str = ", ".join(last_events) if last_events else "none"
        feedback = f"\nLAST STEP: reward={last_reward:+.3f} ({quality}), events=[{events_str}]"

    can_doc = masks.get("can_assign_doctor", [])
    can_bed = masks.get("can_assign_bed", [])
    can_dis = masks.get("can_discharge", [])

    return (
        f"{grade_info}{feedback}\n\n"
        f"TICK {obs.get('current_tick','?')}/50 | CUMULATIVE REWARD: {obs.get('current_episode_score', 0):.3f}\n"
        f"RESOURCES: {json.dumps(resources)}\n"
        f"ACTIVE PATIENTS: {json.dumps(active)}\n"
        f"LEGAL ACTIONS:\n"
        f"  assign_doctor indices: {can_doc}\n"
        f"  assign_bed    indices: {can_bed}\n"
        f"  discharge     indices: {can_dis}\n\n"
        f"Output ONLY a JSON array using action_type / target_patient_index / doctor_tier:"
    )


# ── Rule-based fallback ───────────────────────────────────────────────────────
def _rule_based_actions(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    masks    = obs.get("action_masks", {})
    patients = obs.get("patients", [])
    free_t1  = obs.get("free_doctors_t1", 0)
    free_t2  = obs.get("free_doctors_t2", 0)
    free_t3  = obs.get("free_doctors_t3", 0)
    esi_min  = {1: 3, 2: 2, 3: 1, 4: 1, 5: 1}
    actions  = []
    # Prioritize most severe (ESI-1 first)
    for idx in sorted(masks.get("can_assign_doctor", []),
                      key=lambda i: patients[i]["esi_level"] if i < len(patients) else 99):
        if idx >= len(patients):
            continue
        mt = esi_min.get(patients[idx].get("esi_level", 5), 1)
        tier = None
        if mt <= 1 and free_t1 > 0:   tier = 1; free_t1 -= 1
        elif mt <= 2 and free_t2 > 0: tier = 2; free_t2 -= 1
        elif mt <= 3 and free_t3 > 0: tier = 3; free_t3 -= 1
        if tier:
            actions.append({"action_type": "assign_doctor", "target_patient_index": idx, "doctor_tier": tier})
    for idx in masks.get("can_assign_bed", []):
        actions.append({"action_type": "assign_bed", "target_patient_index": idx})
    for idx in masks.get("can_discharge", []):
        actions.append({"action_type": "discharge", "target_patient_index": idx})
    return actions


def _extract_json_array(text: str) -> List[Any]:
    """Extract the first JSON array, stripping markdown code fences if present."""
    text = text.strip()
    if "```" in text:
        s = text.find("```")
        e = text.rfind("```")
        if s != e:
            inner = text[s + 3:e]
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        return []
    try:
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _normalize_action(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize field-name variants the LLM might use.
    Maps "action"->"action_type", "patient"->"target_patient_index", "t2"->2, etc.
    """
    n: Dict[str, Any] = {}
    for k in ("action_type", "action", "type"):
        if k in a:
            n["action_type"] = a[k]; break
    if "action_type" not in n:
        return {}
    for k in ("target_patient_index", "patient_index", "patient"):
        if k in a and isinstance(a[k], int):
            n["target_patient_index"] = a[k]; break
    if "target_patient_index" not in n:
        return {}
    for k in ("doctor_tier", "doctor", "tier"):
        if k in a:
            tier = a[k]
            if isinstance(tier, str) and tier.startswith("t"):
                try: tier = int(tier[1:])
                except ValueError: pass
            if isinstance(tier, int):
                n["doctor_tier"] = tier
            break
    return n


def _validate_actions(raw: List[Any], obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize and validate LLM actions against current legal action masks."""
    masks      = obs.get("action_masks", {})
    can_doctor = set(masks.get("can_assign_doctor", []))
    can_bed    = set(masks.get("can_assign_bed", []))
    can_dis    = set(masks.get("can_discharge", []))
    valid = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        a = _normalize_action(item)
        if not a:
            continue
        idx = a.get("target_patient_index")
        at  = a.get("action_type", "")
        if not isinstance(idx, int):
            continue
        if at == "assign_doctor":
            if idx not in can_doctor or a.get("doctor_tier") not in (1, 2, 3):
                continue
        elif at == "assign_bed":
            if idx not in can_bed:
                continue
        elif at == "discharge":
            if idx not in can_dis:
                continue
        else:
            continue
        valid.append(a)
    return valid


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_inference(task_id: str) -> None:
    rewards: List[float] = []

    # [START] line — stdout, exactly this format
    print(f"[START] task={task_id} env=hospital model={MODEL_NAME}", flush=True)

    if not _check_env_server():
        print(f"[END] success=false steps=0 score=0.000001 rewards=", flush=True)
        return

    try:
        obs = _post_with_retry(f"{ENV_BASE_URL}/reset", {"task_id": task_id, "seed": 42}).json()
    except Exception as e:
        print(f"[ERROR] /reset failed: {e}", file=sys.stderr)
        print(f"[END] success=false steps=0 score=0.000001 rewards=", flush=True)
        return

    done, step_num = False, 0
    last_reward, last_events = 0.0, []
    final_score = 0.0
    use_llm = bool(OPENAI_API_KEY and OPENAI_API_KEY not in ("", "your_groq_or_hf_token_here", "EMPTY"))

    while not done:
        step_num += 1
        prompt = _build_prompt(obs, last_reward, last_events, step_num, task_id)

        # Default: rule-based. Override with LLM when available and valid.
        actions = _rule_based_actions(obs)

        if use_llm:
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=512,
                )
                content = (resp.choices[0].message.content or "").strip()
                print(f"[DBG] step={step_num} LLM raw: {content[:300]}", file=sys.stderr)
                valid = _validate_actions(_extract_json_array(content), obs)
                if valid:
                    actions = valid
                    print(f"[DBG] step={step_num} LLM gave {len(valid)} valid actions", file=sys.stderr)
                else:
                    print(f"[DBG] step={step_num} LLM gave no valid actions — rule-based fallback", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] LLM error at step {step_num}: {e} — rule-based fallback", file=sys.stderr)

        try:
            data    = _post_with_retry(f"{ENV_BASE_URL}/step", {"action": actions}).json()
            obs     = data["observation"]
            reward  = float(data.get("reward", 0.0))
            done    = bool(data.get("done", False))
            info    = data.get("info", {})
            rewards.append(reward)
            last_reward = reward
            last_events = info.get("last_step_events", [])

            # [STEP] line — stdout, exactly this format
            print(
                f"[STEP] step={step_num} action={json.dumps(actions)} "
                f"reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True,
            )

            if done:
                final_score = float(info.get("episode_grade", 0.0))
                break

        except Exception as e:
            print(f"[STEP] step={step_num} action=null reward=0.00 done=true error={e}", flush=True)
            break

    # Fallback grade if episode_grade was absent from info
    if final_score == 0.0:
        try:
            final_score = requests.get(
                f"{ENV_BASE_URL}/grade/{task_id}", timeout=30
            ).json().get("score", 0.0)
        except Exception:
            final_score = max(1e-6, min(sum(rewards) / max(len(rewards), 1), 1.0 - 1e-6))

    # [END] line — stdout, exactly this format
    print(
        f"[END] success=true steps={len(rewards)} score={final_score:.6f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


if __name__ == "__main__":
    for task in ("easy", "medium", "hard"):
        run_inference(task)
