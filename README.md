---
title: Hospital Resource Environment
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server.py
pinned: false
---

# Hospital Resource Environment

A real-world hospital emergency department simulator built for the **Meta PyTorch Hackathon** using the [OpenEnv](https://huggingface.co/openenv) standard. The agent acts as a hospital operations manager — triaging patients, assigning doctors by skill tier, managing bed allocation, and deciding when to discharge — all under stochastic patient arrivals and constrained hospital resources.

This is a **genuine logistics and scheduling problem**, not a toy game. The environment is designed to challenge agents on:

- Prioritizing high-acuity patients (ESI-1 / ESI-2) when resources are scarce
- Reasoning jointly about doctor scarcity and bed bottlenecks
- Handling delayed outcomes (doctors free after treatment, not immediately)
- Adapting strategy across three difficulty tiers with fundamentally different resource regimes

---

## 🚀 Live Deployment

Hugging Face Space (Production Environment):

```
https://ramcharan2905-hospital-resource-env.hf.space
```

Quick health check:

```bash
curl https://ramcharan2905-hospital-resource-env.hf.space/health
```

Expected response:

```json
{"status": "healthy", "service": "hospital-resource-env"}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   inference.py                      │
│        OpenAI-client Agent (makes decisions)        │
└────────────────────┬────────────────────────────────┘
                     │ HTTP (POST /reset, POST /step)
┌────────────────────▼────────────────────────────────┐
│                    server.py                        │
│         FastAPI OpenEnv Server (port 7860)          │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                 environment.py                      │
│         Hospital Simulator (core physics)           │
└─────────────────────────────────────────────────────┘
```

The **server** hosts the environment over HTTP. The **inference script** is a pure API client — it reads the observation, sends it to an LLM via the OpenAI client, parses the response back into hospital actions, and steps the environment.

---

## HTTP API

The server exposes a standard OpenEnv-compatible REST API:

| Method | Endpoint  | Description                                          |
| :----- | :-------- | :--------------------------------------------------- |
| GET    | `/health` | Health check — returns `{"status": "healthy"}`       |
| GET    | `/state`  | Full internal environment state                      |
| POST   | `/reset`  | Start a new episode (accepts `task_id` and `seed`)   |
| POST   | `/step`   | Apply one or more actions, returns obs/reward/done   |

### `/reset` Request Body
```json
{ "task_id": "easy", "seed": 42 }
```

### `/step` Request Body
```json
{
  "action": [
    { "action_type": "assign_doctor", "target_patient_index": 0, "doctor_tier": 2 },
    { "action_type": "assign_bed",    "target_patient_index": 3 },
    { "action_type": "discharge",     "target_patient_index": 5 }
  ]
}
```

The `action` field accepts `null` (no-op), a single action object, or a list of actions.

---

## Observation Space

Each observation tick contains:

| Field                   | Type         | Description                               |
| :---------------------- | :----------- | :---------------------------------------- |
| `current_tick`          | `int`        | Current simulation minute (0–49)          |
| `free_doctors_t1`       | `int`        | Available Tier-1 doctors (general)        |
| `free_doctors_t2`       | `int`        | Available Tier-2 doctors (specialist)     |
| `free_doctors_t3`       | `int`        | Available Tier-3 doctors (senior)         |
| `free_beds`             | `int`        | Available inpatient beds                  |
| `current_episode_score` | `float`      | Cumulative reward so far this episode     |
| `patients`              | `list`       | Up to 50 patient slots (see below)        |
| `action_masks`          | `object`     | Per-slot legal action bitmasks            |

Each patient slot includes: `patient_id`, `esi_level` (1–5), `phase`, `hp`, `wealth_multiplier`, `time_in_phase`, `treatment_ticks_remaining`.

---

## Action Space

Each action targets a specific patient by `target_patient_index` (the slot index in the patients array):

| Action Type      | Required Extra Field | Description                          |
| :--------------- | :------------------- | :----------------------------------- |
| `no_op`          | —                    | Do nothing for this patient this tick |
| `assign_doctor`  | `doctor_tier: 1\|2\|3` (integer) | Assign a doctor to begin treatment |
| `assign_bed`     | —                    | Move patient from waiting to a bed   |
| `discharge`      | —                    | Discharge a fully-treated patient    |

**Doctor Tier Rules**: Higher ESI severity requires a higher-tier doctor. ESI-1 requires Tier-3; ESI-2 requires Tier-2+; ESI-3/4/5 can use any tier. Assigning an under-qualified doctor is an illegal action and incurs a penalty.

---

## Task Difficulties

Three tasks are available, each with a different resource configuration and reward regime:

### Easy
A forgiving environment to learn the basic patient flow (Triage → Bed → Treat → Discharge):
- High doctor and bed availability
- No wait penalties
- No doctor costs
- No wealth multiplier

### Medium
Resource-constrained — the agent must manage scarcity and cost:
- Fewer doctors (requiring careful tier selection) and fewer beds
- Doctor costs enabled — over-using Tier-3 doctors is expensive
- Higher patient arrival rate
- No wealth multiplier

### Hard (Full Hospital Control)
The complete, unforgiving environment with all mechanics active:
- **Surge arrivals** — high patient volume at unpredictable rates
- **Doctor costs** — every tier assignment has a cost
- **Wealth multiplier** — wealthy patients generate more reward, creating triage trade-offs
- **Wait penalties** — patients deteriorate if left waiting too long
- **Post-treatment penalties** — beds must be freed promptly after treatment
- **Tight bed capacity** — bed bottlenecks cascade quickly under surge

---

## Reward Design

The reward function provides **dense, per-tick feedback** rather than a single end-of-episode score:

| Event                          | Reward Signal |
| :----------------------------- | :------------ |
| Patient successfully discharged | `+` positive  |
| Patient death                  | `−` penalty   |
| Patient waiting too long        | `−` penalty (Hard/Medium only) |
| Post-treatment bed not freed   | `−` penalty (Hard only) |
| Doctor cost per assignment      | `−` deduction (Medium/Hard) |
| Illegal action taken           | `−` penalty   |

Dense rewards allow the agent to distinguish "somewhat bad" from "catastrophically bad" within a single episode rather than only learning from final outcomes.

---

## Grading

The grader in [`tasks.py`](./tasks.py) returns a normalized score from `0.0` to `1.0`:

```
score = 0.55 × survival_rate + 0.25 × throughput_rate + 0.20 × reward_component
```

- **Survival Rate**: fraction of patients who did not die
- **Throughput Rate**: fraction of patients successfully discharged
- **Reward Component**: cumulative reward normalized against a task-specific baseline

The grader is deterministic — the same agent on the same seed always produces the same score.

---

## Files

| File                     | Purpose                                                   |
| :----------------------- | :-------------------------------------------------------- |
| `environment.py`         | Core hospital simulator — all physics and state transitions |
| `models.py`              | Typed Pydantic models for observations, actions, state    |
| `patient_generator.py`   | Stochastic patient arrival logic                          |
| `tasks.py`               | Task configurations (Easy/Medium/Hard) and grader         |
| `server.py`              | FastAPI OpenEnv server (exposes `/reset`, `/step`, etc.)  |
| `inference.py`           | Pure OpenAI-client agent — connects to server via HTTP    |
| `random_ai_agent.py`     | Random baseline agent for sanity-checking the environment |
| `train_model.py`         | PPO training pipeline (off-line, not needed for serving)  |
| `best_hospital_model.pth`| Pre-trained RL agent weights (Actor-Critic, curriculum trained) |
| `openenv.yaml`           | OpenEnv metadata — required for HF Space evaluation       |
| `Dockerfile`             | Container definition for Hugging Face Spaces deployment   |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Configure environment variables

```bash
cp .env.example .env
```

Example `.env`:

```env
PORT=7860
API_BASE_URL=https://ramcharan2905-hospital-resource-env.hf.space
ENV_BASE_URL=https://ramcharan2905-hospital-resource-env.hf.space
HF_TOKEN=your_token_here
MODEL_NAME=gpt-4o
```

---

### 3. Run locally (optional)

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

---

### 4. Run inference agent

```bash
python inference.py
```

---

### 5. Validate environment

```bash
python validate_submission.py
```

---

## Docker

Build and run:

```bash
docker build -t hospital-resource-env .
docker run -p 7860:7860 hospital-resource-env
```

Test:

```bash
curl http://localhost:7860/health
```

---

## Hugging Face Spaces Deployment

```bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/hospital-resource-env
git push hf main
```

> **Important**: Use **Docker SDK**. The app runs on port **7860** as required by Hugging Face.

---

## Final Notes

- This environment is fully OpenEnv compliant
- Supports deterministic evaluation via seeds
- Includes both RL and LLM-based agents
- Designed for real-world logistics complexity rather than synthetic benchmarks