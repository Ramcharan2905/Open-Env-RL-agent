"""Simple baseline agent that samples a random batch of currently legal actions (Hackathon Compliant)."""

import json
import random
from typing import List

from environment import HospitalEnv
from models import ActionType, HospitalAction

def run_random_agent(task_id: str) -> None:
    """Run one full episode for one task using a random valid-action batch policy."""

    # 1. MANDATORY START LOG
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "model": "random_baseline"
    }))

    env = HospitalEnv(seed=42, task_id=task_id)
    obs = env.reset()
    done = False

    while not done:
        action_plan: List[HospitalAction] = []
        
        # Local resource trackers so the random agent doesn't over-allocate in a single tick
        free_doctors = {
            1: obs.free_doctors_t1,
            2: obs.free_doctors_t2,
            3: obs.free_doctors_t3,
        }
        free_beds = obs.free_beds

        # --- Randomly Assign Doctors ---
        assignable_doctors = obs.action_masks.can_assign_doctor.copy()
        random.shuffle(assignable_doctors) # Shuffle to randomly pick who gets treated first
        
        for patient_idx in assignable_doctors:
            patient = obs.patients[patient_idx]
            min_doc_required = env.config.esi_rules[patient.esi_level]["min_doc"]
            
            # Find the lowest available doctor tier that meets the requirement
            assigned_doc = None
            for doc_tier in range(min_doc_required, 4):
                if free_doctors[doc_tier] > 0:
                    free_doctors[doc_tier] -= 1
                    assigned_doc = doc_tier
                    break
                    
            if assigned_doc is not None:
                action_plan.append(HospitalAction(
                    action_type=ActionType.ASSIGN_DOCTOR,
                    target_patient_index=patient_idx,
                    doctor_tier=assigned_doc
                ))

        # --- Randomly Assign Beds ---
        assignable_beds = obs.action_masks.can_assign_bed.copy()
        for patient_idx in assignable_beds:
            if free_beds > 0:
                free_beds -= 1
                action_plan.append(HospitalAction(
                    action_type=ActionType.ASSIGN_BED,
                    target_patient_index=patient_idx
                ))

        # --- Randomly Discharge Patients ---
        assignable_discharges = obs.action_masks.can_discharge.copy()
        for patient_idx in assignable_discharges:
            action_plan.append(HospitalAction(
                action_type=ActionType.DISCHARGE,
                target_patient_index=patient_idx
            ))

        # Step the environment
        obs, reward, done, info = env.step(action_plan)

        # 2. MANDATORY STEP LOG
        print(json.dumps({
            "type": "STEP",
            "tick": obs.current_tick,
            "reward": round(reward, 2),
            "actions": [a.to_dict() for a in action_plan]
        }))

    # 3. MANDATORY END LOG
    final_score = info.get("episode_grade", obs.current_episode_score)
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "score": round(final_score, 4)
    }))

def main():
    """Run the random agent against all tasks."""
    tasks = ["easy", "medium", "hard"]
    for task_id in tasks:
        run_random_agent(task_id)

if __name__ == "__main__":
    main()