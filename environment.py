"""Hospital simulation environment.

This file contains the main runtime logic of the project. Each call to
`step(...)` represents one minute of simulated time and may contain a batch
of actions for different patients.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from models import (
    PatientPhase, ActionType, PatientVector, ActionMasks, 
    HospitalObservation, HospitalAction, StepInfo, EnvConfig,
    OpenEnvAction, OpenEnvObservation, OpenEnvReward, OpenEnvState, OpenEnvPatient,
)
from patient_generator import PatientGenerator
from tasks import TaskDefinition, build_task_definition, grade_episode

class HospitalEnv:
    """Stateful hospital simulator with finite doctors, beds, and patient slots."""

    def __init__(
        self,
        seed: int = 42,
        max_ticks: Optional[int] = None,
        config: Optional[EnvConfig] = None,
        task_id: str = "hard",
    ):
        # `base_config` is the generic hospital setup. `build_task_definition`
        # derives a task-specific copy of that config so easy / medium / hard
        # all share one engine but differ in reward shaping and capacity.
        self.base_config = config or EnvConfig()
        self.task: TaskDefinition = build_task_definition(task_id, self.base_config)
        self.config = self.task.config
        self.task_id = self.task.task_id
        # `max_ticks` caps the episode length even if patients are still active.
        self.max_ticks = max_ticks or self.task.max_ticks
        # This map tracks which doctor tier is currently treating each patient.
        # We need it later to bill doctor cost per tick and to free the correct
        # doctor tier when treatment ends.
        self._active_doc_assignments: Dict[str, int] = {}
        self.seed(seed)
        self.reset()

    def seed(self, seed: int) -> None:
        """Reset all stochastic behavior to a reproducible random seed."""

        # Every random sample in the environment goes through this RNG, so a
        # fixed seed reproduces the same arrivals and patient characteristics.
        self.np_random = np.random.RandomState(seed)
        self.generator = PatientGenerator(self.np_random)

    def reset(self) -> HospitalObservation:
        """Start a fresh episode and return the initial observation."""

        # Episode-level counters track how the whole run is going.
        self.current_tick = 0
        self.current_score = 0.0
        self.next_patient_id = 0
        self.total_deaths = 0
        self.total_discharged = 0
        self.total_patients_seen = 0
        self.illegal_actions = 0
        self.total_wait_time = 0
        self.completed_waits = 0
        self.bed_utilization_sum = 0.0
        self.doctor_utilization_sum = 0.0
        self.t1_utilization_sum = 0.0
        self.t2_utilization_sum = 0.0
        self.t3_utilization_sum = 0.0
        self.last_step_events: List[str] = []

        # Resource pools start fully available at the beginning of an episode.
        self.free_t1 = self.config.total_t1_docs
        self.free_t2 = self.config.total_t2_docs
        self.free_t3 = self.config.total_t3_docs
        self.free_beds = self.config.total_beds

        # The patient list is fixed-size so slot indices remain stable.
        # Stable indices matter because actions target patient slots directly.
        self.patients = [self._empty_patient_slot() for _ in range(self.config.max_patients)]
        self._active_doc_assignments.clear()

        return self._get_observation()

    def step(
        self, action: Optional[HospitalAction | OpenEnvAction | List[HospitalAction] | List[OpenEnvAction]] = None
    ) -> Tuple[HospitalObservation, float, bool, Dict[str, Any]]:
        """Advance the hospital by one tick.

        Runtime order:
        1. Apply the per-slot action plan, if any.
        2. Progress all patients not newly affected by that action.
        3. Sample and spawn new arrivals.
        4. Record utilization, update score, and return outputs.
        """

        reward = 0.0
        # Convert any accepted action format into a full "one action per slot"
        # plan. This makes the rest of the step logic easy to reason about.
        # After normalization, every patient slot has exactly one entry, even if
        # that entry is just NO_OP.
        chosen_actions = self._normalize_actions(action)
        progressed_patients = set()
        events: List[str] = []

        # First process the agent's decisions for this minute.
        for chosen_action in chosen_actions:
            if chosen_action.action_type == ActionType.NO_OP:
                continue
            # `_apply_agent_action` performs legality checks and applies only the
            # immediate phase change. Ongoing minute-by-minute effects happen in
            # the second loop below.
            reward += self._apply_agent_action(chosen_action, progressed_patients, events)

        # Then let the world evolve for everyone not just touched by an action.
        for i, patient in enumerate(self.patients):
            if not patient.is_active:
                continue
            if patient.patient_id in progressed_patients:
                # A patient changed phase because of this tick's action, so their
                # new state starts progressing on the next minute instead.
                continue

            patient.time_in_state += 1
            rules = self.config.esi_rules[patient.esi_level]

            if patient.phase == PatientPhase.TREATMENT:
                # Treatment is the only phase that uses the active-doctor map.
                reward += self._progress_treatment(i, patient, rules, events)
            elif patient.phase == PatientPhase.RECOVERY:
                reward += self._progress_recovery(i, patient, events)
            elif patient.phase == PatientPhase.WAITING:
                reward += self._progress_waiting(i, patient, rules, events)
            elif patient.phase == PatientPhase.POST_TREAT_WAIT:
                # Patients finished treatment but are blocked waiting for a bed.
                if self.config.use_post_treat_penalty:
                    reward -= self.config.post_treat_penalty

        # New patients arrive after the current minute's state transitions.
        arrivals_this_tick = self._sample_arrivals_this_tick()
        spawned_patients, self.next_patient_id = self.generator.spawn_patients_for_tick(
            self.patients,
            self.current_tick,
            self.next_patient_id,
            arrivals_this_tick,
        )
        # `next_patient_id` is monotonic across the episode even when slots are
        # reused, which keeps historical patient identities unique.
        self.total_patients_seen += len(spawned_patients)
        for spawned_patient in spawned_patients:
            events.append(
                f"arrival {spawned_patient.patient_id} esi={spawned_patient.esi_level} hp={spawned_patient.current_hp:.1f}"
            )

        # Save utilization and bookkeeping after all state changes are done.
        self._record_utilization_snapshot()
        self.current_tick += 1
        self.current_score += reward
        # The episode ends strictly on time horizon, not when the hospital empties.
        done = self.current_tick >= self.max_ticks
        self.last_step_events = events
        info = self._build_info()
        info["task_id"] = self.task_id
        if done:
            info["episode_grade"] = self.grade_episode()

        return self._get_observation(), reward, done, info

    def sample_noop_action(self) -> HospitalAction:
        """Convenience helper for callers that want an explicit no-op action."""

        return HospitalAction(action_type=ActionType.NO_OP)

    def openenv_observation(self) -> OpenEnvObservation:
        """Return the current observation as a typed OpenEnv model."""

        return OpenEnvObservation.from_internal(self._get_observation())

    def state(self) -> OpenEnvState:
        """Return the full internal simulator state."""

        # `state()` is the debugging view: it exposes internals that do not
        # belong in the agent observation, such as active doctor assignments.
        return OpenEnvState(
            task_id=self.task_id,
            current_tick=self.current_tick,
            current_score=self.current_score,
            total_patients_seen=self.total_patients_seen,
            total_deaths=self.total_deaths,
            total_discharged=self.total_discharged,
            illegal_actions=self.illegal_actions,
            free_doctors_t1=self.free_t1,
            free_doctors_t2=self.free_t2,
            free_doctors_t3=self.free_t3,
            free_beds=self.free_beds,
            active_doctor_assignments=self._active_doc_assignments.copy(),
            patients=[OpenEnvPatient.from_internal(patient) for patient in self.patients],
            last_step_events=self.last_step_events.copy(),
            config=self.config.model_dump(),
        )

    def openenv_step(
        self, action: Optional[OpenEnvAction | List[OpenEnvAction]] = None
    ) -> Tuple[OpenEnvObservation, OpenEnvReward]:
        """OpenEnv-style wrapper returning typed observation and reward payload."""

        observation, reward, done, info = self.step(action)
        return OpenEnvObservation.from_internal(observation), OpenEnvReward(
            reward=reward,
            done=done,
            info=info,
        )

    def grade_episode(self) -> float:
        """Grade the current episode for the active task."""

        return grade_episode(self.task_id, self._build_info(), self.current_score)

    def _normalize_actions(
        self, action: Optional[HospitalAction | OpenEnvAction | List[HospitalAction] | List[OpenEnvAction]]
    ) -> List[HospitalAction]:
        """Return a full per-slot action plan for the current tick.

        The returned list always has length `max_patients`, where position `i`
        describes the action for patient slot `i`.
        """

        # Default plan: no action for every patient slot.
        plan = [HospitalAction(action_type=ActionType.NO_OP, target_patient_index=i) for i in range(self.config.max_patients)]
        if action is None:
            return plan
        if isinstance(action, OpenEnvAction):
            return self._normalize_actions(action.to_internal())
        if isinstance(action, HospitalAction):
            # Allow callers to pass a single action and place it into the
            # matching patient slot.
            if action.action_type == ActionType.NO_OP:
                return plan
            if action.target_patient_index is not None and 0 <= action.target_patient_index < self.config.max_patients:
                # A single sparse action gets dropped into its target slot.
                plan[action.target_patient_index] = action
            return plan
        if isinstance(action, list):
            converted_actions: List[HospitalAction] = [
                action_item.to_internal() if isinstance(action_item, OpenEnvAction) else action_item
                for action_item in action
            ]
            if len(converted_actions) == self.config.max_patients:
                # Already a full per-slot plan. Fill in missing indices so slot
                # position and target index stay aligned.
                normalized_plan: List[HospitalAction] = []
                for i, slot_action in enumerate(converted_actions):
                    if slot_action.target_patient_index is None:
                        # If the caller gave a full list but omitted explicit
                        # targets, the slot position itself becomes the target.
                        normalized_plan.append(
                            HospitalAction(
                                action_type=slot_action.action_type,
                                target_patient_index=i,
                                doctor_tier=slot_action.doctor_tier,
                            )
                        )
                    else:
                        normalized_plan.append(slot_action)
                return normalized_plan

            # Sparse list of actions: place each one into its target slot and
            # leave every other slot as NO_OP.
            for sparse_action in converted_actions:
                if sparse_action.target_patient_index is None:
                    continue
                if 0 <= sparse_action.target_patient_index < self.config.max_patients:
                    plan[sparse_action.target_patient_index] = sparse_action
            return plan
        return plan

    def _progress_treatment(
        self,
        index: int,
        patient: PatientVector,
        rules: Dict[str, Any],
        events: List[str],
    ) -> float:
        """Advance one patient already in treatment by one tick."""

        reward = 0.0
        # Treatment consumes one minute of time and keeps its doctor occupied.
        patient.treatment_ticks_left -= 1

        doc_tier = self._active_doc_assignments.get(patient.patient_id)
        if doc_tier is not None:
            if self.config.use_doctor_costs:
                # Doctor cost is charged each tick of treatment, not once at the
                # start, so longer treatments naturally become more expensive.
                reward -= self.config.doctor_costs[doc_tier]

        if patient.treatment_ticks_left == 0:
            # Finishing treatment gives an immediate health boost.
            patient.current_hp = min(100.0, patient.current_hp + rules["boost"])
            if doc_tier is not None:
                self._free_doctor(doc_tier)
                del self._active_doc_assignments[patient.patient_id]
                events.append(f"treatment complete {patient.patient_id}; freed doctor tier {doc_tier}")

            if rules["needs_bed"]:
                # Some cases are not ready to leave yet and must wait for a bed.
                # This is a key congestion point: treatment is finished, but the
                # patient still blocks downstream hospital flow until bed access.
                patient.phase = PatientPhase.POST_TREAT_WAIT
                patient.time_in_state = 0
                events.append(f"{patient.patient_id} moved to post_treat_wait")
            else:
                # Simple cases can leave the hospital immediately after treatment.
                reward += self._discharge_patient(index, free_bed=False, mode="auto", events=events)

        return reward

    def _progress_recovery(self, index: int, patient: PatientVector, events: List[str]) -> float:
        """Advance recovery and auto-discharge once health reaches 100."""

        # Recovery slowly restores health each minute while the bed stays occupied.
        patient.current_hp = min(100.0, patient.current_hp + self.config.recovery_rate)
        if patient.current_hp >= 100.0:
            return self._discharge_patient(index, free_bed=True, mode="auto", events=events)
        return 0.0

    def _progress_waiting(
        self,
        index: int,
        patient: PatientVector,
        rules: Dict[str, Any],
        events: List[str],
    ) -> float:
        """Apply waiting-room deterioration, penalties, and possible death."""

        reward = 0.0
        # Waiting patients deteriorate, but some low-acuity cases have a floor
        # below which waiting alone cannot push them.
        patient.current_hp = max(rules["min_hp"], patient.current_hp - rules["decay"])
        # Lower ESI means more urgent, so it should hurt more to leave them waiting.
        if self.config.use_wait_penalty:
            reward -= (6 - patient.esi_level) * self.config.wait_penalty_multiplier

        if patient.current_hp <= 0:
            self.total_wait_time += patient.time_in_state
            self.completed_waits += 1
            patient.phase = PatientPhase.DEAD
            self.total_deaths += 1
            events.append(f"death {patient.patient_id} while waiting")
            self.patients[index] = self._empty_patient_slot()
            reward -= self.config.death_penalty

        return reward

    def _apply_agent_action(
        self,
        action: HospitalAction,
        progressed_patients: set[str],
        events: List[str],
    ) -> float:
        """Validate and dispatch the agent's chosen action."""

        idx = action.target_patient_index
        # Any malformed index is treated as an illegal action.
        if idx is None or idx < 0 or idx >= self.config.max_patients:
            return self._invalid_action_penalty()

        patient = self.patients[idx]
        # The target slot must currently contain a patient.
        if not patient.is_active:
            return self._invalid_action_penalty()
        if patient.patient_id in progressed_patients:
            # At most one action per patient is allowed within a single tick.
            return self._invalid_action_penalty()

        if action.action_type == ActionType.ASSIGN_DOCTOR:
            # The doctor tier is not chosen automatically by the environment.
            # It must already be present inside action.doctor_tier.
            return self._assign_doctor(patient, action, progressed_patients, events)

        if action.action_type == ActionType.ASSIGN_BED:
            # Bed actions are only valid for patients waiting after treatment.
            return self._assign_bed(patient, progressed_patients, events)

        if action.action_type == ActionType.DISCHARGE:
            # This is an explicit early/manual discharge request by the agent.
            return self._manual_discharge(idx, patient, events)

        return 0.0

    def _assign_doctor(
        self,
        patient: PatientVector,
        action: HospitalAction,
        progressed_patients: set[str],
        events: List[str],
    ) -> float:
        """Move a waiting patient into treatment and reserve one doctor."""

        if patient.phase != PatientPhase.WAITING:
            return self._invalid_action_penalty()

        doc_tier = action.doctor_tier
        if doc_tier is None or doc_tier not in (1, 2, 3):
            return self._invalid_action_penalty()

        # The agent explicitly chooses which doctor tier to use. The environment
        # only validates that choice against severity rules and resource limits.
        # Example: an ESI-1 patient may require tier 3, so sending tier 1 is illegal.
        min_required = self.config.esi_rules[patient.esi_level]["min_doc"]
        if doc_tier < min_required:
            return self._invalid_action_penalty()

        if not self._consume_doctor(doc_tier):
            return self._invalid_action_penalty()

        # Once treatment starts, the patient leaves the waiting queue and the
        # chosen doctor tier remains reserved until treatment completes.
        self.total_wait_time += patient.time_in_state
        self.completed_waits += 1
        # Resetting `time_in_state` here is important because later logic uses it
        # as "minutes spent in the current phase", not total time in hospital.
        patient.phase = PatientPhase.TREATMENT
        patient.time_in_state = 0
        patient.treatment_ticks_left = self.config.esi_rules[patient.esi_level]["treat_ticks"]
        self._active_doc_assignments[patient.patient_id] = doc_tier
        progressed_patients.add(patient.patient_id)
        events.append(f"agent assigned doctor tier {doc_tier} to {patient.patient_id}")
        return 0.0

    def _assign_bed(
        self,
        patient: PatientVector,
        progressed_patients: set[str],
        events: List[str],
    ) -> float:
        """Move a post-treatment patient into recovery and occupy one bed."""

        if patient.phase != PatientPhase.POST_TREAT_WAIT:
            return self._invalid_action_penalty()

        if self.free_beds <= 0:
            return self._invalid_action_penalty()

        # A bed is now occupied until this patient leaves recovery.
        self.free_beds -= 1
        patient.phase = PatientPhase.RECOVERY
        patient.time_in_state = 0
        progressed_patients.add(patient.patient_id)
        events.append(f"agent assigned bed to {patient.patient_id}")
        return 0.0

    def _manual_discharge(self, index: int, patient: PatientVector, events: List[str]) -> float:
        """Allow an early discharge once the configured threshold is reached."""

        if patient.phase != PatientPhase.RECOVERY:
            return self._invalid_action_penalty()

        if patient.current_hp < self.config.discharge_threshold:
            return self._invalid_action_penalty()

        # Manual discharge lets the agent release a patient before full recovery.
        return self._discharge_patient(index, free_bed=True, mode="manual", events=events)

    def _discharge_patient(
        self,
        index: int,
        free_bed: bool,
        mode: str,
        events: List[str],
    ) -> float:
        """Remove a patient from the hospital and award discharge reward."""

        patient = self.patients[index]
        # Reward is based on the patient's condition at the moment they leave.
        # That means discharging earlier can free a bed sooner but may earn less reward.
        reward = self._calculate_discharge_reward(patient)
        if free_bed:
            self.free_beds += 1
        self.total_discharged += 1
        events.append(f"{mode} discharge {patient.patient_id} hp={patient.current_hp:.1f}")
        patient.phase = PatientPhase.DISCHARGED
        self.patients[index] = self._empty_patient_slot()
        return reward

    def _invalid_action_penalty(self) -> float:
        """Record and return the penalty for an illegal action."""

        # Illegal actions are counted even when the task turns off their reward
        # penalty, so evaluation can still inspect policy quality separately.
        self.illegal_actions += 1
        if not self.config.use_invalid_action_penalty:
            return 0.0
        return -self.config.invalid_action_penalty

    def _calculate_discharge_reward(self, patient: PatientVector) -> float:
        """Compute reward from discharge health and patient wealth multiplier."""

        # Healthier discharges are better. Some tasks optionally multiply that
        # value by patient wealth to create an ethical tradeoff for the agent.
        base_value = (patient.current_hp / 100.0) * self.config.discharge_reward_scale
        multiplier = patient.wealth_multiplier if self.config.use_wealth_multiplier else 1.0
        return base_value * multiplier

    def _sample_arrivals_this_tick(self) -> int:
        """Sample how many patients arrive on this tick.

        A Poisson distribution is a standard model for random event counts over
        a fixed time interval. The result is capped to avoid extreme spikes.
        """

        if self.config.max_arrivals_per_tick <= 0:
            return 0
        sampled_arrivals = int(self.np_random.poisson(self.config.mean_arrivals_per_tick))
        return min(sampled_arrivals, self.config.max_arrivals_per_tick)

    def _empty_patient_slot(self) -> PatientVector:
        """Return a blank patient slot marker."""

        # The environment removes patients by replacing them with a fresh empty
        # marker, instead of deleting list entries. That preserves slot indices.
        return PatientVector(is_active=False)

    def _consume_doctor(self, tier: int) -> bool:
        """Reserve one doctor of the requested tier, if available."""

        # Doctors are tracked by tier-specific free counters, not as separate objects.
        if tier == 3 and self.free_t3 > 0:
            self.free_t3 -= 1
            return True
        if tier == 2 and self.free_t2 > 0:
            self.free_t2 -= 1
            return True
        if tier == 1 and self.free_t1 > 0:
            self.free_t1 -= 1
            return True
        return False

    def _free_doctor(self, tier: int) -> None:
        """Release one doctor back into the free pool."""

        # Free counters must mirror `_consume_doctor` exactly so resource
        # accounting stays balanced over long episodes.
        if tier == 3:
            self.free_t3 += 1
        elif tier == 2:
            self.free_t2 += 1
        elif tier == 1:
            self.free_t1 += 1

    def _build_info(self) -> Dict[str, Any]:
        """Build the metrics dictionary returned as `info` from step()."""

        # Current utilization is an instantaneous snapshot at this tick.
        avg_wait_time = self.total_wait_time / self.completed_waits if self.completed_waits else 0.0
        used_beds = self.config.total_beds - self.free_beds
        used_t1 = self.config.total_t1_docs - self.free_t1
        used_t2 = self.config.total_t2_docs - self.free_t2
        used_t3 = self.config.total_t3_docs - self.free_t3
        used_doctors = (
            self.config.total_t1_docs + self.config.total_t2_docs + self.config.total_t3_docs
            - self.free_t1 - self.free_t2 - self.free_t3
        )
        total_doctors = (
            self.config.total_t1_docs + self.config.total_t2_docs + self.config.total_t3_docs
        )
        elapsed_ticks = max(self.current_tick, 1)
        # Average utilizations are time-averaged over the whole episode so far.
        info = StepInfo(
            total_deaths=self.total_deaths,
            total_discharged=self.total_discharged,
            total_patients_seen=self.total_patients_seen,
            avg_wait_time=avg_wait_time,
            bed_utilization=used_beds / self.config.total_beds if self.config.total_beds else 0.0,
            doctor_utilization=used_doctors / total_doctors if total_doctors else 0.0,
            avg_bed_utilization=self.bed_utilization_sum / elapsed_ticks,
            avg_doctor_utilization=self.doctor_utilization_sum / elapsed_ticks,
            avg_t1_doctor_utilization=(
                self.t1_utilization_sum / elapsed_ticks if self.config.total_t1_docs else 0.0
            ),
            avg_t2_doctor_utilization=(
                self.t2_utilization_sum / elapsed_ticks if self.config.total_t2_docs else 0.0
            ),
            avg_t3_doctor_utilization=(
                self.t3_utilization_sum / elapsed_ticks if self.config.total_t3_docs else 0.0
            ),
            illegal_actions_attempted=self.illegal_actions,
            active_patients=sum(1 for patient in self.patients if patient.is_active),
            last_step_events=self.last_step_events.copy(),
        )
        return info.to_dict()

    def _get_observation(self) -> HospitalObservation:
        """Construct the current agent-facing observation and legal action masks."""

        # These lists tell the agent which patient slots can accept each action type.
        can_assign_doctor: List[int] = []
        can_assign_bed: List[int] = []
        can_discharge: List[int] = []

        for i, patient in enumerate(self.patients):
            if not patient.is_active:
                continue

            if patient.phase == PatientPhase.WAITING and self._has_available_doctor_for_patient(patient):
                can_assign_doctor.append(i)
            elif patient.phase == PatientPhase.POST_TREAT_WAIT and self.free_beds > 0:
                # Bed actions are advertised only when at least one bed exists,
                # so callers can avoid obviously illegal actions.
                can_assign_bed.append(i)
            elif (
                patient.phase == PatientPhase.RECOVERY
                and patient.current_hp >= self.config.discharge_threshold
            ):
                can_discharge.append(i)

        return HospitalObservation(
            current_tick=self.current_tick,
            free_doctors_t1=self.free_t1,
            free_doctors_t2=self.free_t2,
            free_doctors_t3=self.free_t3,
            free_beds=self.free_beds,
            patients=self.patients.copy(),
            action_masks=ActionMasks(
                can_assign_doctor=can_assign_doctor,
                can_assign_bed=can_assign_bed,
                can_discharge=can_discharge,
            ),
            current_episode_score=self.current_score,
        )

    def _has_available_doctor_for_patient(self, patient: PatientVector) -> bool:
        """Check whether at least one legal doctor tier is currently free."""

        # Higher-tier doctors can serve more severe patients, but we do not let
        # lower tiers treat patients above their allowed minimum.
        min_required = self.config.esi_rules[patient.esi_level]["min_doc"]
        if min_required <= 1 and self.free_t1 > 0:
            return True
        if min_required <= 2 and self.free_t2 > 0:
            return True
        if min_required <= 3 and self.free_t3 > 0:
            return True
        return False

    def _record_utilization_snapshot(self) -> None:
        """Accumulate one-tick resource utilization for episode averages."""

        # This function is called once per tick after all state changes are done.
        used_beds = self.config.total_beds - self.free_beds
        used_t1 = self.config.total_t1_docs - self.free_t1
        used_t2 = self.config.total_t2_docs - self.free_t2
        used_t3 = self.config.total_t3_docs - self.free_t3
        total_doctors = (
            self.config.total_t1_docs + self.config.total_t2_docs + self.config.total_t3_docs
        )
        used_doctors = used_t1 + used_t2 + used_t3

        if self.config.total_beds:
            self.bed_utilization_sum += used_beds / self.config.total_beds
        if self.config.total_t1_docs:
            self.t1_utilization_sum += used_t1 / self.config.total_t1_docs
        if self.config.total_t2_docs:
            self.t2_utilization_sum += used_t2 / self.config.total_t2_docs
        if self.config.total_t3_docs:
            self.t3_utilization_sum += used_t3 / self.config.total_t3_docs
        if total_doctors:
            self.doctor_utilization_sum += used_doctors / total_doctors
