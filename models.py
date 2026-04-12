"""Core data models shared by the hospital environment and the agent."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# These enums give the simulator a fixed vocabulary for patient states and action names, 
# which helps avoid bugs from raw string comparisons.
class PatientPhase(str, Enum):
    """Discrete phases a patient can occupy while in the hospital."""

    # The environment forms a mostly linear state machine:
    # WAITING -> TREATMENT -> POST_TREAT_WAIT -> RECOVERY -> DISCHARGED
    # with DEAD as a terminal failure state reachable from waiting.
    WAITING = "waiting"
    TREATMENT = "treatment"
    POST_TREAT_WAIT = "post_treat_wait"
    RECOVERY = "recovery"
    DISCHARGED = "discharged"
    DEAD = "dead"

class ActionType(str, Enum):
    """Action categories available to the agent each tick."""

    NO_OP = "no_op"
    ASSIGN_DOCTOR = "assign_doctor"
    ASSIGN_BED = "assign_bed"
    DISCHARGE = "discharge"

class ModelBase(BaseModel):
    """Shared base class for project models."""

    # `use_enum_values=True` keeps JSON payloads simple by serializing enums to
    # their string values instead of Python enum objects.
    model_config = ConfigDict(use_enum_values=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the model into plain Python data."""

        return self.model_dump()


# This object is the full state of one patient slot. If `is_active` is
# false, the slot is empty and available for a future arrival.
class PatientVector(ModelBase):
    """State stored for one patient slot in the environment."""

    is_active: bool = False
    patient_id: str = ""
    esi_level: int = 5
    wealth_multiplier: float = 1.0
    phase: PatientPhase = PatientPhase.WAITING
    current_hp: float = 100.0
    time_in_state: int = 0
    treatment_ticks_left: int = 0

class ActionMasks(ModelBase):
    """Lists of patient indices for which each action type is currently legal."""

    # These masks are hints to the agent about which slots can currently accept
    # each type of action.
    can_assign_doctor: List[int] = Field(default_factory=list)
    can_assign_bed: List[int] = Field(default_factory=list)
    can_discharge: List[int] = Field(default_factory=list)

class HospitalObservation(ModelBase):
    """Observation returned to the agent after reset() and step()."""

    # This is the compact world-state summary the agent receives every tick.
    # It intentionally excludes internal bookkeeping such as cumulative deaths
    # or exact doctor assignments by patient ID.
    current_tick: int
    free_doctors_t1: int
    free_doctors_t2: int
    free_doctors_t3: int
    free_beds: int
    patients: List[PatientVector]
    action_masks: ActionMasks
    current_episode_score: float

class HospitalAction(ModelBase):
    """Action payload passed into env.step(...).

    `target_patient_index` and `doctor_tier` are optional because not every
    action type needs them.
    """

    # `action_type` says what to do. The optional fields provide the action
    # parameters only when they are needed.
    # Example:
    # - `assign_bed` needs only `target_patient_index`
    # - `assign_doctor` needs both `target_patient_index` and `doctor_tier`
    action_type: ActionType = ActionType.NO_OP
    target_patient_index: Optional[int] = None
    doctor_tier: Optional[int] = None

class StepInfo(ModelBase):
    """Additional metrics returned in the `info` dictionary after each step."""

    # These values are mainly for debugging, printing, and episode analysis.
    # Some are instantaneous snapshots (for example `bed_utilization`) while
    # others are running episode averages (for example `avg_bed_utilization`).
    total_deaths: int
    total_discharged: int
    total_patients_seen: int
    avg_wait_time: float
    bed_utilization: float
    doctor_utilization: float
    avg_bed_utilization: float
    avg_doctor_utilization: float
    avg_t1_doctor_utilization: float
    avg_t2_doctor_utilization: float
    avg_t3_doctor_utilization: float
    illegal_actions_attempted: int
    active_patients: int
    last_step_events: List[str]

class EnvConfig(ModelBase):
    """Static environment parameters used to define hospital dynamics."""

    # Capacity limits define how many resources the hospital can use at once.
    max_patients: int = 50
    total_t1_docs: int = 4
    total_t2_docs: int = 2
    total_t3_docs: int = 1
    total_beds: int = 5
    initial_patients: int = 0
    mean_arrivals_per_tick: float = 2.0
    max_arrivals_per_tick: int = 8
    # Reward and penalty scales shape what behavior the agent should prefer.
    # These values are the main knobs used by task presets to make the same
    # simulator feel easier or harder.
    recovery_rate: float = 2.0
    discharge_threshold: float = 60.0
    discharge_reward_scale: float = 20.0
    wait_penalty_multiplier: float = 2.5
    post_treat_penalty: float = 12.0
    death_penalty: float = 250.0
    invalid_action_penalty: float = 50.0
    use_wait_penalty: bool = True
    use_post_treat_penalty: bool = True
    use_doctor_costs: bool = True
    use_wealth_multiplier: bool = True
    use_invalid_action_penalty: bool = True
    doctor_costs: Dict[int, float] = Field(
        default_factory=lambda: {1: 2.0, 2: 6.0, 3: 15.0}
    )
    # Each ESI level has its own waiting decay, minimum safe floor,
    # treatment duration, treatment boost, and minimum doctor tier.
    # `needs_bed=False` means the patient can be discharged directly after
    # treatment instead of entering the post-treatment bed queue.
    esi_rules: Dict[int, Dict[str, Any]] = Field(
        default_factory=lambda: {
            1: {"decay": 8.0, "min_hp": 0.0, "treat_ticks": 6, "boost": 10.0, "min_doc": 3, "needs_bed": True},
            2: {"decay": 4.0, "min_hp": 0.0, "treat_ticks": 3, "boost": 20.0, "min_doc": 2, "needs_bed": True},
            3: {"decay": 1.5, "min_hp": 20.0, "treat_ticks": 2, "boost": 30.0, "min_doc": 1, "needs_bed": True},
            4: {"decay": 0.3, "min_hp": 30.0, "treat_ticks": 1, "boost": 100.0, "min_doc": 1, "needs_bed": False},
            5: {"decay": 0.1, "min_hp": 60.0, "treat_ticks": 1, "boost": 100.0, "min_doc": 1, "needs_bed": False},
        }
    )

class OpenEnvPatient(PatientVector):
    """Pydantic patient schema for OpenEnv-facing state serialization."""

    @classmethod
    def from_internal(cls, patient: PatientVector) -> "OpenEnvPatient":
        # Reuse the internal patient data without manually remapping every field.
        return cls(**patient.to_dict())


class OpenEnvActionMasks(ActionMasks):
    """Pydantic version of current legal-action hints."""

    @classmethod
    def from_internal(cls, masks: ActionMasks) -> "OpenEnvActionMasks":
        # This keeps the API-facing schema aligned with the internal mask object.
        return cls(**masks.to_dict())


class OpenEnvObservation(ModelBase):
    """Typed observation model for OpenEnv compatibility."""

    # This is the exact shape returned by the HTTP layer or OpenEnv wrapper.
    # It mirrors `HospitalObservation`, but uses the OpenEnv-specific Pydantic
    # wrappers so the public API schema is explicit and stable.
    current_tick: int
    free_doctors_t1: int
    free_doctors_t2: int
    free_doctors_t3: int
    free_beds: int
    patients: List[OpenEnvPatient]
    action_masks: OpenEnvActionMasks
    current_episode_score: float

    @classmethod
    def from_internal(cls, observation: HospitalObservation) -> "OpenEnvObservation":
        # Convert the full internal observation into the API-safe model.
        return cls(
            current_tick=observation.current_tick,
            free_doctors_t1=observation.free_doctors_t1,
            free_doctors_t2=observation.free_doctors_t2,
            free_doctors_t3=observation.free_doctors_t3,
            free_beds=observation.free_beds,
            patients=[OpenEnvPatient.from_internal(patient) for patient in observation.patients],
            action_masks=OpenEnvActionMasks.from_internal(observation.action_masks),
            current_episode_score=observation.current_episode_score,
        )


class OpenEnvAction(HospitalAction):
    """Typed action model for OpenEnv compatibility."""

    def to_internal(self) -> HospitalAction:
        # The environment uses `HospitalAction`, so HTTP/API actions get
        # normalized back into that runtime type here.
        return HospitalAction(
            action_type=ActionType(self.action_type),
            target_patient_index=self.target_patient_index,
            doctor_tier=self.doctor_tier,
        )


class OpenEnvReward(ModelBase):
    """Typed reward wrapper for OpenEnv compatibility."""

    # This mirrors the standard environment return shape closely.
    # Keeping reward/done/info together makes HTTP responses easier to consume.
    reward: float
    done: bool
    info: Dict[str, Any]


class OpenEnvState(ModelBase):
    """Full internal simulator state, richer than the agent observation."""

    # `state()` is intentionally more detailed than `observation()` so debugging
    # and validation can inspect hidden environment internals if needed.
    # This endpoint is useful for inspection tools, but agents should usually
    # rely on `HospitalObservation` / `OpenEnvObservation` instead.
    task_id: str
    current_tick: int
    current_score: float
    total_patients_seen: int
    total_deaths: int
    total_discharged: int
    illegal_actions: int
    free_doctors_t1: int
    free_doctors_t2: int
    free_doctors_t3: int
    free_beds: int
    active_doctor_assignments: Dict[str, int]
    patients: List[OpenEnvPatient]
    last_step_events: List[str]
    config: Dict[str, Any]
