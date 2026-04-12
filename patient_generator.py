"""Patient arrival generation logic for the hospital environment."""

import numpy as np
from typing import List, Optional, Tuple
from models import PatientVector, PatientPhase


class PatientGenerator:
    """Samples new patients from the configured arrival distributions."""

    def __init__(self, rng: np.random.RandomState):
        """Store the RNG so the environment controls reproducibility."""

        self.rng = rng

    def generate_patient(self, tick: int, slot_index: int, arrival_id: int) -> PatientVector:
        """Create one newly arrived patient in the waiting state."""

        # First sample the clinical profile of the arriving patient.
        # The probabilities make low-acuity patients common and critical cases rare.
        esi = int(self.rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.3, 0.4]))
        # Wealth is a simple binary multiplier used only in tasks that enable it.
        wealth = float(self.rng.choice([1.0, 2.5], p=[0.8, 0.2]))

        # Lower ESI number means higher acuity, so critical patients start sicker.
        hp_ranges = {1: (10, 30), 2: (20, 40), 3: (40, 60), 4: (60, 80), 5: (80, 100)}
        initial_hp = float(self.rng.uniform(*hp_ranges[esi]))

        # The patient enters directly into the waiting phase and occupies one
        # open slot in the environment state.
        return PatientVector(
            is_active=True,
            # The ID encodes arrival order, tick, and slot. That makes logs much
            # easier to read when tracing how one patient moved through the system.
            patient_id=f"P{arrival_id:05d}_T{tick:04d}_S{slot_index:03d}",
            esi_level=esi,
            wealth_multiplier=wealth,
            phase=PatientPhase.WAITING,
            current_hp=initial_hp,
            time_in_state=0,
            treatment_ticks_left=0,
        )

    def spawn_patient_if_space(
        self,
        patients: List[PatientVector],
        tick: int,
        next_patient_id: int,
    ) -> Tuple[Optional[PatientVector], int]:
        """Create and insert one patient into the first free slot, if possible."""

        # The generator now owns both patient creation and slot insertion so
        # arrival-related logic stays in one place.
        for slot_index, patient in enumerate(patients):
            if patient.is_active:
                continue

            # Slot indices stay stable over time, so we reuse the first inactive
            # slot rather than growing the list dynamically.
            spawned_patient = self.generate_patient(tick, slot_index, next_patient_id)
            patients[slot_index] = spawned_patient
            return spawned_patient, next_patient_id + 1

        return None, next_patient_id

    def spawn_patients_for_tick(
        self,
        patients: List[PatientVector],
        tick: int,
        next_patient_id: int,
        num_arrivals: int,
    ) -> Tuple[List[PatientVector], int]:
        """Insert up to `num_arrivals` patients into open slots for this tick."""

        spawned_patients: List[PatientVector] = []
        current_next_id = next_patient_id

        # Stop early if the hospital runs out of free patient slots.
        # This silently drops excess arrivals once capacity is exhausted.
        for _ in range(num_arrivals):
            spawned_patient, current_next_id = self.spawn_patient_if_space(
                patients,
                tick,
                current_next_id,
            )
            if spawned_patient is None:
                break
            spawned_patients.append(spawned_patient)

        return spawned_patients, current_next_id
