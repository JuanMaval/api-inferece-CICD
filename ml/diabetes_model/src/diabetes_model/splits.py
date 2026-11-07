"""Deterministic, patient-disjoint train/validation/test assignment."""

import hashlib

SPLIT_SALT = "diabetes-patient-split-v1"
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_NAMES = ("train", "validation", "test")


def assign_split(patient_id: str) -> str:
    """Assign a patient to a split from a salted hash of its identifier.

    The assignment depends only on the patient, so every encounter of a patient
    lands in the same partition regardless of the batch or order it arrives in,
    and earlier assignments never change as new batches are loaded.
    """
    if patient_id is None or not str(patient_id).strip():
        raise ValueError("A patient identifier is required to assign a split.")
    digest = hashlib.sha256(f"{SPLIT_SALT}:{str(patient_id).strip()}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < TRAIN_FRACTION:
        return "train"
    if position < TRAIN_FRACTION + VALIDATION_FRACTION:
        return "validation"
    return "test"
