"""Source schema, inference input contract, and per-record validation rules."""

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

DATASET_SCHEMA_VERSION = "diabetes-130-us-v1"
TARGET_COLUMN = "readmitted"
PATIENT_COLUMN = "patient_nbr"
ENCOUNTER_COLUMN = "encounter_id"
TARGET_CLASSES = ("NO", ">30", "<30")

# "?" and empty strings mark missing values in the source. "None" is NOT a
# missing token: in max_glu_serum and A1Cresult it means "test not performed".
MISSING_TOKENS = frozenset({"", "?"})

SOURCE_COLUMNS = (
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmitted",
)

# Never accepted as model inputs: identifiers, the target, a 96.9% missing
# column, and two columns with a single constant value.
NON_INPUT_COLUMNS = frozenset(
    {
        ENCOUNTER_COLUMN,
        PATIENT_COLUMN,
        TARGET_COLUMN,
        "weight",
        "examide",
        "citoglipton",
    }
)
INPUT_COLUMNS = tuple(
    column for column in SOURCE_COLUMNS if column not in NON_INPUT_COLUMNS
)
NUMERIC_INPUTS = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)
CATEGORICAL_INPUTS = tuple(
    column for column in INPUT_COLUMNS if column not in NUMERIC_INPUTS
)

AGE_BUCKETS = tuple(f"[{low}-{low + 10})" for low in range(0, 100, 10))
MEDICATION_COLUMNS = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
)
MEDICATION_VALUES = frozenset({"No", "Steady", "Up", "Down"})
ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "gender": frozenset({"Female", "Male", "Unknown/Invalid"}),
    "age": frozenset(AGE_BUCKETS),
    "max_glu_serum": frozenset({"None", "Norm", ">200", ">300"}),
    "A1Cresult": frozenset({"None", "Norm", ">7", ">8"}),
    "change": frozenset({"No", "Ch"}),
    "diabetesMed": frozenset({"No", "Yes"}),
    **{column: MEDICATION_VALUES for column in MEDICATION_COLUMNS},
}
CODE_COLUMNS = (
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
)
TIME_IN_HOSPITAL_RANGE = (1, 14)

# Discharged as expired (11, 19, 20, 21) or to hospice (13, 14): a readmission
# cannot be observed, so these encounters carry no usable label for training.
EXCLUDED_DISCHARGE_IDS = frozenset({"11", "13", "14", "19", "20", "21"})
EXCLUSION_EXPIRED_OR_HOSPICE = "discharged_expired_or_hospice"


def is_missing(value: Any) -> bool:
    """Return whether a raw or API value represents a missing observation."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in MISSING_TOKENS
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def validate_source_record(record: Mapping[str, Any]) -> list[str]:
    """Return the data-quality violations of one RAW source record."""
    errors: list[str] = []
    missing_columns = [column for column in SOURCE_COLUMNS if column not in record]
    if missing_columns:
        return ["missing_columns:" + ",".join(missing_columns)]

    for column in (ENCOUNTER_COLUMN, PATIENT_COLUMN):
        value = record[column]
        if is_missing(value) or not str(value).strip().isdigit():
            errors.append(f"invalid_{column}")
    if record[TARGET_COLUMN] not in TARGET_CLASSES:
        errors.append("invalid_readmitted")

    for column in NUMERIC_INPUTS:
        value = record[column]
        if is_missing(value) or not str(value).strip().isdigit():
            errors.append(f"invalid_{column}")
    time_in_hospital = str(record["time_in_hospital"]).strip()
    if time_in_hospital.isdigit():
        low, high = TIME_IN_HOSPITAL_RANGE
        if not low <= int(time_in_hospital) <= high:
            errors.append("time_in_hospital_out_of_range")

    for column in CODE_COLUMNS:
        value = record[column]
        if is_missing(value) or not str(value).strip().isdigit():
            errors.append(f"invalid_{column}")
    for column, allowed in ALLOWED_VALUES.items():
        if record[column] not in allowed:
            errors.append(f"invalid_{column}")
    return errors


def exclusion_reason(record: Mapping[str, Any]) -> str | None:
    """Return why a valid record cannot be used for training, if it cannot."""
    if str(record.get("discharge_disposition_id", "")).strip() in EXCLUDED_DISCHARGE_IDS:
        return EXCLUSION_EXPIRED_OR_HOSPICE
    return None


def prepare_input(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one record to the model input contract (44 typed fields).

    Missing tokens become ``None``; numeric fields become ``float``; categorical
    fields, including numeric-looking codes, become ``str``. Extra keys such as
    identifiers or the target are ignored.
    """
    prepared: dict[str, Any] = {}
    for column in INPUT_COLUMNS:
        value = record.get(column)
        if is_missing(value):
            prepared[column] = None
        elif column in NUMERIC_INPUTS:
            try:
                prepared[column] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column} must be numeric, got {value!r}") from exc
        elif isinstance(value, float) and value.is_integer():
            prepared[column] = str(int(value))
        else:
            prepared[column] = str(value).strip()
    return prepared


def input_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a model input DataFrame with the contract's columns and dtypes."""
    frame = pd.DataFrame.from_records(
        [prepare_input(record) for record in records], columns=list(INPUT_COLUMNS)
    )
    for column in NUMERIC_INPUTS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    for column in CATEGORICAL_INPUTS:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame
