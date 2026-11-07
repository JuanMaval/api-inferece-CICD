"""Feature engineering shared by CLEAN generation and the served model."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from diabetes_model.contract import (
    AGE_BUCKETS,
    INPUT_COLUMNS,
    MEDICATION_COLUMNS,
    NUMERIC_INPUTS,
    is_missing,
)

TRANSFORMATION_VERSION = "readmission-features-v2"

# Medications whose non-"No" share is at least 0.1% keep their own column.
# The rest (under 90 treated encounters each) only feed the count features.
MODEL_MEDICATIONS = (
    "metformin",
    "repaglinide",
    "nateglinide",
    "glimepiride",
    "glipizide",
    "glyburide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "insulin",
    "glyburide-metformin",
)
LAB_RESULT_COLUMNS = ("max_glu_serum", "A1Cresult")
NOT_MEASURED = "None"

ENGINEERED_NUMERIC = (
    *NUMERIC_INPUTS,
    "age_years",
    "total_visits",
    "num_active_medications",
    "num_medication_changes",
)
ENGINEERED_CATEGORICAL = (
    "race",
    "gender",
    "payer_code",
    "medical_specialty_group",
    "admission_type",
    "admission_source",
    "discharge_group",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    *LAB_RESULT_COLUMNS,
    *MODEL_MEDICATIONS,
    "change",
    "diabetesMed",
)
FEATURE_COLUMNS = (*ENGINEERED_NUMERIC, *ENGINEERED_CATEGORICAL)

# ICD-9 chapters as grouped by Strack et al. (2014) for this dataset.
ICD9_RANGES = (
    ("circulatory", 390, 460),
    ("respiratory", 460, 520),
    ("digestive", 520, 580),
    ("genitourinary", 580, 630),
    ("musculoskeletal", 710, 740),
    ("injury", 800, 1000),
    ("neoplasms", 140, 240),
)
ICD9_SYMPTOM_CODES = {
    785: "circulatory",
    786: "respiratory",
    787: "digestive",
    788: "genitourinary",
}
SPECIALTY_GROUPS = {
    "InternalMedicine": "internal_medicine",
    "Emergency/Trauma": "emergency",
    "Family/GeneralPractice": "family_practice",
    "Cardiology": "cardiology",
    "Nephrology": "nephrology",
    "Pulmonology": "pulmonology",
    "Psychiatry": "psychiatry",
    "Urology": "urology",
    "Gastroenterology": "gastroenterology",
    "Oncology": "oncology",
    "Hematology/Oncology": "oncology",
    "ObstetricsandGynecology": "obstetrics_gynecology",
}
SPECIALTY_PREFIX_GROUPS = (
    ("Surgery", "surgery"),
    ("Surgeon", "surgery"),
    ("Orthopedics", "orthopedics"),
    ("Radiolog", "radiology"),
    ("Pediatrics", "pediatrics"),
)
ADMISSION_TYPE_GROUPS = {
    "1": "emergency",
    "2": "urgent",
    "3": "elective",
    "4": "other",
    "7": "other",
    "5": "unknown",
    "6": "unknown",
    "8": "unknown",
}
ADMISSION_SOURCE_GROUPS = {
    **dict.fromkeys(("1", "2", "3"), "referral"),
    "7": "emergency_room",
    **dict.fromkeys(("4", "5", "6", "10", "18", "22", "25", "26"), "transfer"),
    **dict.fromkeys(("9", "15", "17", "20", "21"), "unknown"),
}
DISCHARGE_GROUPS = {
    "1": "home",
    **dict.fromkeys(("6", "8"), "home_health"),
    **dict.fromkeys(
        ("2", "3", "4", "5", "10", "15", "22", "23", "24", "27", "28", "29", "30"),
        "facility_transfer",
    ),
    "7": "left_against_advice",
    **dict.fromkeys(("18", "25", "26"), "unknown"),
}


def icd9_group(code: Any) -> str | None:
    """Map an ICD-9 diagnosis code to its clinical chapter."""
    if is_missing(code):
        return None
    code = str(code).strip()
    if code[:1] in ("V", "E"):
        return "other"
    try:
        value = float(code)
    except ValueError:
        return "other"
    if 250 <= value < 251:
        return "diabetes"
    if int(value) in ICD9_SYMPTOM_CODES:
        return ICD9_SYMPTOM_CODES[int(value)]
    for group, low, high in ICD9_RANGES:
        if low <= value < high:
            return group
    return "other"


def specialty_group(specialty: Any) -> str | None:
    """Collapse 72 admitting specialties into frequent clinical groups."""
    if is_missing(specialty):
        return None
    specialty = str(specialty).strip()
    if specialty in SPECIALTY_GROUPS:
        return SPECIALTY_GROUPS[specialty]
    for prefix, group in SPECIALTY_PREFIX_GROUPS:
        if specialty.startswith(prefix):
            return group
    return "other"


def _code_group(groups: Mapping[str, str], fallback: str) -> Any:
    def group(code: Any) -> str:
        if is_missing(code):
            return "unknown"
        return groups.get(str(code).strip(), fallback)

    return group


def _age_years(bucket: Any) -> float:
    if is_missing(bucket) or bucket not in AGE_BUCKETS:
        return float("nan")
    return AGE_BUCKETS.index(bucket) * 10.0 + 5.0


def _categorical(values: pd.Series) -> pd.Series:
    """Use np.nan for missing categories: SimpleImputer does not detect None."""
    values = values.astype(object)
    return values.where(~values.map(is_missing), np.nan)


def engineer_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Derive the model features from contract inputs (see ``prepare_input``)."""
    missing = [column for column in INPUT_COLUMNS if column not in inputs.columns]
    if missing:
        raise ValueError("Inputs are missing columns: " + ", ".join(missing))

    features = pd.DataFrame(index=inputs.index)
    for column in NUMERIC_INPUTS:
        features[column] = pd.to_numeric(inputs[column], errors="coerce").astype(
            "float64"
        )
    features["age_years"] = inputs["age"].map(_age_years).astype("float64")
    features["total_visits"] = features[
        ["number_outpatient", "number_emergency", "number_inpatient"]
    ].sum(axis=1, min_count=1)
    medications = inputs[list(MEDICATION_COLUMNS)].astype(object)
    features["num_active_medications"] = (
        medications.isin(["Steady", "Up", "Down"]).sum(axis=1).astype("float64")
    )
    features["num_medication_changes"] = (
        medications.isin(["Up", "Down"]).sum(axis=1).astype("float64")
    )

    features["race"] = _categorical(inputs["race"])
    gender = inputs["gender"].astype(object)
    features["gender"] = _categorical(gender.where(gender != "Unknown/Invalid", None))
    features["payer_code"] = _categorical(inputs["payer_code"])
    features["medical_specialty_group"] = _categorical(
        inputs["medical_specialty"].map(specialty_group)
    )
    features["admission_type"] = inputs["admission_type_id"].map(
        _code_group(ADMISSION_TYPE_GROUPS, "unknown")
    )
    features["admission_source"] = inputs["admission_source_id"].map(
        _code_group(ADMISSION_SOURCE_GROUPS, "other")
    )
    features["discharge_group"] = inputs["discharge_disposition_id"].map(
        _code_group(DISCHARGE_GROUPS, "other")
    )
    for column in ("diag_1", "diag_2", "diag_3"):
        features[f"{column}_group"] = _categorical(inputs[column].map(icd9_group))
    for column in LAB_RESULT_COLUMNS:
        features[column] = (
            inputs[column]
            .astype(object)
            .where(~inputs[column].map(is_missing), NOT_MEASURED)
        )
    for column in (*MODEL_MEDICATIONS, "change", "diabetesMed"):
        features[column] = _categorical(inputs[column])

    return features.loc[:, list(FEATURE_COLUMNS)]


def feature_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Rebuild a typed feature DataFrame from CLEAN ``features`` JSON objects."""
    frame = pd.DataFrame.from_records(list(rows), columns=list(FEATURE_COLUMNS))
    for column in ENGINEERED_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ENGINEERED_CATEGORICAL:
        frame[column] = _categorical(frame[column])
    return frame


def features_to_json(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize engineered features to JSON-safe dicts (NaN becomes null)."""
    records = []
    for row in features.itertuples(index=False, name=None):
        record = {}
        for name, value in zip(features.columns, row, strict=True):
            if is_missing(value):
                record[name] = None
            elif hasattr(value, "item"):
                record[name] = value.item()
            else:
                record[name] = value
        records.append(record)
    return records


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Stateless sklearn step so the served model accepts contract inputs."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=list(INPUT_COLUMNS))
        return engineer_features(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(FEATURE_COLUMNS, dtype=object)
