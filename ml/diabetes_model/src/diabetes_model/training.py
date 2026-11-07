"""Offline equivalent of the DAG: CSV -> CLEAN rules -> candidates -> metrics."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from diabetes_model.contract import (
    PATIENT_COLUMN,
    TARGET_COLUMN,
    exclusion_reason,
    input_frame,
    validate_source_record,
)
from diabetes_model.features import engineer_features
from diabetes_model.models import CANDIDATES, PRIMARY_METRIC, build_candidate, evaluate
from diabetes_model.splits import assign_split


@dataclass(frozen=True)
class PreparedDataset:
    """Training-eligible rows with contract inputs, features, label, and split."""

    inputs: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    split: pd.Series
    rejected_rows: int
    excluded_rows: int


def prepare_dataset(csv_path: str | Path) -> PreparedDataset:
    """Apply the same RAW validation, exclusion, and CLEAN rules as the DAG."""
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    records = frame.to_dict(orient="records")
    valid = [record for record in records if not validate_source_record(record)]
    eligible = [record for record in valid if exclusion_reason(record) is None]
    inputs = input_frame(eligible)
    return PreparedDataset(
        inputs=inputs,
        features=engineer_features(inputs),
        target=pd.Series([r[TARGET_COLUMN] for r in eligible], dtype="string"),
        split=pd.Series([assign_split(r[PATIENT_COLUMN]) for r in eligible], dtype="string"),
        rejected_rows=len(records) - len(valid),
        excluded_rows=len(valid) - len(eligible),
    )


def train_candidates(
    data: PreparedDataset, candidates: tuple[str, ...] = CANDIDATES
) -> dict[str, tuple[Pipeline, dict[str, float]]]:
    """Fit every candidate on train and report train/validation/test metrics."""
    results = {}
    masks = {name: data.split.eq(name).to_numpy() for name in ("train", "validation", "test")}
    for name in candidates:
        model = build_candidate(name)
        model.fit(data.features[masks["train"]], data.target[masks["train"]])
        metrics: dict[str, float] = {}
        for split_name, mask in masks.items():
            metrics.update(
                evaluate(model, data.features[mask], data.target[mask], prefix=split_name)
            )
        results[name] = (model, metrics)
    return results


def best_candidate(results: dict[str, tuple[Pipeline, dict[str, float]]]) -> str:
    """Return the candidate with the highest primary selection metric."""
    return max(results, key=lambda name: results[name][1][PRIMARY_METRIC])
