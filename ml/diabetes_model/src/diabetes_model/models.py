"""Candidate estimators, evaluation metrics, and the served pipeline."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from diabetes_model.contract import TARGET_CLASSES
from diabetes_model.features import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    FeatureEngineer,
)

RANDOM_STATE = 42
PRIMARY_METRIC = "validation_macro_f1"
CANDIDATES = ("logistic_regression", "hist_gradient_boosting", "random_forest")
CLASS_METRIC_NAMES = {
    "NO": "no_readmission",
    ">30": "readmitted_after_30_days",
    "<30": "readmitted_within_30_days",
}
MISSING_CATEGORY = "__missing__"


def _one_hot_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), list(ENGINEERED_NUMERIC)),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=MISSING_CATEGORY,
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                list(ENGINEERED_CATEGORICAL),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def _ordinal_preprocessor() -> ColumnTransformer:
    # Unknown categories become -1, which gradient boosting treats as missing.
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(ENGINEERED_NUMERIC)),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=MISSING_CATEGORY,
                                keep_empty_features=True,
                            ),
                        ),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                list(ENGINEERED_CATEGORICAL),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_candidate(name: str) -> Pipeline:
    """Build an unfitted preprocessing-plus-classifier pipeline.

    Every candidate weights classes inversely to their frequency so the 11%
    early-readmission class is not ignored in favour of the majority class.
    """
    if name == "logistic_regression":
        preprocessor = _one_hot_preprocessor(scale_numeric=True)
        classifier = LogisticRegression(
            class_weight="balanced",
            C=0.1,
            max_iter=1000,
            tol=1e-3,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    elif name == "hist_gradient_boosting":
        preprocessor = _ordinal_preprocessor()
        categorical_mask = [False] * len(ENGINEERED_NUMERIC) + [True] * len(
            ENGINEERED_CATEGORICAL
        )
        classifier = HistGradientBoostingClassifier(
            categorical_features=categorical_mask,
            class_weight="balanced",
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )
    elif name == "random_forest":
        preprocessor = _one_hot_preprocessor(scale_numeric=False)
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=50,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown candidate model: {name}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate(model: Any, features: pd.DataFrame, target: pd.Series, *, prefix: str) -> dict[str, float]:
    """Class-aware metrics; ``{prefix}_macro_f1`` is the selection metric."""
    predictions = model.predict(features)
    metrics = evaluate_predictions(target, predictions, prefix=prefix)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        classes = list(model.classes_)
        metrics[f"{prefix}_roc_auc_ovr_macro"] = float(
            roc_auc_score(
                target,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
        metrics[f"{prefix}_log_loss"] = float(
            log_loss(target, probabilities, labels=classes)
        )
    return metrics


def evaluate_predictions(target: pd.Series, predictions: Any, *, prefix: str) -> dict[str, float]:
    """Label-only metrics, usable for any model that returns class labels."""
    labels = list(TARGET_CLASSES)
    metrics = {
        f"{prefix}_macro_f1": float(
            f1_score(target, predictions, labels=labels, average="macro", zero_division=0)
        ),
        f"{prefix}_macro_precision": float(
            precision_score(
                target, predictions, labels=labels, average="macro", zero_division=0
            )
        ),
        f"{prefix}_macro_recall": float(
            recall_score(target, predictions, labels=labels, average="macro", zero_division=0)
        ),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
    }
    recalls = recall_score(target, predictions, labels=labels, average=None, zero_division=0)
    f1_scores = f1_score(target, predictions, labels=labels, average=None, zero_division=0)
    for label, recall, f1 in zip(labels, recalls, f1_scores, strict=True):
        metrics[f"{prefix}_recall_{CLASS_METRIC_NAMES[label]}"] = float(recall)
        metrics[f"{prefix}_f1_{CLASS_METRIC_NAMES[label]}"] = float(f1)
    return metrics


def build_serving_pipeline(fitted_candidate: Pipeline) -> Pipeline:
    """Prefix a fitted candidate with the stateless feature step.

    The result accepts the 44-field input contract, so the API and the model
    comparison feed it the same records without re-implementing features.
    """
    return Pipeline([("features", FeatureEngineer()), *fitted_candidate.steps])
