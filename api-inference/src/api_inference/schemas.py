"""Pydantic models for the HTTP contract (see docs/data-contract.md)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from diabetes_model.contract import INPUT_COLUMNS, NUMERIC_INPUTS, TARGET_CLASSES
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, create_model


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    database: Literal["ok", "error"]


class ModelInfoResponse(BaseModel):
    name: str
    alias: str
    status: Literal["ready", "unavailable"]
    version: str | None
    run_id: str | None
    loaded_at: datetime | None
    last_check_at: datetime | None
    last_error: str | None
    refresh_seconds: float
    classes: list[str] | None
    tracking_uri: str | None


class ReloadResponse(BaseModel):
    previous_version: str | None
    current_version: str
    changed: bool


class ErrorDetail(BaseModel):
    error_code: str
    message: str


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def _python_name(column: str) -> str:
    """Contract keys such as ``glyburide-metformin`` are not identifiers."""
    return column.replace("-", "_")


def _feature_fields() -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for column in INPUT_COLUMNS:
        if column in NUMERIC_INPUTS:
            # Strict types keep "6" out of numeric fields and 1 out of codes.
            annotation: Any = StrictFloat | StrictInt | None
            description = "Numeric input; JSON number or null."
        else:
            annotation = StrictStr | None
            description = "Categorical input; JSON string (codes included) or null."
        fields[_python_name(column)] = (
            annotation,
            Field(..., alias=column, description=description),
        )
    return fields


# All 44 contract inputs, built from the shared contract so the API and the
# training pipeline cannot drift. Identifiers, the target and the excluded
# columns are unknown keys here and therefore rejected (``extra="forbid"``).
PredictFeatures = create_model(
    "PredictFeatures",
    __config__=ConfigDict(extra="forbid", populate_by_name=False),
    **_feature_fields(),
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID | None = Field(
        default=None, description="Idempotency/audit key; generated when omitted."
    )
    features: PredictFeatures  # type: ignore[valid-type]

    def feature_values(self) -> dict[str, Any]:
        """Validated features keyed by the contract column names."""
        return self.features.model_dump(by_alias=True)


class ModelIdentity(BaseModel):
    name: str
    version: str
    alias: str


class PredictResponse(BaseModel):
    request_id: UUID
    prediction: str
    probabilities: dict[str, float] | None
    model: ModelIdentity
    processing_time_ms: float


# --------------------------------------------------------------------------- #
# /predict/{request_id}/feedback
# --------------------------------------------------------------------------- #
ReadmittedLabel = Literal[TARGET_CLASSES]  # type: ignore[valid-type]


class FeedbackRequest(BaseModel):
    """Observed outcome for a request already stored in ``inference.logs``."""

    model_config = ConfigDict(extra="forbid")

    readmitted: ReadmittedLabel = Field(
        description="Observed readmission outcome; same domain as the training target."
    )
    source: str | None = Field(
        default=None, max_length=200, description="Free-text origin of the label (EHR, audit...)."
    )


class FeedbackResponse(BaseModel):
    request_id: UUID
    readmitted: ReadmittedLabel
    prediction: str | None = Field(
        description="Label the model returned for this request; null when it failed to score."
    )
    created: bool = Field(description="False when a previous label was replaced.")
