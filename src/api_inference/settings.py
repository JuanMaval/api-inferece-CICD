"""Runtime configuration read from the environment (12-factor style)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All knobs the API reads. Names map 1:1 to environment variables."""

    model_config = SettingsConfigDict(extra="ignore", frozen=True)

    # MLflow registry. ``MLFLOW_TRACKING_URI`` is also honoured natively by the
    # MLflow client; we keep it here so ``/model-info`` and tests can see it.
    mlflow_tracking_uri: str | None = None
    model_name: str = "diabetes"
    model_alias: str = "production"
    # Background polling of the alias. ``0`` disables the refresh task.
    model_refresh_seconds: float = Field(default=60, ge=0)
    # Startup load: bounded retries with linear backoff before the API gives up
    # and starts in ``unavailable`` state (the refresh task keeps retrying).
    model_load_retries: int = Field(default=5, ge=1)
    model_load_backoff_seconds: float = Field(default=2.0, ge=0)

    # Training database (hosts ``inference.logs``). Either a full URL or parts.
    train_database_url: str | None = None
    train_db_host: str | None = None
    train_db_port: int = 5432
    train_db_name: str | None = None
    train_db_user: str | None = None
    train_db_password: str | None = None
    db_pool_min: int = Field(default=1, ge=1)
    db_pool_max: int = Field(default=4, ge=1)
    db_connect_timeout_seconds: float = Field(default=5.0, gt=0)

    # On-disk spool for ``inference.logs`` rows the database could not take.
    # Relative paths resolve against the working directory (``/app`` in the
    # image). An empty value disables the spool (rows are then only counted).
    inference_log_spool_path: str = "spool/inference-logs.jsonl"
    # Period of the background replay of spooled rows; ``0`` disables it (the
    # spool still fills, ``replay`` is then only called explicitly).
    inference_log_replay_seconds: float = Field(default=15, ge=0)

    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        """Resolve the inference-log DSN from ``TRAIN_DATABASE_URL`` or parts."""
        if self.train_database_url:
            return self.train_database_url
        parts = {
            "TRAIN_DB_HOST": self.train_db_host,
            "TRAIN_DB_NAME": self.train_db_name,
            "TRAIN_DB_USER": self.train_db_user,
            "TRAIN_DB_PASSWORD": self.train_db_password,
        }
        missing = [name for name, value in parts.items() if not value]
        if missing:
            raise RuntimeError(
                "Training database configuration is missing: " + ", ".join(missing)
            )
        return (
            f"postgresql://{self.train_db_user}:{self.train_db_password}"
            f"@{self.train_db_host}:{self.train_db_port}/{self.train_db_name}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
