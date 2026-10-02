"""Foundation-layer configuration with no import-time credential checks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


PROJECT_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"

@dataclass(frozen=True)
class Settings:
    """Configuration shared by the future HTTP service and utility scripts."""

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None
    jev_api_key: str | None = None
    jev_model: str = "jev-latest"
    jev_base_url: str = "https://api.typesafe.ai"
    local_base_url: str = "http://127.0.0.1:8000"
    max_model_retries: int = 1
    model_timeout_seconds: float = 30.0
    probability_sum_tolerance: float = 0.01
    noul_boolean_threshold: float = 0.5
    max_questions: int = 10
    log_level: str = "INFO"


def _optional(value: str | None) -> str | None:
    return value if value else None


def _number(environ: Mapping[str, str | None], name: str, default: int | float) -> int | float:
    value = environ.get(name)
    if value is None or value == "":
        return default
    try:
        return type(default)(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid {type(default).__name__}") from error


def _load_dotenv(path: Path) -> dict[str, str | None]:
    """Read a dotenv file without mutating process environment variables."""
    return dict(dotenv_values(path))


def get_settings(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> Settings:
    """Read settings without requiring provider credentials.

    Credential validation belongs to the code path that actually makes a
    provider call, so imports and offline tests remain safe. Normal runtime
    reads the repository-root ``.env`` first, then lets process environment
    variables override it. Passing an explicit environment mapping keeps
    tests hermetic unless they also pass a temporary ``dotenv_path``.
    """
    if environ is None:
        env = _load_dotenv(dotenv_path or PROJECT_DOTENV_PATH)
        env.update(os.environ)
    elif dotenv_path is None:
        env = dict(environ)
    else:
        env = _load_dotenv(dotenv_path)
        env.update(environ)
    defaults = Settings()
    return Settings(
        openai_api_key=_optional(env.get("OPENAI_API_KEY")),
        openai_base_url=_optional(env.get("OPENAI_BASE_URL")),
        openai_model=_optional(env.get("OPENAI_MODEL")),
        jev_api_key=_optional(env.get("JEV_API_KEY")),
        jev_model=env.get("JEV_MODEL") or defaults.jev_model,
        jev_base_url=(env.get("JEV_BASE_URL") or defaults.jev_base_url).rstrip("/"),
        local_base_url=(env.get("LOCAL_BASE_URL") or defaults.local_base_url).rstrip("/"),
        max_model_retries=_number(env, "MAX_MODEL_RETRIES", defaults.max_model_retries),
        model_timeout_seconds=_number(env, "MODEL_TIMEOUT_SECONDS", defaults.model_timeout_seconds),
        probability_sum_tolerance=_number(
            env, "PROBABILITY_SUM_TOLERANCE", defaults.probability_sum_tolerance
        ),
        noul_boolean_threshold=_number(
            env, "NOUL_BOOLEAN_THRESHOLD", defaults.noul_boolean_threshold
        ),
        max_questions=_number(env, "MAX_QUESTIONS", defaults.max_questions),
        log_level=env.get("LOG_LEVEL") or defaults.log_level,
    )
