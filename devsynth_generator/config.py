"""Application configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    log_level: str
    output_dir: Path
    default_count: int
    random_seed: int
    openrouter_api_key: str | None
    openrouter_base_url: str
    openrouter_model: str
    openrouter_timeout_seconds: float
    openrouter_max_retries: int
    openrouter_backoff_seconds: float
    dedup_model: str
    dedup_threshold: float
    quality_threshold: float
    quality_model: str | None
    quality_temperature: float


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from `.env` and process environment."""

    load_dotenv(dotenv_path=env_file or PROJECT_ROOT / ".env")

    output_dir = Path(os.getenv("DEVSYNTH_OUTPUT_DIR", PACKAGE_ROOT / "datasets"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    quality_model_env = os.getenv("DEVSYNTH_QUALITY_MODEL")

    return Settings(
        log_level=os.getenv("DEVSYNTH_LOG_LEVEL", "INFO").upper(),
        output_dir=output_dir,
        default_count=int(os.getenv("DEVSYNTH_DEFAULT_COUNT", "10")),
        random_seed=int(os.getenv("DEVSYNTH_RANDOM_SEED", "42")),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        openrouter_timeout_seconds=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60")),
        openrouter_max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "3")),
        openrouter_backoff_seconds=float(os.getenv("OPENROUTER_BACKOFF_SECONDS", "1.0")),
        dedup_model=os.getenv("DEVSYNTH_DEDUP_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        dedup_threshold=float(os.getenv("DEVSYNTH_DEDUP_THRESHOLD", "0.92")),
        quality_threshold=float(os.getenv("DEVSYNTH_QUALITY_THRESHOLD", "0.7")),
        quality_model=quality_model_env if quality_model_env else None,
        quality_temperature=float(os.getenv("DEVSYNTH_QUALITY_TEMPERATURE", "0.2")),
    )


def configure_logging(level: str) -> None:
    """Configure root logging once for CLI and library use."""

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
