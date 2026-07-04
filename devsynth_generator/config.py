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


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from `.env` and process environment."""

    load_dotenv(dotenv_path=env_file or PROJECT_ROOT / ".env")

    output_dir = Path(os.getenv("DEVSYNTH_OUTPUT_DIR", PACKAGE_ROOT / "datasets"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return Settings(
        log_level=os.getenv("DEVSYNTH_LOG_LEVEL", "INFO").upper(),
        output_dir=output_dir,
        default_count=int(os.getenv("DEVSYNTH_DEFAULT_COUNT", "10")),
        random_seed=int(os.getenv("DEVSYNTH_RANDOM_SEED", "42")),
    )


def configure_logging(level: str) -> None:
    """Configure root logging once for CLI and library use."""

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
