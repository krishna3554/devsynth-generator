"""Generate dataset metadata (dataset_info.json)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)


class DatasetInfo(BaseModel):
    """Metadata describing the generated dataset and its splits."""

    model_config = ConfigDict(extra="forbid")

    name: str = "DevSynth"
    version: str = "1.0.0"
    description: str = "Synthetic multi-turn developer conversation dataset."
    synthetic: bool = True
    generator_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    license: str = "Apache-2.0"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    splits: dict[str, int] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_dataset_info(
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    generator_model: str | None = None,
    name: str = "DevSynth",
    version: str = "1.0.0",
) -> DatasetInfo:
    """Build a DatasetInfo with split counts and provenance metadata.

    Args:
        train_size: Number of conversations in the training split.
        validation_size: Number of conversations in the validation split.
        test_size: Number of conversations in the test split.
        generator_model: The model used to generate the data (optional override).
        name: Dataset name.
        version: Dataset version.

    Returns:
        A populated DatasetInfo instance.
    """
    info = DatasetInfo(
        name=name,
        version=version,
        splits={
            "train": train_size,
            "validation": validation_size,
            "test": test_size,
        },
    )
    if generator_model:
        info.generator_model = generator_model

    LOGGER.info(
        "Built dataset info: name=%s, version=%s, splits=%s",
        info.name,
        info.version,
        info.splits,
    )
    return info


def save_dataset_info(info: DatasetInfo, output_dir: Path) -> Path:
    """Write dataset_info.json to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset_info.json"
    path.write_text(
        json.dumps(info.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    LOGGER.info("Wrote dataset info to %s", path)
    return path
