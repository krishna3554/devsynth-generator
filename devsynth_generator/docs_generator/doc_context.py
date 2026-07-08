"""Shared documentation context loaded from dataset metadata and statistics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from devsynth_generator.models import Conversation
from devsynth_generator.taxonomy import Taxonomy, default_taxonomy

LOGGER = logging.getLogger(__name__)

# Maximum example conversations to load from the training split.
MAX_EXAMPLES = 20


@dataclass
class DocContext:
    """Aggregated context used by all documentation generators.

    Holds dataset metadata, statistics, taxonomy values, and sample
    conversations so every generator can produce consistent output
    without re-reading files.
    """

    # From dataset_info.json
    dataset_name: str = "DevSynth"
    version: str = "1.0.0"
    description: str = "Synthetic multi-turn developer conversation dataset."
    license: str = "Apache-2.0"
    generator_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    synthetic: bool = True
    splits: dict[str, int] = field(default_factory=dict)

    # From statistics.json
    statistics: dict[str, Any] = field(default_factory=dict)

    # From taxonomy
    taxonomy: Taxonomy = field(default_factory=default_taxonomy)

    # Sample conversations loaded from train.jsonl
    examples: list[Conversation] = field(default_factory=list)

    @property
    def total_conversations(self) -> int:
        return int(self.statistics.get("total_conversations", sum(self.splits.values())))

    @property
    def num_categories(self) -> int:
        return len(self.taxonomy.categories)


def load_doc_context(
    input_dir: Path,
    *,
    taxonomy: Taxonomy | None = None,
    max_examples: int = MAX_EXAMPLES,
) -> DocContext:
    """Build a DocContext from dataset files on disk.

    Reads ``dataset_info.json``, ``statistics.json``, and sample
    conversations from ``train.jsonl``.

    Args:
        input_dir: Directory containing the dataset split files.
        taxonomy: Optional taxonomy override; defaults to the built-in taxonomy.
        max_examples: Maximum example conversations to load.

    Returns:
        A fully populated DocContext.
    """
    ctx = DocContext(taxonomy=taxonomy or default_taxonomy())

    # --- dataset_info.json ---
    info_path = input_dir / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        ctx.dataset_name = info.get("name", ctx.dataset_name)
        ctx.version = info.get("version", ctx.version)
        ctx.description = info.get("description", ctx.description)
        ctx.license = info.get("license", ctx.license)
        ctx.generator_model = info.get("generator_model", ctx.generator_model)
        ctx.created_at = info.get("created_at", ctx.created_at)
        ctx.synthetic = info.get("synthetic", ctx.synthetic)
        ctx.splits = info.get("splits", ctx.splits)
        LOGGER.info("Loaded dataset info from %s", info_path)
    else:
        LOGGER.warning("dataset_info.json not found in %s; using defaults", input_dir)

    # --- statistics.json ---
    stats_path = input_dir / "statistics.json"
    if stats_path.exists():
        ctx.statistics = json.loads(stats_path.read_text(encoding="utf-8"))
        LOGGER.info("Loaded statistics from %s", stats_path)
    else:
        LOGGER.warning("statistics.json not found in %s; statistics will be empty", input_dir)

    # --- Sample conversations from train.jsonl ---
    train_path = input_dir / "train.jsonl"
    if train_path.exists():
        ctx.examples = _load_examples(train_path, max_examples)
        LOGGER.info("Loaded %s example conversations from %s", len(ctx.examples), train_path)
    else:
        LOGGER.warning("train.jsonl not found in %s; no examples available", input_dir)

    return ctx


def _load_examples(path: Path, limit: int) -> list[Conversation]:
    """Load up to *limit* valid conversations from a JSONL file."""
    conversations: list[Conversation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(conversations) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                conversations.append(Conversation.model_validate(record))
            except (json.JSONDecodeError, PydanticValidationError):
                continue
    return conversations
