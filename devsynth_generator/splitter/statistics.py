"""Compute comprehensive statistics for a split dataset."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from devsynth_generator.models import Conversation

LOGGER = logging.getLogger(__name__)


class DistributionEntry(BaseModel):
    """A single value in a categorical distribution."""

    model_config = ConfigDict(extra="forbid")

    value: str
    count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)


class DatasetStatistics(BaseModel):
    """Comprehensive statistics for a split conversation dataset."""

    model_config = ConfigDict(extra="forbid")

    total_conversations: int = Field(..., ge=0)
    train_size: int = Field(..., ge=0)
    validation_size: int = Field(..., ge=0)
    test_size: int = Field(..., ge=0)
    average_turns: float = Field(..., ge=0.0)
    average_messages: float = Field(..., ge=0.0)
    average_code_snippet_percentage: float = Field(..., ge=0.0, le=100.0)
    average_conversation_length: float = Field(..., ge=0.0)
    category_distribution: list[DistributionEntry] = Field(default_factory=list)
    subcategory_distribution: list[DistributionEntry] = Field(default_factory=list)
    intent_distribution: list[DistributionEntry] = Field(default_factory=list)
    difficulty_distribution: list[DistributionEntry] = Field(default_factory=list)
    learning_stage_distribution: list[DistributionEntry] = Field(default_factory=list)
    programming_language_distribution: list[DistributionEntry] = Field(default_factory=list)
    top_skills: list[DistributionEntry] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def compute_statistics(
    all_conversations: list[Conversation],
    train: list[Conversation],
    validation: list[Conversation],
    test: list[Conversation],
) -> DatasetStatistics:
    """Compute statistics across all conversations and per-split sizes.

    Args:
        all_conversations: All conversations before splitting (for global stats).
        train: Training split conversations.
        validation: Validation split conversations.
        test: Test split conversations.

    Returns:
        A fully populated DatasetStatistics instance.
    """
    total = len(all_conversations)
    if total == 0:
        return DatasetStatistics(
            total_conversations=0,
            train_size=0,
            validation_size=0,
            test_size=0,
            average_turns=0.0,
            average_messages=0.0,
            average_code_snippet_percentage=0.0,
            average_conversation_length=0.0,
        )

    # Turn and message counts.
    turn_counts = [len(conv.messages) for conv in all_conversations]
    avg_turns = sum(turn_counts) / total
    avg_messages = avg_turns  # turns == messages in this schema

    # Code snippet percentage: fraction of conversations that contain at least one snippet.
    snippet_counts: list[int] = []
    for conv in all_conversations:
        count = len(conv.code_snippets)
        for msg in conv.messages:
            count += len(msg.code_snippets)
        snippet_counts.append(count)
    conversations_with_snippets = sum(1 for c in snippet_counts if c > 0)
    avg_snippet_pct = (conversations_with_snippets / total) * 100.0

    # Average conversation length (total characters in all messages).
    total_chars = [
        sum(len(msg.content) for msg in conv.messages) for conv in all_conversations
    ]
    avg_conv_length = sum(total_chars) / total

    # Categorical distributions.
    category_dist = _build_distribution(
        [conv.category or "unknown" for conv in all_conversations], total
    )
    subcategory_dist = _build_distribution(
        [conv.subcategory or "unknown" for conv in all_conversations], total
    )
    intent_dist = _build_distribution(
        [conv.intent or "unknown" for conv in all_conversations], total
    )
    difficulty_dist = _build_distribution(
        [conv.difficulty for conv in all_conversations], total
    )
    learning_stage_dist = _build_distribution(
        [conv.learning_stage or "unknown" for conv in all_conversations], total
    )
    language_dist = _build_distribution(
        [conv.language for conv in all_conversations], total
    )

    # Top skills: aggregate tools used across all conversations.
    all_tools: list[str] = []
    for conv in all_conversations:
        all_tools.extend(conv.tools)
    top_skills = _build_distribution(all_tools, total) if all_tools else []

    stats = DatasetStatistics(
        total_conversations=total,
        train_size=len(train),
        validation_size=len(validation),
        test_size=len(test),
        average_turns=round(avg_turns, 2),
        average_messages=round(avg_messages, 2),
        average_code_snippet_percentage=round(avg_snippet_pct, 2),
        average_conversation_length=round(avg_conv_length, 2),
        category_distribution=category_dist,
        subcategory_distribution=subcategory_dist,
        intent_distribution=intent_dist,
        difficulty_distribution=difficulty_dist,
        learning_stage_distribution=learning_stage_dist,
        programming_language_distribution=language_dist,
        top_skills=top_skills,
    )

    LOGGER.info(
        "Statistics computed: total=%s, avg_turns=%.2f, avg_snippet_pct=%.2f%%",
        total,
        avg_turns,
        avg_snippet_pct,
    )
    return stats


def save_statistics(statistics: DatasetStatistics, output_dir: Path) -> Path:
    """Write statistics.json to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "statistics.json"
    path.write_text(
        json.dumps(statistics.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    LOGGER.info("Wrote statistics to %s", path)
    return path


def _build_distribution(values: list[str], total: int) -> list[DistributionEntry]:
    """Build a sorted distribution from a list of categorical values."""
    counter = Counter(values)
    denominator = total if total > 0 else 1
    entries = [
        DistributionEntry(
            value=value,
            count=count,
            percentage=round((count / denominator) * 100.0, 2),
        )
        for value, count in counter.most_common()
    ]
    return entries
