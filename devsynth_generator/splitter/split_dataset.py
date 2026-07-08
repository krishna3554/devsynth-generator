"""Core dataset splitting engine with stratified distribution preservation."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from devsynth_generator.models import Conversation

LOGGER = logging.getLogger(__name__)


class DatasetSplitError(ValueError):
    """Raised when a dataset cannot be split due to validation failures."""


class DuplicateError(DatasetSplitError):
    """Raised when duplicate IDs or conversations are detected."""


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for dataset splitting."""

    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    input_path: Path = field(default_factory=lambda: Path("datasets/cleaned/conversations.jsonl"))
    output_dir: Path = field(default_factory=lambda: Path("datasets"))

    def __post_init__(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise DatasetSplitError(
                f"Split ratios must sum to 1.0, got {total:.6f} "
                f"(train={self.train_ratio}, validation={self.validation_ratio}, test={self.test_ratio})"
            )
        for name, ratio in [
            ("train_ratio", self.train_ratio),
            ("validation_ratio", self.validation_ratio),
            ("test_ratio", self.test_ratio),
        ]:
            if ratio < 0.0 or ratio > 1.0:
                raise DatasetSplitError(f"{name} must be between 0.0 and 1.0, got {ratio}")


@dataclass(frozen=True)
class SplitResult:
    """Result of a dataset split operation."""

    train: list[Conversation]
    validation: list[Conversation]
    test: list[Conversation]
    skipped_count: int
    duplicates_removed: int

    @property
    def total_valid(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


class DatasetSplitter:
    """Load, validate, deduplicate, and stratify-split a JSONL conversation dataset."""

    def __init__(self, config: SplitConfig | None = None) -> None:
        self.config = config or SplitConfig()

    def run(self) -> SplitResult:
        """Execute the full load → deduplicate → split → save pipeline."""
        conversations, skipped = self.load(self.config.input_path)
        if not conversations:
            raise DatasetSplitError("No valid conversations found in input file")

        conversations, duplicates_removed = self.check_duplicates(conversations)
        result = self.split(conversations, skipped_count=skipped, duplicates_removed=duplicates_removed)

        self.save(result, self.config.output_dir)
        return result

    def load(self, path: Path) -> tuple[list[Conversation], int]:
        """Load and validate conversations from a JSONL file.

        Returns a tuple of (valid_conversations, skipped_count).
        """
        if not path.exists():
            raise FileNotFoundError(f"Input dataset not found: {path}")

        conversations: list[Conversation] = []
        skipped = 0

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as error:
                    LOGGER.warning("Skipping line %s: malformed JSON — %s", line_number, error)
                    skipped += 1
                    continue
                try:
                    conversation = Conversation.model_validate(record)
                    conversations.append(conversation)
                except PydanticValidationError as error:
                    record_id = record.get("id", "<missing>")
                    LOGGER.warning(
                        "Skipping line %s (id=%s): schema validation failed — %s",
                        line_number,
                        record_id,
                        error.error_count(),
                    )
                    skipped += 1
                    continue

        LOGGER.info("Loaded %s valid conversations, skipped %s records from %s", len(conversations), skipped, path)
        return conversations, skipped

    def check_duplicates(self, conversations: list[Conversation]) -> tuple[list[Conversation], int]:
        """Detect and remove duplicate conversations by ID and content hash.

        Returns (deduplicated_conversations, duplicates_removed_count).
        Raises DuplicateError if duplicate IDs are found (IDs must be unique).
        """
        # Phase 1: Check for duplicate IDs (always an error).
        id_counts: dict[str, int] = defaultdict(int)
        for conv in conversations:
            id_counts[conv.id] += 1
        duplicate_ids = {cid: count for cid, count in id_counts.items() if count > 1}
        if duplicate_ids:
            examples = list(duplicate_ids.items())[:5]
            formatted = ", ".join(f"{cid} (×{count})" for cid, count in examples)
            raise DuplicateError(
                f"Found {len(duplicate_ids)} duplicate conversation IDs: {formatted}"
            )

        # Phase 2: Remove content-duplicate conversations (keep first occurrence).
        seen_hashes: dict[str, str] = {}  # hash → first conversation id
        unique: list[Conversation] = []
        removed = 0

        for conv in conversations:
            content_hash = self._content_hash(conv)
            if content_hash in seen_hashes:
                LOGGER.warning(
                    "Removing content-duplicate conversation %s (matches %s)",
                    conv.id,
                    seen_hashes[content_hash],
                )
                removed += 1
            else:
                seen_hashes[content_hash] = conv.id
                unique.append(conv)

        if removed:
            LOGGER.info("Removed %s content-duplicate conversations", removed)
        else:
            LOGGER.info("No duplicate conversations detected")
        return unique, removed

    def split(
        self,
        conversations: list[Conversation],
        *,
        skipped_count: int = 0,
        duplicates_removed: int = 0,
    ) -> SplitResult:
        """Perform a stratified split preserving distribution of key fields.

        Stratification key priority: (category, difficulty, intent).
        Groups too small to split are distributed round-robin.
        """
        rng = random.Random(self.config.seed)
        shuffled = list(conversations)
        rng.shuffle(shuffled)

        # Group by stratification key.
        groups: dict[tuple[str | None, ...], list[Conversation]] = defaultdict(list)
        for conv in shuffled:
            key = (conv.category, conv.difficulty, conv.intent)
            groups[key].append(conv)

        train: list[Conversation] = []
        validation: list[Conversation] = []
        test: list[Conversation] = []

        # Distribute each group proportionally.
        for key in sorted(groups.keys(), key=str):
            group = groups[key]
            n = len(group)

            if n < 3:
                # Too small for proportional slicing; assign each conversation
                # to a split randomly weighted by the configured ratios.
                weights = [
                    self.config.train_ratio,
                    self.config.validation_ratio,
                    self.config.test_ratio,
                ]
                buckets = [train, validation, test]
                for conv in group:
                    chosen = rng.choices(buckets, weights=weights, k=1)[0]
                    chosen.append(conv)
                continue

            # Compute split boundaries.
            n_train = max(1, round(n * self.config.train_ratio))
            n_val = max(1, round(n * self.config.validation_ratio))
            n_test = n - n_train - n_val

            # Ensure at least 1 in test if ratio > 0 and we have enough samples.
            if n_test < 1 and self.config.test_ratio > 0 and n >= 3:
                n_test = 1
                n_train = n - n_val - n_test

            train.extend(group[:n_train])
            validation.extend(group[n_train : n_train + n_val])
            test.extend(group[n_train + n_val :])

        # Final shuffle within each split for training randomness.
        rng.shuffle(train)
        rng.shuffle(validation)
        rng.shuffle(test)

        # Verify no cross-split leakage.
        self._verify_no_leakage(train, validation, test)

        LOGGER.info(
            "Split complete: train=%s, validation=%s, test=%s",
            len(train),
            len(validation),
            len(test),
        )
        return SplitResult(
            train=train,
            validation=validation,
            test=test,
            skipped_count=skipped_count,
            duplicates_removed=duplicates_removed,
        )

    def save(self, result: SplitResult, output_dir: Path) -> dict[str, Path]:
        """Write train/validation/test JSONL files to the output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        for split_name, conversations in [
            ("train", result.train),
            ("validation", result.validation),
            ("test", result.test),
        ]:
            path = output_dir / f"{split_name}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for conv in conversations:
                    handle.write(json.dumps(conv.to_dict(), ensure_ascii=True) + "\n")
            paths[split_name] = path
            LOGGER.info("Wrote %s conversations to %s", len(conversations), path)

        return paths

    def _content_hash(self, conversation: Conversation) -> str:
        """Compute a deterministic hash of conversation content for dedup."""
        # Hash the messages (role + content) as the identity signal.
        parts: list[str] = []
        for msg in conversation.messages:
            parts.append(f"{msg.role}:{msg.content}")
        content = "\n".join(parts)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _verify_no_leakage(
        self,
        train: list[Conversation],
        validation: list[Conversation],
        test: list[Conversation],
    ) -> None:
        """Verify no conversation ID appears in multiple splits."""
        train_ids = {c.id for c in train}
        val_ids = {c.id for c in validation}
        test_ids = {c.id for c in test}

        train_val = train_ids & val_ids
        train_test = train_ids & test_ids
        val_test = val_ids & test_ids

        leaks: list[str] = []
        if train_val:
            leaks.append(f"train∩validation: {sorted(train_val)[:3]}")
        if train_test:
            leaks.append(f"train∩test: {sorted(train_test)[:3]}")
        if val_test:
            leaks.append(f"validation∩test: {sorted(val_test)[:3]}")

        if leaks:
            raise DatasetSplitError(f"Cross-split leakage detected: {'; '.join(leaks)}")
