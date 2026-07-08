"""Resumable batch generation pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import ValidationError as PydanticValidationError

from devsynth_generator.deduplication import SemanticDeduplicator
from devsynth_generator.exporter import DatasetExporter
from devsynth_generator.generator import LLMScenarioGenerator, ScenarioSpec
from devsynth_generator.models import Conversation
from devsynth_generator.validator import ValidationError, ValidationPipeline

if TYPE_CHECKING:
    from devsynth_generator.quality import QualityEvaluator

LOGGER = logging.getLogger(__name__)


class ScenarioGeneratorProtocol(Protocol):
    def generate_one(self, index: int, spec: ScenarioSpec) -> Conversation:
        ...


@dataclass(frozen=True)
class BatchGenerationResult:
    path: Path
    requested_count: int
    existing_count: int
    generated_count: int

    @property
    def total_count(self) -> int:
        return self.existing_count + self.generated_count


class BatchGenerationPipeline:
    """Generate one conversation per request and append valid rows to JSONL."""

    def __init__(
        self,
        *,
        generator: LLMScenarioGenerator,
        exporter: DatasetExporter,
        validator: ValidationPipeline | None = None,
        deduplicator: SemanticDeduplicator | None = None,
        quality_evaluator: QualityEvaluator | None = None,
    ) -> None:
        self.generator = generator
        self.exporter = exporter
        self.validator = validator or ValidationPipeline(taxonomy=generator.taxonomy)
        self.deduplicator = deduplicator
        self.quality_evaluator = quality_evaluator

    def run(self, *, count: int, filename: str) -> BatchGenerationResult:
        if count < 0:
            raise ValueError("count must be non-negative")

        path = self.exporter.output_dir / filename
        existing_count = self.count_successful_samples(path)
        existing_conversations = self.load_successful_samples(path, limit=existing_count) if self.deduplicator else []
        if existing_count >= count:
            LOGGER.info("Resume skipped generation; %s already has %s/%s samples", path, existing_count, count)
            return BatchGenerationResult(path=path, requested_count=count, existing_count=existing_count, generated_count=0)

        specs = self.generator.coverage_matrix.build(count)
        generated_count = 0
        for index in range(existing_count, count):
            spec = specs[index]
            LOGGER.info("Generating sample %s/%s", index + 1, count)
            conversation = self.generator.generate_one(index, spec)
            errors = self.validator.validate_record(conversation.to_dict()).errors
            if errors:
                self._log_validation_errors(conversation.id, errors)
                raise ValueError(f"Generated conversation {conversation.id} failed validation")
            if self.deduplicator is not None:
                duplicate = self.deduplicator.is_duplicate(conversation, existing_conversations)
                if duplicate is not None:
                    LOGGER.warning(
                        "Generated conversation %s duplicates %s similarity=%.4f",
                        duplicate.duplicate_id,
                        duplicate.canonical_id,
                        duplicate.similarity,
                    )
                    raise ValueError(
                        f"Generated conversation {duplicate.duplicate_id} duplicates {duplicate.canonical_id}"
                    )
            if self.quality_evaluator is not None:
                quality_result = self.quality_evaluator.evaluate(conversation)
                # Persist quality scores in conversation metadata before export.
                conversation.metadata.__dict__["quality_scores"] = quality_result.to_dict()
                if not quality_result.passed:
                    dims = ", ".join(
                        f"{d.name}={d.score}/10" for d in quality_result.dimensions
                    )
                    raise ValueError(
                        f"Generated conversation {conversation.id} failed quality evaluation: "
                        f"overall={quality_result.overall_score:.4f} < "
                        f"threshold={quality_result.threshold} ({dims})"
                    )
            self.exporter.append_jsonl(conversation, filename)
            existing_conversations.append(conversation)
            generated_count += 1

        return BatchGenerationResult(
            path=path,
            requested_count=count,
            existing_count=existing_count,
            generated_count=generated_count,
        )

    def count_successful_samples(self, path: Path) -> int:
        if not path.exists():
            return 0

        count = 0
        last_valid_position = 0
        should_truncate = False
        with path.open("r+", encoding="utf-8") as handle:
            line_number = 0
            while line := handle.readline():
                line_number += 1
                if not line.strip():
                    last_valid_position = handle.tell()
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Stopping resume scan at invalid JSON line %s in %s", line_number, path)
                    should_truncate = True
                    break
                errors = self.validator.validate_record(record).errors
                if errors:
                    LOGGER.warning("Stopping resume scan at invalid sample line %s in %s", line_number, path)
                    should_truncate = True
                    break
                count += 1
                last_valid_position = handle.tell()
            if should_truncate:
                handle.seek(last_valid_position)
                handle.truncate()
                LOGGER.warning("Truncated %s to the last successful sample", path)
        LOGGER.info("Found %s successful samples in %s", count, path)
        return count

    def load_successful_samples(self, path: Path, limit: int | None = None) -> list[Conversation]:
        if not path.exists():
            return []

        conversations: list[Conversation] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and len(conversations) >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    conversations.append(Conversation.model_validate(record))
                except (json.JSONDecodeError, PydanticValidationError):
                    break
        return conversations

    def _log_validation_errors(self, record_id: str, errors: list[ValidationError]) -> None:
        for error in errors:
            LOGGER.error("%s %s: %s", record_id, error.field, error.message)

