"""Resumable batch generation pipeline with per-sample retry logic."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError as PydanticValidationError

from devsynth_generator.clients import OpenRouterClientError
from devsynth_generator.deduplication import SemanticDeduplicator
from devsynth_generator.exporter import DatasetExporter
from devsynth_generator.generator import LLMScenarioGenerationError, LLMScenarioGenerator, ScenarioSpec
from devsynth_generator.models import Conversation
from devsynth_generator.parser import ResponseParseError
from devsynth_generator.pipeline.generation_metrics import GenerationMetrics
from devsynth_generator.validator import ValidationError, ValidationPipeline

if TYPE_CHECKING:
    from devsynth_generator.quality import QualityEvaluator

LOGGER = logging.getLogger(__name__)

# Default maximum retries per sample before skipping.
DEFAULT_MAX_RETRIES = 5

# Errors that indicate the API key / model / quota is permanently broken.
# These should NOT be retried.
_FATAL_API_ERROR_KEYWORDS = frozenset({
    "invalid api key",
    "authentication",
    "unauthorized",
    "invalid model",
    "quota exceeded",
    "insufficient_quota",
    "billing",
})


class ScenarioGeneratorProtocol(Protocol):
    def generate_one(self, index: int, spec: ScenarioSpec) -> Conversation:
        ...


@dataclass(frozen=True)
class BatchGenerationResult:
    path: Path
    requested_count: int
    existing_count: int
    generated_count: int
    failed_count: int = 0
    total_retries: int = 0
    failures_path: Path | None = None

    @property
    def total_count(self) -> int:
        return self.existing_count + self.generated_count


class BatchGenerationPipeline:
    """Generate one conversation per request with per-sample retry logic.

    Validation failures, parse errors, and transient API errors trigger
    automatic retries.  The batch never crashes because of a single bad
    LLM response.
    """

    def __init__(
        self,
        *,
        generator: LLMScenarioGenerator,
        exporter: DatasetExporter,
        validator: ValidationPipeline | None = None,
        deduplicator: SemanticDeduplicator | None = None,
        quality_evaluator: QualityEvaluator | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.generator = generator
        self.exporter = exporter
        self.validator = validator or ValidationPipeline(taxonomy=generator.taxonomy)
        self.deduplicator = deduplicator
        self.quality_evaluator = quality_evaluator
        self.max_retries = max(1, max_retries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *, count: int, filename: str) -> BatchGenerationResult:
        if count < 0:
            raise ValueError("count must be non-negative")

        path = self.exporter.output_dir / filename
        existing_count = self.count_successful_samples(path)
        existing_conversations = (
            self.load_successful_samples(path, limit=existing_count)
            if self.deduplicator
            else []
        )
        if existing_count >= count:
            LOGGER.info(
                "Resume skipped generation; %s already has %s/%s samples",
                path, existing_count, count,
            )
            return BatchGenerationResult(
                path=path,
                requested_count=count,
                existing_count=existing_count,
                generated_count=0,
            )

        metrics = GenerationMetrics(requested=count - existing_count)
        specs = self.generator.coverage_matrix.build(count)

        # Derive the failures filename from the output filename.
        failures_filename = f"{Path(filename).stem}_failures.jsonl"

        for index in range(existing_count, count):
            spec = specs[index]
            LOGGER.info("Generating sample %s/%s", index + 1, count)
            metrics.start_sample()

            success = self._generate_with_retry(
                index=index,
                spec=spec,
                filename=filename,
                failures_filename=failures_filename,
                existing_conversations=existing_conversations,
                metrics=metrics,
            )
            if not success:
                metrics.record_skip()
                LOGGER.warning(
                    "Sample %s/%s: maximum retries (%s) exceeded — skipping",
                    index + 1, count, self.max_retries,
                )

        # Print summary to log.
        summary = metrics.summary()
        LOGGER.info(summary)
        print(summary)

        failures_path = self.exporter.output_dir / failures_filename
        return BatchGenerationResult(
            path=path,
            requested_count=count,
            existing_count=existing_count,
            generated_count=metrics.generated,
            failed_count=metrics.skipped,
            total_retries=metrics.total_retries,
            failures_path=failures_path if failures_path.exists() else None,
        )

    # ------------------------------------------------------------------
    # Per-sample retry loop
    # ------------------------------------------------------------------

    def _generate_with_retry(
        self,
        *,
        index: int,
        spec: ScenarioSpec,
        filename: str,
        failures_filename: str,
        existing_conversations: list[Conversation],
        metrics: GenerationMetrics,
    ) -> bool:
        """Try to generate and validate one sample up to max_retries times.

        Returns True if a valid conversation was saved, False if all
        attempts were exhausted.
        """
        for attempt in range(1, self.max_retries + 1):
            LOGGER.info(
                "Sample %s — attempt %s/%s",
                index + 1, attempt, self.max_retries,
            )
            try:
                conversation = self.generator.generate_one(index, spec)
            except (LLMScenarioGenerationError, ResponseParseError) as exc:
                reason = f"generation_error: {exc}"
                LOGGER.warning("Sample %s attempt %s/%s failed: %s", index + 1, attempt, self.max_retries, reason)
                metrics.record_retry(reason)
                self._save_failure(failures_filename, index, spec, attempt, reason, raw_data=None)
                continue
            except OpenRouterClientError as exc:
                if self._is_fatal_api_error(exc):
                    LOGGER.error("Fatal API error, aborting: %s", exc)
                    raise
                reason = f"api_error: {exc}"
                LOGGER.warning("Sample %s attempt %s/%s API error: %s", index + 1, attempt, self.max_retries, reason)
                metrics.record_api_error()
                metrics.record_retry(reason)
                self._save_failure(failures_filename, index, spec, attempt, reason, raw_data=None)
                continue
            except Exception as exc:
                reason = f"unexpected_error: {type(exc).__name__}: {exc}"
                LOGGER.warning("Sample %s attempt %s/%s unexpected error: %s", index + 1, attempt, self.max_retries, reason)
                metrics.record_retry(reason)
                self._save_failure(failures_filename, index, spec, attempt, reason, raw_data=None)
                continue

            # ---- Validation ----
            record = conversation.to_dict()
            validation_result = self.validator.validate_record(record)
            if not validation_result.is_valid:
                errors_text = "; ".join(f"{e.field}: {e.message}" for e in validation_result.errors)
                reason = f"validation_failed: {errors_text}"
                LOGGER.warning(
                    "Sample %s attempt %s/%s validation failed:\n  %s",
                    index + 1, attempt, self.max_retries,
                    "\n  ".join(f"- {e.field}: {e.message}" for e in validation_result.errors),
                )
                metrics.record_retry(reason)
                self._save_failure(failures_filename, index, spec, attempt, reason, raw_data=record)
                continue

            # ---- Deduplication ----
            if self.deduplicator is not None:
                duplicate = self.deduplicator.is_duplicate(conversation, existing_conversations)
                if duplicate is not None:
                    reason = (
                        f"duplicate: {duplicate.duplicate_id} duplicates "
                        f"{duplicate.canonical_id} similarity={duplicate.similarity:.4f}"
                    )
                    LOGGER.warning("Sample %s attempt %s/%s: %s", index + 1, attempt, self.max_retries, reason)
                    metrics.record_retry(reason)
                    self._save_failure(failures_filename, index, spec, attempt, reason, raw_data=record)
                    continue

            # ---- Quality evaluation ----
            if self.quality_evaluator is not None:
                quality_result = self.quality_evaluator.evaluate(conversation)
                conversation.metadata.__dict__["quality_scores"] = quality_result.to_dict()
                if not quality_result.passed:
                    dims = ", ".join(f"{d.name}={d.score}/10" for d in quality_result.dimensions)
                    reason = (
                        f"quality_failed: overall={quality_result.overall_score:.4f} < "
                        f"threshold={quality_result.threshold} ({dims})"
                    )
                    LOGGER.warning("Sample %s attempt %s/%s: %s", index + 1, attempt, self.max_retries, reason)
                    metrics.record_retry(reason)
                    self._save_failure(failures_filename, index, spec, attempt, reason, raw_data=record)
                    continue

            # ---- Success ----
            self.exporter.append_jsonl(conversation, filename)
            existing_conversations.append(conversation)
            metrics.record_success(attempts=attempt)
            if attempt > 1:
                LOGGER.info("Sample %s succeeded after %s retries", index + 1, attempt - 1)
            return True

        # All retries exhausted.
        return False

    # ------------------------------------------------------------------
    # Failure persistence
    # ------------------------------------------------------------------

    def _save_failure(
        self,
        failures_filename: str,
        index: int,
        spec: ScenarioSpec,
        attempt: int,
        reason: str,
        raw_data: dict[str, Any] | None,
    ) -> None:
        """Append a failed generation record to the failures file."""
        failure_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "sample_index": index,
            "attempt": attempt,
            "reason": reason,
            "scenario_spec": spec.to_dict(),
            "raw_data": raw_data,
        }
        path = self.exporter.output_dir / failures_filename
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure_record, ensure_ascii=True, default=str) + "\n")
                handle.flush()
        except OSError:
            LOGGER.warning("Could not write failure record to %s", path)

    # ------------------------------------------------------------------
    # Fatal API error detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_fatal_api_error(exc: OpenRouterClientError) -> bool:
        """Return True if the error indicates a permanent API problem."""
        msg = str(exc).lower()
        return any(keyword in msg for keyword in _FATAL_API_ERROR_KEYWORDS)

    # ------------------------------------------------------------------
    # Resume helpers (unchanged)
    # ------------------------------------------------------------------

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
