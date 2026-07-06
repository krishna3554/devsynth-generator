"""Comprehensive validation pipeline for generated conversations."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import ValidationError as PydanticValidationError

from devsynth_generator.models import Conversation
from devsynth_generator.taxonomy import Taxonomy, default_taxonomy
from devsynth_generator.validator.conversation_validator import ConversationValidator, ValidationError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationPipelineResult:
    errors: list[ValidationError]

    @property
    def is_valid(self) -> bool:
        return not self.errors


class ValidationPipeline:
    """Run schema, taxonomy, PII, length, and metadata consistency checks."""

    LENGTH_CONSTRAINTS = {
        "short": (2, 2),
        "medium": (3, 4),
        "long": (5, 6),
    }
    PII_PATTERNS = {
        "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    }

    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or default_taxonomy()
        self.conversation_validator = ConversationValidator(taxonomy=self.taxonomy)

    def validate_record(self, record: dict[str, Any]) -> ValidationPipelineResult:
        record_id = str(record.get("id", "<missing>"))
        errors = self.conversation_validator.validate_record(record)
        try:
            conversation = Conversation.model_validate(record)
        except PydanticValidationError as error:
            schema_errors = [
                ValidationError(record_id, self._format_location(item["loc"]), item["msg"])
                for item in error.errors()
            ]
            self._log_errors(schema_errors)
            return ValidationPipelineResult(errors=schema_errors)

        errors.extend(self._detect_pii(conversation))
        errors.extend(self._validate_conversation_length(conversation))
        errors.extend(self._validate_metadata_consistency(conversation))
        self._log_errors(errors)
        return ValidationPipelineResult(errors=errors)

    def validate_many(self, records: Iterable[dict[str, Any]]) -> ValidationPipelineResult:
        errors: list[ValidationError] = []
        for record in records:
            errors.extend(self.validate_record(record).errors)
        return ValidationPipelineResult(errors=errors)

    def _detect_pii(self, conversation: Conversation) -> list[ValidationError]:
        errors: list[ValidationError] = []
        for index, message in enumerate(conversation.messages):
            errors.extend(self._pii_errors(conversation.id, f"messages[{index}].content", message.content))
            for snippet_index, snippet in enumerate(message.code_snippets):
                errors.extend(
                    self._pii_errors(
                        conversation.id,
                        f"messages[{index}].code_snippets[{snippet_index}].code",
                        snippet.code,
                    )
                )
        for snippet_index, snippet in enumerate(conversation.code_snippets):
            errors.extend(self._pii_errors(conversation.id, f"code_snippets[{snippet_index}].code", snippet.code))
        return errors

    def _pii_errors(self, record_id: str, field: str, text: str) -> list[ValidationError]:
        return [
            ValidationError(record_id, field, f"Possible PII detected: {name}")
            for name, pattern in self.PII_PATTERNS.items()
            if pattern.search(text)
        ]

    def _validate_conversation_length(self, conversation: Conversation) -> list[ValidationError]:
        if conversation.conversation_length is None:
            return []
        bounds = self.LENGTH_CONSTRAINTS.get(conversation.conversation_length)
        if bounds is None:
            return []
        minimum, maximum = bounds
        turn_count = len(conversation.messages)
        if minimum <= turn_count <= maximum:
            return []
        return [
            ValidationError(
                conversation.id,
                "messages",
                f"Conversation length '{conversation.conversation_length}' requires {minimum}-{maximum} turns",
            )
        ]

    def _validate_metadata_consistency(self, conversation: Conversation) -> list[ValidationError]:
        errors: list[ValidationError] = []
        if conversation.metadata.turn_count != len(conversation.messages):
            errors.append(
                ValidationError(
                    conversation.id,
                    "metadata.turn_count",
                    "Metadata turn_count must match number of messages",
                )
            )

        coverage = getattr(conversation.metadata, "coverage", None)
        if isinstance(coverage, dict):
            for field in ("category", "subcategory", "intent", "difficulty", "learning_stage", "conversation_length"):
                if field in coverage and getattr(conversation, field) != coverage[field]:
                    errors.append(
                        ValidationError(
                            conversation.id,
                            f"metadata.coverage.{field}",
                            f"Coverage metadata must match conversation {field}",
                        )
                    )

        generator = conversation.generator
        if generator is not None:
            matrix = generator.parameters.get("coverage_matrix")
            if isinstance(matrix, dict):
                for field in ("category", "subcategory", "intent", "difficulty", "learning_stage", "conversation_length"):
                    if field in matrix and getattr(conversation, field) != matrix[field]:
                        errors.append(
                            ValidationError(
                                conversation.id,
                                f"generator.parameters.coverage_matrix.{field}",
                                f"Generator coverage matrix must match conversation {field}",
                            )
                        )
        return errors

    def _format_location(self, location: tuple[str | int, ...]) -> str:
        field = ""
        for part in location:
            if isinstance(part, int):
                field += f"[{part}]"
            elif field:
                field += f".{part}"
            else:
                field = part
        return field

    def _log_errors(self, errors: list[ValidationError]) -> None:
        for error in errors:
            LOGGER.error("%s %s: %s", error.record_id, error.field, error.message)
