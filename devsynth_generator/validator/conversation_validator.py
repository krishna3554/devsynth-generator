"""Validate generated conversation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from devsynth_generator.taxonomy import Taxonomy, default_taxonomy


@dataclass(frozen=True)
class ValidationError:
    record_id: str
    field: str
    message: str


class ConversationValidator:
    """Validate records against required fields and taxonomy values."""

    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or default_taxonomy()

    def validate_record(self, record: dict[str, Any]) -> list[ValidationError]:
        record_id = str(record.get("id", "<missing>"))
        errors: list[ValidationError] = []

        for field in ("id", "task_type", "difficulty", "language", "messages"):
            if field not in record:
                errors.append(ValidationError(record_id, field, "Missing required field"))

        if record.get("task_type") not in self.taxonomy.task_types:
            errors.append(ValidationError(record_id, "task_type", "Unknown task type"))
        if record.get("difficulty") not in self.taxonomy.difficulties:
            errors.append(ValidationError(record_id, "difficulty", "Unknown difficulty"))
        if record.get("language") not in self.taxonomy.languages:
            errors.append(ValidationError(record_id, "language", "Unknown language"))
        if "tools" in record:
            errors.extend(self._validate_list_values(record_id, record["tools"], "tools", self.taxonomy.tools))
        if "interaction_pattern" in record and record["interaction_pattern"] not in self.taxonomy.interaction_patterns:
            errors.append(ValidationError(record_id, "interaction_pattern", "Unknown interaction pattern"))

        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append(ValidationError(record_id, "messages", "Messages must be a non-empty list"))
            return errors

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(ValidationError(record_id, f"messages[{index}]", "Message must be an object"))
                continue
            if message.get("role") not in self.taxonomy.roles:
                errors.append(ValidationError(record_id, f"messages[{index}].role", "Unknown role"))
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                errors.append(ValidationError(record_id, f"messages[{index}].content", "Content is required"))

        return errors

    def _validate_list_values(
        self,
        record_id: str,
        values: Any,
        field: str,
        allowed_values: tuple[str, ...],
    ) -> list[ValidationError]:
        if not isinstance(values, list):
            return [ValidationError(record_id, field, "Must be a list")]
        return [
            ValidationError(record_id, f"{field}[{index}]", f"Unknown {field} value")
            for index, value in enumerate(values)
            if value not in allowed_values
        ]

    def validate_many(self, records: list[dict[str, Any]]) -> list[ValidationError]:
        errors: list[ValidationError] = []
        for record in records:
            errors.extend(self.validate_record(record))
        return errors
