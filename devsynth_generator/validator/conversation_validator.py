"""Validate generated conversation records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from devsynth_generator.models import Conversation
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

        try:
            conversation = Conversation.model_validate(record)
        except PydanticValidationError as error:
            return [
                ValidationError(record_id, self._format_location(item["loc"]), item["msg"])
                for item in error.errors()
            ]

        if conversation.task_type not in self.taxonomy.task_types:
            errors.append(ValidationError(record_id, "task_type", "Unknown task type"))
        if conversation.category is not None and conversation.category not in self.taxonomy.categories:
            errors.append(ValidationError(record_id, "category", "Unknown category"))
        if conversation.subcategory is not None and conversation.subcategory not in self.taxonomy.subcategories:
            errors.append(ValidationError(record_id, "subcategory", "Unknown subcategory"))
        if conversation.intent is not None and conversation.intent not in self.taxonomy.intents:
            errors.append(ValidationError(record_id, "intent", "Unknown intent"))
        if conversation.difficulty not in self.taxonomy.difficulties:
            errors.append(ValidationError(record_id, "difficulty", "Unknown difficulty"))
        if conversation.learning_stage is not None and conversation.learning_stage not in self.taxonomy.learning_stages:
            errors.append(ValidationError(record_id, "learning_stage", "Unknown learning stage"))
        if (
            conversation.conversation_length is not None
            and conversation.conversation_length not in self.taxonomy.conversation_lengths
        ):
            errors.append(ValidationError(record_id, "conversation_length", "Unknown conversation length"))
        if conversation.language not in self.taxonomy.languages:
            errors.append(ValidationError(record_id, "language", "Unknown language"))
        errors.extend(
            ValidationError(record_id, f"code_snippets[{index}].language", "Unknown code snippet language")
            for index, snippet in enumerate(conversation.code_snippets)
            if snippet.language not in self.taxonomy.languages
        )
        errors.extend(self._validate_list_values(record_id, conversation.tools, "tools", self.taxonomy.tools))
        if (
            conversation.interaction_pattern is not None
            and conversation.interaction_pattern not in self.taxonomy.interaction_patterns
        ):
            errors.append(ValidationError(record_id, "interaction_pattern", "Unknown interaction pattern"))

        for index, message in enumerate(conversation.messages):
            if message.role not in self.taxonomy.roles:
                errors.append(ValidationError(record_id, f"messages[{index}].role", "Unknown role"))
            errors.extend(
                ValidationError(
                    record_id,
                    f"messages[{index}].code_snippets[{snippet_index}].language",
                    "Unknown code snippet language",
                )
                for snippet_index, snippet in enumerate(message.code_snippets)
                if snippet.language not in self.taxonomy.languages
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
