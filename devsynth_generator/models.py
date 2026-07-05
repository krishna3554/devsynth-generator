"""Core Pydantic models for synthetic developer conversation datasets."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CodeSnippet(BaseModel):
    """Optional code attached to a message or scenario."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)
    filename: str | None = None
    purpose: str | None = None

    @field_validator("code", "language")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be blank")
        return value


class Message(BaseModel):
    """Single conversation turn."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    code_snippets: list[CodeSnippet] = Field(default_factory=list)

    @field_validator("content", "role")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be blank")
        return value


class ConversationMetadata(BaseModel):
    """Optional provenance and quality metadata for a generated scenario."""

    model_config = ConfigDict(extra="allow")

    source: str = "synthetic"
    turn_count: int | None = Field(default=None, ge=1)
    tags: list[str] = Field(default_factory=list)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GeneratorInfo(BaseModel):
    """Information about the generator that produced the scenario."""

    model_config = ConfigDict(extra="allow")

    name: str = "devsynth-generator"
    version: str = "0.1.0"
    seed: int | None = None
    prompt_template: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class TaxonomyValidationMixin:
    """Shared taxonomy validation helpers."""

    def validate_taxonomy(self, taxonomy: Any) -> list[str]:
        raise NotImplementedError


class Conversation(BaseModel, TaxonomyValidationMixin):
    """One synthetic developer conversation scenario."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1)
    category: str | None = None
    subcategory: str | None = None
    intent: str | None = None
    difficulty: str = Field(..., min_length=1)
    learning_stage: str | None = None
    conversation_length: str | None = None
    language: str = Field(..., min_length=1)
    messages: list[Message] = Field(..., min_length=1)
    code_snippets: list[CodeSnippet] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    interaction_pattern: str | None = None
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)
    generator: GeneratorInfo | None = None

    @model_validator(mode="after")
    def ensure_metadata_turn_count(self) -> Conversation:
        if self.metadata.turn_count is None:
            self.metadata.turn_count = len(self.messages)
        return self

    def validate_taxonomy(self, taxonomy: Any) -> list[str]:
        errors: list[str] = []
        if self.task_type not in taxonomy.task_types:
            errors.append("task_type")
        if self.category is not None and self.category not in taxonomy.categories:
            errors.append("category")
        if self.subcategory is not None and self.subcategory not in taxonomy.subcategories:
            errors.append("subcategory")
        if self.intent is not None and self.intent not in taxonomy.intents:
            errors.append("intent")
        if self.difficulty not in taxonomy.difficulties:
            errors.append("difficulty")
        if self.learning_stage is not None and self.learning_stage not in taxonomy.learning_stages:
            errors.append("learning_stage")
        if self.conversation_length is not None and self.conversation_length not in taxonomy.conversation_lengths:
            errors.append("conversation_length")
        if self.language not in taxonomy.languages:
            errors.append("language")
        errors.extend(
            f"code_snippets[{index}].language"
            for index, snippet in enumerate(self.code_snippets)
            if snippet.language not in taxonomy.languages
        )
        if self.interaction_pattern is not None and self.interaction_pattern not in taxonomy.interaction_patterns:
            errors.append("interaction_pattern")
        errors.extend(f"tools[{index}]" for index, tool in enumerate(self.tools) if tool not in taxonomy.tools)
        for index, message in enumerate(self.messages):
            if message.role not in taxonomy.roles:
                errors.append(f"messages[{index}].role")
            errors.extend(
                f"messages[{index}].code_snippets[{snippet_index}].language"
                for snippet_index, snippet in enumerate(message.code_snippets)
                if snippet.language not in taxonomy.languages
            )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ConversationDataset(BaseModel):
    """A collection of synthetic conversation scenarios."""

    model_config = ConfigDict(extra="forbid")

    name: str = "synthetic-developer-conversations"
    version: str = "0.1.0"
    generator: GeneratorInfo = Field(default_factory=GeneratorInfo)
    conversations: list[Conversation] = Field(default_factory=list)
    split: Literal["train", "validation", "test", "unspecified"] = "unspecified"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def validate_taxonomy(self, taxonomy: Any) -> dict[str, list[str]]:
        return {
            conversation.id: errors
            for conversation in self.conversations
            if (errors := conversation.validate_taxonomy(taxonomy))
        }

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
