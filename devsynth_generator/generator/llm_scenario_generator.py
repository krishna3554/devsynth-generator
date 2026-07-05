"""LLM-backed scenario generation."""

from __future__ import annotations

import json
import logging
import uuid

from pydantic import ValidationError as PydanticValidationError

from devsynth_generator.clients import OpenRouterClient
from devsynth_generator.models import Conversation, GeneratorInfo
from devsynth_generator.prompts import PromptBuilder
from devsynth_generator.taxonomy import Taxonomy, default_taxonomy
from devsynth_generator.validator import ConversationValidator

from .scenario_generator import CoverageMatrix, ScenarioGenerator, ScenarioSpec

LOGGER = logging.getLogger(__name__)


class LLMScenarioGenerationError(RuntimeError):
    """Raised when an LLM response cannot be converted into a valid scenario."""


class LLMScenarioGenerator:
    """Generate balanced scenarios by prompting an OpenAI-compatible chat model."""

    def __init__(
        self,
        *,
        client: OpenRouterClient,
        prompt_builder: PromptBuilder | None = None,
        taxonomy: Taxonomy | None = None,
        seed: int | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.taxonomy = taxonomy or default_taxonomy()
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.coverage_matrix = CoverageMatrix(self.taxonomy, seed=seed)
        self.fallback_generator = ScenarioGenerator(taxonomy=self.taxonomy, seed=seed)
        self.validator = ConversationValidator(taxonomy=self.taxonomy)

    def generate(self, count: int) -> list[Conversation]:
        LOGGER.info("Generating %s LLM-authored coverage-balanced scenarios", count)
        return [self.generate_one(index, spec) for index, spec in enumerate(self.coverage_matrix.build(count))]

    def generate_one(self, index: int, spec: ScenarioSpec) -> Conversation:
        prompt_seed = self.fallback_generator.generate_one(index, spec)
        prompt = self.prompt_builder.build(prompt_seed)
        text = self.client.generate_text(
            prompt,
            system_prompt="You generate valid JSON datasets for developer-assistant conversations.",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        conversation = self._parse_response(text, index, spec)
        errors = self.validator.validate_record(conversation.to_dict())
        if errors:
            formatted = ", ".join(f"{error.field}: {error.message}" for error in errors)
            raise LLMScenarioGenerationError(f"Generated scenario failed validation: {formatted}")
        return conversation

    def _parse_response(self, text: str, index: int, spec: ScenarioSpec) -> Conversation:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise LLMScenarioGenerationError("Model response was not valid JSON") from error

        if "conversation" in payload and isinstance(payload["conversation"], dict):
            payload = payload["conversation"]

        payload.setdefault("id", f"llm-scenario-{index:05d}-{uuid.uuid4().hex[:8]}")
        payload.setdefault("task_type", spec.category)
        payload.setdefault("category", spec.category)
        payload.setdefault("subcategory", spec.subcategory)
        payload.setdefault("intent", spec.intent)
        payload.setdefault("difficulty", spec.difficulty)
        payload.setdefault("learning_stage", spec.learning_stage)
        payload.setdefault("conversation_length", spec.conversation_length)
        payload.setdefault("metadata", {})
        payload["metadata"].setdefault("source", "openrouter")
        payload["metadata"].setdefault("coverage", spec.to_dict())
        payload["metadata"].setdefault("token_usage", self.client.usage.to_dict())
        payload["generator"] = {
            "name": "LLMScenarioGenerator",
            "version": "0.1.0",
            "seed": self.seed,
            "parameters": {
                "coverage_matrix": spec.to_dict(),
                "model": self.client.model,
            },
        }

        try:
            return Conversation.model_validate(payload)
        except PydanticValidationError as error:
            raise LLMScenarioGenerationError("Model response did not match Conversation schema") from error
