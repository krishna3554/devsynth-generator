"""LLM-backed scenario generation."""

from __future__ import annotations

import logging
import uuid

from devsynth_generator.clients import OpenRouterClient
from devsynth_generator.models import Conversation, GeneratorInfo
from devsynth_generator.parser import ModelResponseParser, ResponseParseError
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
        max_parse_retries: int = 2,
        response_parser: ModelResponseParser | None = None,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.taxonomy = taxonomy or default_taxonomy()
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_parser = response_parser or ModelResponseParser(max_parse_retries=max_parse_retries)
        self.coverage_matrix = CoverageMatrix(self.taxonomy, seed=seed)
        self.fallback_generator = ScenarioGenerator(taxonomy=self.taxonomy, seed=seed)
        self.validator = ConversationValidator(taxonomy=self.taxonomy)

    def generate(self, count: int) -> list[Conversation]:
        LOGGER.info("Generating %s LLM-authored coverage-balanced scenarios", count)
        return [self.generate_one(index, spec) for index, spec in enumerate(self.coverage_matrix.build(count))]

    def generate_one(self, index: int, spec: ScenarioSpec) -> Conversation:
        prompt_seed = self.fallback_generator.generate_one(index, spec)
        prompt = self.prompt_builder.build(prompt_seed)
        conversation = self._parse_response_with_retries(prompt, index, spec)
        errors = self.validator.validate_record(conversation.to_dict())
        if errors:
            formatted = ", ".join(f"{error.field}: {error.message}" for error in errors)
            raise LLMScenarioGenerationError(f"Generated scenario failed validation: {formatted}")
        return conversation

    def _parse_response_with_retries(self, prompt: str, index: int, spec: ScenarioSpec) -> Conversation:
        try:
            return self.response_parser.parse_with_retries(
                lambda: self._generate_response_text(prompt),
                lambda payload: self._conversation_from_payload(payload, index, spec),
            )
        except ResponseParseError as error:
            raise LLMScenarioGenerationError("Could not parse a valid conversation from model responses") from error

    def _generate_response_text(self, prompt: str) -> str:
        return self.client.generate_text(
            prompt,
            system_prompt="You generate valid JSON datasets for developer-assistant conversations.",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

    def _conversation_from_payload(self, payload: object, index: int, spec: ScenarioSpec) -> Conversation:
        if not isinstance(payload, dict):
            raise ResponseParseError("Model response JSON must be an object")
        if "conversation" in payload and isinstance(payload["conversation"], dict):
            payload = payload["conversation"]

        # Sanitize tools: keep only values that exist in the taxonomy.
        allowed_tools = set(self.taxonomy.tools)
        if "tools" in payload and isinstance(payload["tools"], list):
            sanitized = [t for t in payload["tools"] if t in allowed_tools]
            payload["tools"] = sanitized if sanitized else ["shell"]
        else:
            payload["tools"] = ["shell"]

        # Sanitize language: ensure it exists in the taxonomy.
        allowed_languages = set(self.taxonomy.languages)
        if payload.get("language") not in allowed_languages:
            payload["language"] = self.taxonomy.languages[0]

        # Sanitize interaction_pattern: ensure it exists in the taxonomy.
        allowed_patterns = set(self.taxonomy.interaction_patterns)
        if payload.get("interaction_pattern") and payload["interaction_pattern"] not in allowed_patterns:
            payload["interaction_pattern"] = None

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
        # Sanitize code snippet languages (top-level and per-message).
        def _sanitize_snippets(snippets: list) -> list:
            for snippet in snippets:
                if isinstance(snippet, dict) and snippet.get("language") not in allowed_languages:
                    snippet["language"] = payload.get("language", self.taxonomy.languages[0])
            return snippets

        if "code_snippets" in payload and isinstance(payload["code_snippets"], list):
            _sanitize_snippets(payload["code_snippets"])
        if "messages" in payload and isinstance(payload["messages"], list):
            for msg in payload["messages"]:
                if isinstance(msg, dict) and "code_snippets" in msg and isinstance(msg["code_snippets"], list):
                    _sanitize_snippets(msg["code_snippets"])

        return Conversation.model_validate(payload)
