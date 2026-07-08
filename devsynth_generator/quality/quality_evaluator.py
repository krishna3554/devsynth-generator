"""LLM-based quality evaluation for generated conversations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from devsynth_generator.clients import OpenRouterClient
from devsynth_generator.config import PACKAGE_ROOT
from devsynth_generator.models import Conversation
from devsynth_generator.parser import ModelResponseParser, ResponseParseError

LOGGER = logging.getLogger(__name__)

DIMENSION_NAMES = ("technical_accuracy", "helpfulness", "clarity", "realism")

DEFAULT_WEIGHTS: dict[str, float] = {
    "technical_accuracy": 0.35,
    "helpfulness": 0.25,
    "clarity": 0.20,
    "realism": 0.20,
}


class QualityDimension(BaseModel):
    """A single scored quality dimension."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    score: int = Field(..., ge=1, le=10)
    reasoning: str = Field(..., min_length=1)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value not in DIMENSION_NAMES:
            raise ValueError(f"Unknown dimension: {value!r}, expected one of {DIMENSION_NAMES}")
        return value

    @property
    def normalized_score(self) -> float:
        """Map 1-10 integer score to 0.0-1.0 float."""
        return (self.score - 1) / 9.0


class QualityResult(BaseModel):
    """Aggregated quality evaluation result for a conversation."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    dimensions: list[QualityDimension]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    threshold: float = Field(..., ge=0.0, le=1.0)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class QualityConfig:
    """Configurable parameters for quality evaluation."""

    threshold: float = 0.7
    dimension_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_eval_retries: int = 2
    temperature: float = 0.2
    model: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {self.threshold}")
        if self.max_eval_retries < 0:
            raise ValueError(f"max_eval_retries must be non-negative, got {self.max_eval_retries}")
        total = sum(self.dimension_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"dimension_weights must sum to 1.0, got {total}")
        for name in DIMENSION_NAMES:
            if name not in self.dimension_weights:
                raise ValueError(f"Missing weight for dimension: {name}")


class QualityEvaluationError(RuntimeError):
    """Raised when quality evaluation fails after all retry attempts."""


class QualityEvaluator:
    """Score conversations using an LLM judge on four quality dimensions."""

    SYSTEM_PROMPT = "You are an expert evaluator of synthetic developer-assistant conversations. Return only valid JSON."

    def __init__(
        self,
        *,
        client: OpenRouterClient,
        config: QualityConfig | None = None,
        response_parser: ModelResponseParser | None = None,
    ) -> None:
        self.client = client
        self.config = config or QualityConfig()
        self.response_parser = response_parser or ModelResponseParser(max_parse_retries=self.config.max_eval_retries)
        self._prompt_template = self._load_prompt_template()
        self._output_schema = self._load_output_schema()

    def evaluate(self, conversation: Conversation) -> QualityResult:
        """Evaluate a single conversation and return a scored result."""
        prompt = self._build_prompt(conversation)
        try:
            raw_result = self.response_parser.parse_with_retries(
                lambda: self._generate_response(prompt),
                self._validate_payload,
            )
        except ResponseParseError as error:
            raise QualityEvaluationError(
                f"Could not parse quality evaluation for conversation {conversation.id}"
            ) from error

        dimensions = [QualityDimension.model_validate(dim) for dim in raw_result["dimensions"]]
        overall_score = self._compute_overall_score(dimensions)
        passed = overall_score >= self.config.threshold

        result = QualityResult(
            conversation_id=conversation.id,
            dimensions=dimensions,
            overall_score=round(overall_score, 4),
            passed=passed,
            threshold=self.config.threshold,
        )

        LOGGER.info(
            "Quality evaluation conversation=%s overall=%.4f passed=%s scores=%s",
            conversation.id,
            overall_score,
            passed,
            {dim.name: dim.score for dim in dimensions},
        )
        return result

    def evaluate_many(self, conversations: Iterable[Conversation]) -> list[QualityResult]:
        """Evaluate multiple conversations."""
        return [self.evaluate(conversation) for conversation in conversations]

    def filter(
        self, conversations: Iterable[Conversation]
    ) -> tuple[list[Conversation], list[QualityResult]]:
        """Filter conversations, returning accepted ones and all results."""
        accepted: list[Conversation] = []
        results: list[QualityResult] = []
        for conversation in conversations:
            result = self.evaluate(conversation)
            results.append(result)
            if result.passed:
                accepted.append(conversation)
            else:
                LOGGER.warning(
                    "Rejected conversation %s: overall_score=%.4f < threshold=%.4f",
                    conversation.id,
                    result.overall_score,
                    self.config.threshold,
                )
        return accepted, results

    def _build_prompt(self, conversation: Conversation) -> str:
        conversation_json = json.dumps(conversation.to_dict(), indent=2, sort_keys=True)
        return self._prompt_template.format(
            conversation_json=conversation_json,
            output_schema=self._output_schema,
        )

    def _generate_response(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "system_prompt": self.SYSTEM_PROMPT,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.config.model:
            kwargs["model"] = self.config.model
        return self.client.generate_text(prompt, **kwargs)

    def _validate_payload(self, payload: Any) -> dict[str, Any]:
        """Validate that the parsed JSON has the expected evaluation structure."""
        if not isinstance(payload, dict):
            raise ResponseParseError("Evaluation response must be a JSON object")

        dimensions = payload.get("dimensions")
        if not isinstance(dimensions, list):
            raise ResponseParseError("Evaluation response must contain a 'dimensions' array")

        seen_names: set[str] = set()
        for i, dim in enumerate(dimensions):
            if not isinstance(dim, dict):
                raise ResponseParseError(f"dimensions[{i}] must be an object")
            name = dim.get("name")
            if name not in DIMENSION_NAMES:
                raise ResponseParseError(
                    f"dimensions[{i}].name={name!r} is not a valid dimension"
                )
            if name in seen_names:
                raise ResponseParseError(f"Duplicate dimension name: {name!r}")
            seen_names.add(name)
            score = dim.get("score")
            if not isinstance(score, (int, float)) or not (1 <= score <= 10):
                raise ResponseParseError(
                    f"dimensions[{i}].score must be an integer between 1 and 10, got {score!r}"
                )
            dim["score"] = int(score)
            reasoning = dim.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ResponseParseError(
                    f"dimensions[{i}].reasoning must be a non-empty string"
                )

        missing = set(DIMENSION_NAMES) - seen_names
        if missing:
            raise ResponseParseError(f"Missing dimension scores: {sorted(missing)}")

        return payload

    def _compute_overall_score(self, dimensions: list[QualityDimension]) -> float:
        """Compute weighted average of normalized dimension scores."""
        total = 0.0
        for dim in dimensions:
            weight = self.config.dimension_weights[dim.name]
            total += dim.normalized_score * weight
        return total

    def _load_prompt_template(self) -> str:
        path = PACKAGE_ROOT / "prompts" / "quality_evaluation.txt"
        return path.read_text(encoding="utf-8")

    def _load_output_schema(self) -> str:
        path = PACKAGE_ROOT / "schemas" / "quality_evaluation.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(schema, indent=2, sort_keys=True)
