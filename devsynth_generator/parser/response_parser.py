"""Parse and validate JSON returned by model responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

LOGGER = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
ParsedT = TypeVar("ParsedT")


class PayloadValidator(Protocol[ParsedT]):
    def __call__(self, payload: Any) -> ParsedT:
        ...


class ResponseParseError(ValueError):
    """Raised when model output cannot be parsed or validated."""


class ModelResponseParser:
    """Extract JSON from model text and validate it with Pydantic."""

    FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

    def __init__(self, max_parse_retries: int = 2) -> None:
        if max_parse_retries < 0:
            raise ValueError("max_parse_retries must be non-negative")
        self.max_parse_retries = max_parse_retries

    def parse_model(self, text: str, model: type[ModelT]) -> ModelT:
        payload = self.parse_json(text)
        try:
            return model.model_validate(payload)
        except PydanticValidationError as error:
            raise ResponseParseError("Model response did not match Pydantic schema") from error

    def parse_json(self, text: str) -> Any:
        candidates = self._json_candidates(text)
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                LOGGER.debug("JSON candidate failed to parse", exc_info=True)
        raise ResponseParseError("Model response did not contain valid JSON")

    def parse_with_retries(self, generate: Callable[[], str], validator: type[ModelT] | PayloadValidator[ParsedT]) -> ModelT | ParsedT:
        attempts = self.max_parse_retries + 1
        last_error: ResponseParseError | None = None
        for attempt in range(1, attempts + 1):
            text = generate()
            try:
                payload = self.parse_json(text)
                if isinstance(validator, type) and issubclass(validator, BaseModel):
                    return validator.model_validate(payload)
                return validator(payload)
            except (ResponseParseError, PydanticValidationError, ValueError, TypeError) as error:
                last_error = error
                LOGGER.warning(
                    "Model response parse failed attempt=%s/%s error=%s",
                    attempt,
                    attempts,
                    error,
                )
        raise ResponseParseError(f"Model response parsing failed after {attempts} attempts") from last_error

    def _json_candidates(self, text: str) -> list[str]:
        fenced = [match.strip() for match in self.FENCED_JSON_PATTERN.findall(text)]
        bracketed = self._balanced_json_candidates(text)
        stripped = text.strip()
        return [candidate for candidate in [*fenced, *bracketed, stripped] if candidate]

    def _balanced_json_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            if start == -1:
                continue
            end = self._find_balanced_end(text, start, opener, closer)
            if end is not None:
                candidates.append(text[start : end + 1])
        return candidates

    def _find_balanced_end(self, text: str, start: int, opener: str, closer: str) -> int | None:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return index
        return None
