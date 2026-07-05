"""OpenRouter client using the OpenAI-compatible SDK."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from openai import OpenAI

from devsynth_generator.config import Settings, load_settings

LOGGER = logging.getLogger(__name__)


class OpenRouterClientError(RuntimeError):
    """Raised when an OpenRouter request fails after all retry attempts."""


@dataclass
class TokenUsage:
    """Cumulative token usage tracked across client requests."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            self.requests += 1
            return
        self.prompt_tokens += int(self._read_usage_value(usage, "prompt_tokens"))
        self.completion_tokens += int(self._read_usage_value(usage, "completion_tokens"))
        self.total_tokens += int(self._read_usage_value(usage, "total_tokens"))
        self.requests += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "requests": self.requests,
        }

    def _read_usage_value(self, usage: Any, field: str) -> int:
        if isinstance(usage, Mapping):
            return int(usage.get(field, 0) or 0)
        return int(getattr(usage, field, 0) or 0)


class OpenRouterClient:
    """Small resilient wrapper around OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
        settings: Settings | None = None,
        client: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        settings = settings or load_settings()
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key and client is None:
            raise OpenRouterClientError("OPENROUTER_API_KEY is required")

        self.model = model or settings.openrouter_model
        self.base_url = base_url or settings.openrouter_base_url
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.openrouter_timeout_seconds
        self.max_retries = max_retries if max_retries is not None else settings.openrouter_max_retries
        self.backoff_seconds = backoff_seconds if backoff_seconds is not None else settings.openrouter_backoff_seconds
        self.sleeper = sleeper
        self.usage = TokenUsage()
        self.client = client or OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def chat_completion(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion with retries and usage tracking."""

        request_model = model or self.model
        payload: dict[str, Any] = {
            "model": request_model,
            "messages": list(messages),
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_headers is not None:
            payload["extra_headers"] = dict(extra_headers)

        response = self._request_with_retries(payload)
        self.usage.add(getattr(response, "usage", None))
        LOGGER.info("OpenRouter request completed model=%s usage=%s", request_model, self.usage.to_dict())
        return response

    def generate_text(self, prompt: str, *, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Generate text from a user prompt and return the first response message."""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.chat_completion(messages, **kwargs)
        return response.choices[0].message.content or ""

    def reset_usage(self) -> None:
        self.usage = TokenUsage()

    def _request_with_retries(self, payload: Mapping[str, Any]) -> Any:
        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            LOGGER.info(
                "OpenRouter request attempt=%s/%s model=%s messages=%s timeout=%s",
                attempt,
                attempts,
                payload["model"],
                len(payload["messages"]),
                self.timeout_seconds,
            )
            try:
                return self.client.chat.completions.create(**payload)
            except Exception as error:
                last_error = error
                if attempt == attempts:
                    break
                delay = self.backoff_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "OpenRouter request failed attempt=%s/%s retry_in=%.2fs error=%s",
                    attempt,
                    attempts,
                    delay,
                    error,
                )
                self.sleeper(delay)

        raise OpenRouterClientError(f"OpenRouter request failed after {attempts} attempts") from last_error
