from types import SimpleNamespace

import pytest

from devsynth_generator.clients import OpenRouterClient, OpenRouterClientError


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeOpenAIClient:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))


def response(prompt_tokens=3, completion_tokens=4, total_tokens=7, content="done"):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def test_openrouter_client_tracks_token_usage():
    fake_client = FakeOpenAIClient([response(), response(1, 2, 3)])
    client = OpenRouterClient(
        api_key="test-key",
        client=fake_client,
        model="test-model",
        max_retries=0,
        timeout_seconds=5,
    )

    client.chat_completion([{"role": "user", "content": "hello"}])
    text = client.generate_text("again")

    assert text == "done"
    assert client.usage.to_dict() == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
        "requests": 2,
    }
    assert fake_client.chat.completions.calls[0]["model"] == "test-model"
    assert fake_client.chat.completions.calls[0]["temperature"] == 0.7


def test_openrouter_client_retries_with_exponential_backoff():
    fake_client = FakeOpenAIClient([RuntimeError("temporary"), RuntimeError("again"), response()])
    delays = []
    client = OpenRouterClient(
        api_key="test-key",
        client=fake_client,
        max_retries=2,
        backoff_seconds=0.5,
        sleeper=delays.append,
    )

    client.chat_completion([{"role": "user", "content": "hello"}])

    assert delays == [0.5, 1.0]
    assert len(fake_client.chat.completions.calls) == 3
    assert client.usage.requests == 1


def test_openrouter_client_raises_after_retries_exhausted():
    fake_client = FakeOpenAIClient([RuntimeError("nope"), RuntimeError("still nope")])
    client = OpenRouterClient(
        api_key="test-key",
        client=fake_client,
        max_retries=1,
        backoff_seconds=0,
        sleeper=lambda _: None,
    )

    with pytest.raises(OpenRouterClientError):
        client.chat_completion([{"role": "user", "content": "hello"}])

    assert client.usage.requests == 0
    assert len(fake_client.chat.completions.calls) == 2


def test_openrouter_client_requires_api_key_without_injected_client(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(OpenRouterClientError):
        OpenRouterClient(api_key=None, client=None)
