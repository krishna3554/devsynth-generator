import json

import pytest

from devsynth_generator.generator import LLMScenarioGenerationError, LLMScenarioGenerator
from devsynth_generator.models import Conversation


class FakeOpenRouterClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "fake/model"
        self.usage = type("Usage", (), {"to_dict": lambda _: {"total_tokens": 12, "requests": 1}})()
        self.prompts = []

    def generate_text(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return self.responses.pop(0)


def valid_payload():
    return {
        "id": "model-made",
        "task_type": "bug_fix",
        "category": "bug_fix",
        "subcategory": "api_design",
        "intent": "ask_for_implementation",
        "difficulty": "easy",
        "learning_stage": "novice",
        "conversation_length": "short",
        "language": "python",
        "messages": [
            {"role": "user", "content": "Please fix this endpoint."},
            {"role": "assistant", "content": "I will add validation and tests."},
        ],
    }


def test_llm_scenario_generator_parses_and_validates_model_output():
    client = FakeOpenRouterClient([json.dumps(valid_payload())])
    generator = LLMScenarioGenerator(client=client, seed=5, temperature=0.1)

    conversation = generator.generate(1)[0]

    assert isinstance(conversation, Conversation)
    assert conversation.generator.name == "LLMScenarioGenerator"
    assert conversation.generator.parameters["model"] == "fake/model"
    assert conversation.metadata.source == "openrouter"
    assert conversation.metadata.token_usage == {"total_tokens": 12, "requests": 1}
    assert "Return only JSON matching this output schema" in client.prompts[0][0]
    assert client.prompts[0][1]["response_format"] == {"type": "json_object"}


def test_llm_scenario_generator_fills_missing_coverage_fields():
    payload = valid_payload()
    for field in ("id", "task_type", "category", "subcategory", "intent", "difficulty", "learning_stage"):
        payload.pop(field, None)
    client = FakeOpenRouterClient([json.dumps(payload)])
    generator = LLMScenarioGenerator(client=client, seed=5)

    conversation = generator.generate(1)[0]

    assert conversation.category is not None
    assert conversation.subcategory is not None
    assert conversation.intent is not None
    assert conversation.task_type == conversation.category


def test_llm_scenario_generator_rejects_invalid_json():
    client = FakeOpenRouterClient(["not json"])
    generator = LLMScenarioGenerator(client=client, seed=5)

    with pytest.raises(LLMScenarioGenerationError):
        generator.generate(1)
