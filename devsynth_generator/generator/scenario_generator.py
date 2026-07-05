"""Coverage-aware scenario generation."""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from itertools import cycle, islice

from devsynth_generator.models import Conversation, GeneratorInfo, Message
from devsynth_generator.taxonomy import Taxonomy, default_taxonomy

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioSpec:
    """Coverage matrix row describing one generated scenario."""

    category: str
    subcategory: str
    intent: str
    difficulty: str
    learning_stage: str
    conversation_length: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "intent": self.intent,
            "difficulty": self.difficulty,
            "learning_stage": self.learning_stage,
            "conversation_length": self.conversation_length,
        }


class CoverageMatrix:
    """Build scenario specs with even coverage across each taxonomy dimension."""

    def __init__(self, taxonomy: Taxonomy, seed: int | None = None) -> None:
        self.taxonomy = taxonomy
        self.random = random.Random(seed)

    def build(self, count: int) -> list[ScenarioSpec]:
        if count < 0:
            raise ValueError("count must be non-negative")
        dimensions = {
            "category": self._values("categories"),
            "subcategory": self._values("subcategories"),
            "intent": self._values("intents"),
            "difficulty": self._values("difficulties"),
            "learning_stage": self._values("learning_stages"),
            "conversation_length": self._values("conversation_lengths"),
        }
        rotated = {
            name: self._balanced_cycle(values, count, offset)
            for offset, (name, values) in enumerate(dimensions.items())
        }
        specs = [
            ScenarioSpec(
                category=rotated["category"][index],
                subcategory=rotated["subcategory"][index],
                intent=rotated["intent"][index],
                difficulty=rotated["difficulty"][index],
                learning_stage=rotated["learning_stage"][index],
                conversation_length=rotated["conversation_length"][index],
            )
            for index in range(count)
        ]
        self.random.shuffle(specs)
        return specs

    def _values(self, field: str) -> tuple[str, ...]:
        values = getattr(self.taxonomy, field)
        if not values:
            raise ValueError(f"taxonomy.{field} must contain at least one value")
        return values

    def _balanced_cycle(self, values: tuple[str, ...], count: int, offset: int) -> list[str]:
        rotated = values[offset % len(values) :] + values[: offset % len(values)]
        return list(islice(cycle(rotated), count))


class ScenarioGenerator:
    """Generate conversations from a balanced scenario coverage matrix."""

    def __init__(self, taxonomy: Taxonomy | None = None, seed: int | None = None) -> None:
        self.taxonomy = taxonomy or default_taxonomy()
        self.seed = seed
        self.random = random.Random(seed)
        self.coverage_matrix = CoverageMatrix(self.taxonomy, seed=seed)

    def generate(self, count: int) -> list[Conversation]:
        LOGGER.info("Generating %s coverage-balanced scenarios", count)
        return [self.generate_one(index, spec) for index, spec in enumerate(self.coverage_matrix.build(count))]

    def generate_one(self, index: int, spec: ScenarioSpec) -> Conversation:
        language = self.random.choice(self.taxonomy.languages)
        turn_count = self._turn_count(spec.conversation_length)
        messages = self._messages(spec, language, turn_count)

        return Conversation(
            id=f"scenario-{index:05d}-{uuid.uuid4().hex[:8]}",
            task_type=spec.category,
            category=spec.category,
            subcategory=spec.subcategory,
            intent=spec.intent,
            difficulty=spec.difficulty,
            learning_stage=spec.learning_stage,
            conversation_length=spec.conversation_length,
            language=language,
            messages=messages,
            interaction_pattern=self._interaction_pattern(spec),
            metadata={
                "source": "synthetic",
                "turn_count": len(messages),
                "coverage": spec.to_dict(),
            },
            generator=GeneratorInfo(
                name="ScenarioGenerator",
                version="0.1.0",
                seed=self.seed,
                parameters={"coverage_matrix": spec.to_dict()},
            ),
        )

    def _messages(self, spec: ScenarioSpec, language: str, turn_count: int) -> list[Message]:
        topic = spec.category.replace("_", " ")
        messages = [
            Message(
                role="user",
                content=(
                    f"I am a {spec.learning_stage} developer working on a {spec.difficulty} "
                    f"{topic} scenario in {language}. My intent is to {spec.intent.replace('_', ' ')} "
                    f"around {spec.subcategory.replace('_', ' ')}."
                ),
            ),
            Message(
                role="assistant",
                content=(
                    f"Let's approach the {topic} by clarifying the behavior, choosing a small "
                    f"{language} implementation path, and checking the result."
                ),
            ),
        ]
        while len(messages) < turn_count:
            role = "user" if len(messages) % 2 == 0 else "assistant"
            content = (
                "Can you add edge cases and validation steps?"
                if role == "user"
                else "Yes. I would cover the main success path, invalid inputs, and regression checks."
            )
            messages.append(Message(role=role, content=content))
        return messages

    def _turn_count(self, conversation_length: str) -> int:
        return {"short": 2, "medium": 4, "long": 6}.get(conversation_length, 4)

    def _interaction_pattern(self, spec: ScenarioSpec) -> str:
        if spec.intent == "ask_for_review":
            return "review_findings"
        if spec.intent == "ask_for_diagnosis":
            return "multi_turn_debugging"
        if spec.intent in {"ask_for_implementation", "ask_for_tests"}:
            return "implementation_with_tests"
        if spec.intent == "ask_for_explanation":
            return "design_discussion"
        return "single_turn_answer"
