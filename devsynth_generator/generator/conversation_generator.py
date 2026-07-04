"""Generate synthetic developer conversations."""

from __future__ import annotations

import logging
import random
import uuid

from devsynth_generator.models import Conversation, Message
from devsynth_generator.taxonomy import Taxonomy, default_taxonomy

LOGGER = logging.getLogger(__name__)


class ConversationGenerator:
    """Create deterministic synthetic examples when seeded."""

    def __init__(self, taxonomy: Taxonomy | None = None, seed: int | None = None) -> None:
        self.taxonomy = taxonomy or default_taxonomy()
        self.random = random.Random(seed)

    def generate(self, count: int) -> list[Conversation]:
        LOGGER.info("Generating %s conversations", count)
        return [self.generate_one(index) for index in range(count)]

    def generate_one(self, index: int = 0) -> Conversation:
        task_type = self.random.choice(self.taxonomy.task_types)
        difficulty = self.random.choice(self.taxonomy.difficulties)
        language = self.random.choice(self.taxonomy.languages)
        topic = task_type.replace("_", " ")

        return Conversation(
            id=f"conv-{index:05d}-{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            difficulty=difficulty,
            language=language,
            messages=[
                Message(
                    role="user",
                    content=(
                        f"I need help with a {difficulty} {topic} in {language}. "
                        "Please reason through the approach and provide a concrete implementation."
                    ),
                ),
                Message(
                    role="assistant",
                    content=(
                        f"Let's handle the {topic} by isolating the expected behavior, "
                        f"making a focused {language} change, and adding a verification step."
                    ),
                ),
                Message(
                    role="user",
                    content="Can you include edge cases and explain the tradeoffs briefly?",
                ),
                Message(
                    role="assistant",
                    content=(
                        "Yes. I would cover empty inputs, invalid state, and the main success path, "
                        "then keep the implementation small enough to review safely."
                    ),
                ),
            ],
            metadata={
                "source": "synthetic",
                "turn_count": 4,
            },
        )
