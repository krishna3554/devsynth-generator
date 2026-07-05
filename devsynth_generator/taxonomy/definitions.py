"""Taxonomy values used to synthesize developer conversations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")

TASK_TYPES = (
    "bug_fix",
    "feature_request",
    "code_review",
    "refactor",
    "test_generation",
    "debugging_session",
    "architecture_discussion",
)
DIFFICULTIES = ("easy", "medium", "hard")
LANGUAGES = ("python", "typescript", "go", "rust", "java")
ROLES = ("user", "assistant")


@dataclass(frozen=True)
class Taxonomy:
    categories: tuple[str, ...] = ()
    subcategories: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    learning_stages: tuple[str, ...] = ()
    conversation_lengths: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    interaction_patterns: tuple[str, ...] = ()


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Taxonomy:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Taxonomy(
        categories=tuple(data.get("categories", data["task_types"])),
        subcategories=tuple(data.get("subcategories", ())),
        intents=tuple(data.get("intents", ())),
        task_types=tuple(data["task_types"]),
        difficulties=tuple(data["difficulties"]),
        learning_stages=tuple(data.get("learning_stages", ())),
        conversation_lengths=tuple(data.get("conversation_lengths", ())),
        languages=tuple(data["languages"]),
        roles=tuple(data["roles"]),
        tools=tuple(data.get("tools", ())),
        interaction_patterns=tuple(data.get("interaction_patterns", ())),
    )


def default_taxonomy() -> Taxonomy:
    if TAXONOMY_PATH.exists():
        return load_taxonomy()

    return Taxonomy(task_types=TASK_TYPES, difficulties=DIFFICULTIES, languages=LANGUAGES, roles=ROLES)
