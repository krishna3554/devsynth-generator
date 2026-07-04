"""Taxonomy values used to synthesize developer conversations."""

from __future__ import annotations

from dataclasses import dataclass

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
    task_types: tuple[str, ...]
    difficulties: tuple[str, ...]
    languages: tuple[str, ...]
    roles: tuple[str, ...]


def default_taxonomy() -> Taxonomy:
    return Taxonomy(
        task_types=TASK_TYPES,
        difficulties=DIFFICULTIES,
        languages=LANGUAGES,
        roles=ROLES,
    )
