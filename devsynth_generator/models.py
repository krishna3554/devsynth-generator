"""Core data models for synthetic conversations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Conversation:
    id: str
    task_type: str
    difficulty: str
    language: str
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
