"""Export generated conversations to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from devsynth_generator.models import Conversation

LOGGER = logging.getLogger(__name__)


class DatasetExporter:
    """Write conversations in common dataset formats."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_jsonl(self, conversations: Iterable[Conversation], filename: str) -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for conversation in conversations:
                handle.write(json.dumps(conversation.to_dict(), ensure_ascii=True) + "\n")
        LOGGER.info("Wrote JSONL dataset to %s", path)
        return path

    def append_jsonl(self, conversation: Conversation, filename: str) -> Path:
        path = self.output_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(conversation.to_dict(), ensure_ascii=True) + "\n")
            handle.flush()
        LOGGER.info("Appended conversation %s to %s", conversation.id, path)
        return path

    def to_json(self, conversations: Iterable[Conversation], filename: str) -> Path:
        path = self.output_dir / filename
        records = [conversation.to_dict() for conversation in conversations]
        path.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding="utf-8")
        LOGGER.info("Wrote JSON dataset to %s", path)
        return path
