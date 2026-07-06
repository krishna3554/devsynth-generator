"""Semantic deduplication with sentence-transformers embeddings."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from devsynth_generator.config import load_settings
from devsynth_generator.models import Conversation

LOGGER = logging.getLogger(__name__)


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], normalize_embeddings: bool = False) -> Any:
        ...


@dataclass(frozen=True)
class DuplicateMatch:
    duplicate_id: str
    canonical_id: str
    similarity: float


@dataclass(frozen=True)
class DeduplicationResult:
    unique_conversations: list[Conversation]
    duplicates: list[DuplicateMatch]


class SemanticDeduplicator:
    """Filter near-duplicate conversations with embedding cosine similarity."""

    def __init__(
        self,
        *,
        threshold: float | None = None,
        model_name: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        settings = load_settings()
        self.threshold = threshold if threshold is not None else settings.dedup_threshold
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.model_name = model_name or settings.dedup_model
        self._embedder = embedder

    def deduplicate(self, conversations: Sequence[Conversation]) -> DeduplicationResult:
        if not conversations:
            return DeduplicationResult(unique_conversations=[], duplicates=[])

        texts = [self.conversation_text(conversation) for conversation in conversations]
        embeddings = self._encode(texts)
        unique_conversations: list[Conversation] = []
        unique_embeddings: list[list[float]] = []
        duplicates: list[DuplicateMatch] = []

        for conversation, embedding in zip(conversations, embeddings, strict=True):
            match = self._find_duplicate(conversation, embedding, unique_conversations, unique_embeddings)
            if match is not None:
                duplicates.append(match)
                LOGGER.info(
                    "Dropped semantic duplicate duplicate=%s canonical=%s similarity=%.4f",
                    match.duplicate_id,
                    match.canonical_id,
                    match.similarity,
                )
                continue
            unique_conversations.append(conversation)
            unique_embeddings.append(embedding)

        return DeduplicationResult(unique_conversations=unique_conversations, duplicates=duplicates)

    def is_duplicate(self, candidate: Conversation, existing: Sequence[Conversation]) -> DuplicateMatch | None:
        if not existing:
            return None
        texts = [self.conversation_text(conversation) for conversation in [*existing, candidate]]
        embeddings = self._encode(texts)
        candidate_embedding = embeddings[-1]
        return self._find_duplicate(candidate, candidate_embedding, list(existing), embeddings[:-1])

    def conversation_text(self, conversation: Conversation) -> str:
        parts = [
            conversation.task_type,
            conversation.category or "",
            conversation.subcategory or "",
            conversation.intent or "",
            conversation.difficulty,
            conversation.learning_stage or "",
            conversation.language,
        ]
        parts.extend(f"{message.role}: {message.content}" for message in conversation.messages)
        parts.extend(snippet.code for snippet in conversation.code_snippets)
        for message in conversation.messages:
            parts.extend(snippet.code for snippet in message.code_snippets)
        return "\n".join(part for part in parts if part)

    def _find_duplicate(
        self,
        candidate: Conversation,
        candidate_embedding: list[float],
        existing: Sequence[Conversation],
        existing_embeddings: Sequence[list[float]],
    ) -> DuplicateMatch | None:
        best_match: DuplicateMatch | None = None
        for existing_conversation, existing_embedding in zip(existing, existing_embeddings, strict=True):
            similarity = self.cosine_similarity(candidate_embedding, existing_embedding)
            if similarity >= self.threshold and (best_match is None or similarity > best_match.similarity):
                best_match = DuplicateMatch(
                    duplicate_id=candidate.id,
                    canonical_id=existing_conversation.id,
                    similarity=similarity,
                )
        return best_match

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        return [self._vector_to_list(embedding) for embedding in embeddings]

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers is required for semantic deduplication. "
                    "Install with `pip install -e '.[semantic]'`."
                ) from error
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _vector_to_list(self, vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

    def cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)
