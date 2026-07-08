"""Generate CITATION.cff for machine-readable citation."""

from __future__ import annotations

from .doc_context import DocContext


def generate_citation(ctx: DocContext) -> str:
    """Produce a CITATION.cff from the documentation context."""
    return f"""cff-version: 1.2.0
title: "{ctx.dataset_name}: Synthetic Developer Conversation Dataset"
message: "If you use this dataset, please cite it using the metadata from this file."
type: dataset
version: "{ctx.version}"
date-released: "{ctx.created_at[:10]}"
license: "{ctx.license}"
url: "https://github.com/your-org/devsynth-generator"
repository-code: "https://github.com/your-org/devsynth-generator"
abstract: "{ctx.description}"
keywords:
  - synthetic-data
  - developer-conversations
  - code-assistance
  - multi-turn-dialogue
  - machine-learning
authors:
  - name: "DevSynth Contributors"
"""
