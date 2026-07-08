"""Generate CHANGELOG.md from dataset version information."""

from __future__ import annotations

from .doc_context import DocContext


def generate_changelog(ctx: DocContext) -> str:
    """Produce a CHANGELOG.md from the documentation context."""
    total = ctx.total_conversations
    num_cats = ctx.num_categories

    return f"""# Changelog

All notable changes to the {ctx.dataset_name} dataset will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## v{ctx.version} — {ctx.created_at[:10]}

### Added
- Initial release of the {ctx.dataset_name} dataset
- {total:,} synthetic developer-assistant conversations
- {num_cats} technical categories: {', '.join(f'`{c}`' for c in ctx.taxonomy.categories)}
- {len(ctx.taxonomy.languages)} programming languages: {', '.join(f'`{l}`' for l in ctx.taxonomy.languages)}
- {len(ctx.taxonomy.difficulties)} difficulty levels: {', '.join(f'`{d}`' for d in ctx.taxonomy.difficulties)}
- Multi-turn conversation format with code snippets
- Pydantic schema validation pipeline
- PII detection (email, phone, SSN, credit card, IPv4)
- Semantic deduplication via sentence-transformers
- LLM-based quality evaluation (accuracy, helpfulness, clarity, realism)
- Stratified train / validation / test split
- Comprehensive dataset statistics
- Publication-ready documentation (README, Dataset Card, Schema, Taxonomy)
- CITATION.cff for machine-readable citation
- Apache-2.0 license
"""
