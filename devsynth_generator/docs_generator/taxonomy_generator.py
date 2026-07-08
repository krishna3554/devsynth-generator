"""Generate docs/taxonomy.md documenting all taxonomy values."""

from __future__ import annotations

from .doc_context import DocContext


def generate_taxonomy(ctx: DocContext) -> str:
    """Produce docs/taxonomy.md from the taxonomy in the documentation context."""
    t = ctx.taxonomy
    sections = [
        f"# Taxonomy",
        "",
        f"Complete taxonomy of values used in the **{ctx.dataset_name}** dataset (v{ctx.version}).",
        "",
        _section("Categories", t.categories,
                 "High-level technical categories that classify each conversation."),
        _section("Subcategories", t.subcategories,
                 "More specific areas within each category."),
        _section("Intents", t.intents,
                 "The user's underlying intent in starting the conversation."),
        _section("Difficulty Levels", t.difficulties,
                 "Task difficulty as perceived by the target developer audience."),
        _section("Learning Stages", t.learning_stages,
                 "Developer experience levels based on the Dreyfus model of skill acquisition."),
        _section("Conversation Lengths", t.conversation_lengths,
                 "Expected number of turns in the conversation."),
        _length_constraints(),
        _section("Programming Languages", t.languages,
                 "Supported programming languages for code examples."),
        _section("Roles", t.roles,
                 "Speaker roles in the conversation."),
        _section("Tools", t.tools,
                 "Development tools that may be referenced in conversations."),
        _section("Interaction Patterns", t.interaction_patterns,
                 "Styles of conversation flow."),
    ]
    return "\n".join(sections) + "\n"


def _section(title: str, values: tuple[str, ...], description: str) -> str:
    """Render a taxonomy section with a bulleted list."""
    if not values:
        return f"## {title}\n\n{description}\n\n*No values defined.*\n"

    items = "\n".join(f"- `{v}`" for v in values)
    return f"## {title}\n\n{description}\n\n{items}\n"


def _length_constraints() -> str:
    return """### Turn Count Constraints

| Length | Min Turns | Max Turns |
|--------|-----------|-----------|
| `short` | 2 | 2 |
| `medium` | 3 | 4 |
| `long` | 5 | 6 |
"""
