"""Generate docs/schema.md documenting every dataset field."""

from __future__ import annotations

from .doc_context import DocContext

# Field definitions for the conversation schema.
# Each tuple: (field_name, type, required, description, example)
CONVERSATION_FIELDS = [
    ("id", "string", True, "Unique identifier for the conversation.", '"conv-00001-a1b2c3d4"'),
    ("task_type", "string", True, "Type of development task being addressed.", '"debugging_session"'),
    ("category", "string | null", False, "Technical category of the conversation.", '"bug_fix"'),
    ("subcategory", "string | null", False, "Specific subcategory within the category.", '"api_design"'),
    ("intent", "string | null", False, "The user's underlying intent.", '"ask_for_implementation"'),
    ("difficulty", "string", True, "Difficulty level of the task.", '"medium"'),
    ("learning_stage", "string | null", False, "Developer experience level.", '"competent"'),
    ("conversation_length", "string | null", False, "Expected conversation length.", '"medium"'),
    ("language", "string", True, "Primary programming language.", '"python"'),
    ("messages", "array[Message]", True, "Ordered list of conversation turns.", "See Message schema below"),
    ("code_snippets", "array[CodeSnippet]", False, "Top-level code samples attached to the scenario.", "See CodeSnippet schema below"),
    ("tools", "array[string]", False, "Tools referenced in the conversation.", '["git", "shell"]'),
    ("interaction_pattern", "string | null", False, "Conversation interaction style.", '"multi_turn_debugging"'),
    ("metadata", "ConversationMetadata", False, "Quality scores, provenance, and timestamps.", "See Metadata schema below"),
    ("generator", "GeneratorInfo | null", False, "Information about the generator that produced this record.", "See GeneratorInfo schema below"),
]

MESSAGE_FIELDS = [
    ("role", "string", True, 'The speaker role: "user" or "assistant".', '"user"'),
    ("content", "string", True, "The text content of this turn.", '"How do I fix this bug?"'),
    ("code_snippets", "array[CodeSnippet]", False, "Code samples attached to this message.", "See CodeSnippet schema below"),
]

CODE_SNIPPET_FIELDS = [
    ("language", "string", True, "Programming language of the snippet.", '"python"'),
    ("code", "string", True, "The code content.", '"def hello(): ..."'),
    ("filename", "string | null", False, "Optional filename for the snippet.", '"utils.py"'),
    ("purpose", "string | null", False, "Description of what the snippet demonstrates.", '"Fix for the null check"'),
]

METADATA_FIELDS = [
    ("source", "string", False, "How the conversation was generated.", '"synthetic"'),
    ("turn_count", "integer | null", False, "Number of turns in the conversation.", "4"),
    ("tags", "array[string]", False, "Descriptive tags.", '["python", "debugging"]'),
    ("quality_score", "float | null", False, "Overall quality score (0.0–1.0).", "0.85"),
    ("created_at", "ISO-8601 datetime", False, "Timestamp of creation.", '"2025-01-15T12:00:00Z"'),
]


def generate_schema(ctx: DocContext) -> str:
    """Produce docs/schema.md documenting every field in the dataset."""
    sections = [
        "# Dataset Schema",
        "",
        f"Complete field documentation for the **{ctx.dataset_name}** dataset (v{ctx.version}).",
        "",
        "## Conversation",
        "",
        "The top-level record representing a single developer-assistant conversation.",
        "",
        _render_table(CONVERSATION_FIELDS),
        "",
        "## Message",
        "",
        "A single turn in the conversation.",
        "",
        _render_table(MESSAGE_FIELDS),
        "",
        "## CodeSnippet",
        "",
        "A code sample attached to a conversation or message.",
        "",
        _render_table(CODE_SNIPPET_FIELDS),
        "",
        "## ConversationMetadata",
        "",
        "Provenance and quality metadata. This object allows additional fields beyond those listed.",
        "",
        _render_table(METADATA_FIELDS),
        "",
        "## Field Value Constraints",
        "",
        "- All `string` fields marked as required must be **non-empty** after trimming whitespace.",
        "- `messages` must contain **at least one** message.",
        "- `quality_score` must be between **0.0** and **1.0** inclusive.",
        "- `turn_count` must be a **positive integer** (≥ 1).",
        f"- See [taxonomy.md](taxonomy.md) for allowed values of categorical fields.",
    ]
    return "\n".join(sections) + "\n"


def _render_table(fields: list[tuple[str, str, bool, str, str]]) -> str:
    """Render a field documentation table."""
    lines = [
        "| Field | Type | Required | Description | Example |",
        "|-------|------|----------|-------------|---------|",
    ]
    for name, type_, required, desc, example in fields:
        req = "✅" if required else "❌"
        lines.append(f"| `{name}` | `{type_}` | {req} | {desc} | {example} |")
    return "\n".join(lines)
