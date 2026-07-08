"""Generate a Hugging Face–style DATASET_CARD.md."""

from __future__ import annotations

from .doc_context import DocContext


def generate_dataset_card(ctx: DocContext) -> str:
    """Produce a complete DATASET_CARD.md from the documentation context."""
    sections = [
        _yaml_header(ctx),
        _header(ctx),
        _overview(ctx),
        _motivation(),
        _intended_use(),
        _out_of_scope(),
        _dataset_structure(ctx),
        _schema(),
        _supported_tasks(),
        _languages(ctx),
        _categories(ctx),
        _statistics(ctx),
        _generation_methodology(),
        _validation_methodology(),
        _known_limitations(),
        _bias_discussion(),
        _privacy_statement(),
        _ethical_considerations(),
        _citation(ctx),
    ]
    return "\n\n".join(sections) + "\n"


def _yaml_header(ctx: DocContext) -> str:
    langs = ", ".join(ctx.taxonomy.languages)
    return f"""---
annotations_creators:
- machine-generated
language:
- en
license: {ctx.license.lower()}
multilinguality:
- monolingual
pretty_name: {ctx.dataset_name}
size_categories:
- 10K<n<100K
source_datasets:
- original
tags:
- synthetic
- developer-conversations
- code-assistance
- multi-turn
task_categories:
- conversational
- text-generation
---"""


def _header(ctx: DocContext) -> str:
    return f"""# {ctx.dataset_name}

**Version**: {ctx.version}
**Authors**: DevSynth Contributors
**License**: {ctx.license}
**Homepage**: https://github.com/your-org/devsynth-generator
**Repository**: https://github.com/your-org/devsynth-generator"""


def _overview(ctx: DocContext) -> str:
    return f"""{ctx.description}

This dataset contains **{ctx.total_conversations:,}** synthetic multi-turn developer-assistant conversations covering **{ctx.num_categories}** technical categories across **{len(ctx.taxonomy.languages)}** programming languages."""


def _motivation() -> str:
    return """## Motivation

High-quality developer conversation data is essential for training and evaluating code-assistant models. Real developer conversations are scarce, often proprietary, and contain PII. DevSynth addresses this gap by generating realistic, validated, privacy-safe synthetic conversations that cover a wide range of development scenarios."""


def _intended_use() -> str:
    return """## Intended Use

- **Fine-tuning** code-assistant and developer chatbot models
- **Evaluating** conversational AI systems on developer tasks
- **Benchmarking** multi-turn reasoning capabilities
- **Research** on synthetic data generation methodologies
- **Training** instruction-following models for software engineering"""


def _out_of_scope() -> str:
    return """## Out-of-Scope Use

- **Not suitable** for production deployment without human review
- **Not a substitute** for real-world developer interaction data
- **Not validated** for safety-critical or security-sensitive applications
- **Should not** be used to generate misleading or harmful code
- **Not designed** for non-English language tasks"""


def _dataset_structure(ctx: DocContext) -> str:
    lines = ["## Dataset Structure", ""]
    lines.append("| Split | Conversations | Description |")
    lines.append("|-------|--------------|-------------|")
    for name in ("train", "validation", "test"):
        count = ctx.splits.get(name, 0)
        desc = {
            "train": "Training data",
            "validation": "Validation / hyperparameter tuning",
            "test": "Held-out evaluation set",
        }[name]
        lines.append(f"| `{name}` | {count:,} | {desc} |")
    lines.append("")
    lines.append("Each split is stored as a JSONL file with one conversation per line.")
    return "\n".join(lines)


def _schema() -> str:
    return """## Schema

Each record is a JSON object with the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Unique conversation identifier |
| `task_type` | string | ✅ | Type of development task |
| `category` | string | ❌ | Technical category |
| `subcategory` | string | ❌ | Specific subcategory |
| `intent` | string | ❌ | User's intent |
| `difficulty` | string | ✅ | Difficulty level |
| `learning_stage` | string | ❌ | Developer experience level |
| `language` | string | ✅ | Programming language |
| `messages` | array | ✅ | Array of `{role, content}` turns |
| `code_snippets` | array | ❌ | Top-level code samples |
| `tools` | array | ❌ | Referenced tools |
| `metadata` | object | ❌ | Quality scores, provenance |"""


def _supported_tasks() -> str:
    return """## Supported Tasks

- **Conversational code generation**: Multi-turn dialogue producing code solutions
- **Code explanation**: Explaining code behavior, tradeoffs, and best practices
- **Debugging assistance**: Diagnosing and fixing bugs through conversation
- **Code review**: Providing review feedback and improvement suggestions
- **Architecture discussion**: Discussing system design and architectural decisions
- **Test generation**: Creating test cases and testing strategies"""


def _languages(ctx: DocContext) -> str:
    lang_list = "\n".join(f"- `{lang}`" for lang in ctx.taxonomy.languages)
    return f"""## Languages

### Natural Language
- English (en)

### Programming Languages
{lang_list}"""


def _categories(ctx: DocContext) -> str:
    cat_list = "\n".join(f"- `{cat}`" for cat in ctx.taxonomy.categories)
    return f"""## Categories

{cat_list}"""


def _statistics(ctx: DocContext) -> str:
    stats = ctx.statistics
    if not stats:
        return "## Statistics\n\n*Statistics will be available after running the split pipeline.*"

    lines = ["## Statistics", ""]
    lines.append(f"- **Total conversations**: {stats.get('total_conversations', 'N/A'):,}")
    lines.append(f"- **Train size**: {stats.get('train_size', 'N/A'):,}")
    lines.append(f"- **Validation size**: {stats.get('validation_size', 'N/A'):,}")
    lines.append(f"- **Test size**: {stats.get('test_size', 'N/A'):,}")
    lines.append(f"- **Average turns per conversation**: {stats.get('average_turns', 'N/A')}")
    lines.append(f"- **Code snippet coverage**: {stats.get('average_code_snippet_percentage', 'N/A')}%")
    lines.append("")

    cat_dist = stats.get("category_distribution", [])
    if cat_dist:
        lines.append("### Category Distribution")
        lines.append("")
        lines.append("| Category | Count | % |")
        lines.append("|----------|-------|---|")
        for entry in cat_dist:
            lines.append(f"| {entry['value']} | {entry['count']:,} | {entry['percentage']}% |")
        lines.append("")

    diff_dist = stats.get("difficulty_distribution", [])
    if diff_dist:
        lines.append("### Difficulty Distribution")
        lines.append("")
        lines.append("| Difficulty | Count | % |")
        lines.append("|-----------|-------|---|")
        for entry in diff_dist:
            lines.append(f"| {entry['value']} | {entry['count']:,} | {entry['percentage']}% |")
        lines.append("")

    intent_dist = stats.get("intent_distribution", [])
    if intent_dist:
        lines.append("### Intent Distribution")
        lines.append("")
        lines.append("| Intent | Count | % |")
        lines.append("|--------|-------|---|")
        for entry in intent_dist:
            lines.append(f"| {entry['value']} | {entry['count']:,} | {entry['percentage']}% |")

    lang_dist = stats.get("programming_language_distribution", [])
    if lang_dist:
        lines.append("")
        lines.append("### Programming Language Distribution")
        lines.append("")
        lines.append("| Language | Count | % |")
        lines.append("|----------|-------|---|")
        for entry in lang_dist:
            lines.append(f"| {entry['value']} | {entry['count']:,} | {entry['percentage']}% |")

    return "\n".join(lines)


def _generation_methodology() -> str:
    return """## Generation Methodology

1. **Scenario Generation** — Coverage-balanced scenarios sampled from the taxonomy to ensure uniform category, difficulty, and intent distribution.
2. **Prompt Construction** — Structured prompts with JSON output schemas guide the LLM to produce well-formed conversations.
3. **LLM Generation** — Multi-turn conversations generated via the OpenRouter API using the configured model.
4. **Schema Validation** — Every record is validated with Pydantic models and taxonomy constraints.
5. **PII Detection** — Regex-based scanning for emails, phone numbers, SSNs, credit card numbers, and IP addresses.
6. **Semantic Deduplication** — Near-duplicate conversations are detected using sentence-transformer embeddings with configurable cosine similarity thresholds.
7. **Quality Scoring** — An LLM-as-judge evaluates each conversation on technical accuracy, helpfulness, clarity, and realism. Samples below the threshold are rejected.
8. **Stratified Splitting** — The dataset is split into train/validation/test sets while preserving the distribution of category, difficulty, and intent."""


def _validation_methodology() -> str:
    return """## Validation Methodology

- **Pydantic schema validation**: Type constraints, required fields, value ranges
- **Taxonomy validation**: All categorical values checked against allowed sets
- **PII pattern detection**: Common patterns for personal information
- **Conversation length enforcement**: Turn counts match declared conversation length
- **Metadata consistency**: Turn counts and coverage metadata cross-checked
- **Quality evaluation**: 4-dimension LLM scoring with configurable acceptance threshold"""


def _known_limitations() -> str:
    return """## Known Limitations

- Conversations are **synthetic** and may not capture the full complexity of real developer interactions
- Code snippets are **not executed** or verified for correctness beyond structural validation
- The dataset reflects biases present in the underlying LLM used for generation
- Coverage is limited to **5 programming languages** and **7 task categories**
- **English only** — no multilingual support"""


def _bias_discussion() -> str:
    return """## Bias Discussion

As a synthetic dataset generated by an LLM:

- Conversations may reflect **biases inherent in the training data** of the generating model
- Certain **coding patterns and practices** may be overrepresented based on the model's training distribution
- The **difficulty distribution** is controlled by the taxonomy but subjective interpretations may vary
- **Underrepresentation** of niche programming paradigms and domain-specific patterns is likely

Users should be aware of these biases and consider supplementing with real-world data for production use."""


def _privacy_statement() -> str:
    return """## Privacy Statement

- All conversations are **100% synthetically generated** — no real user data is included
- A **PII detection pipeline** scans every conversation for common personal information patterns
- No real names, email addresses, phone numbers, or other personal identifiers are intentionally present
- The dataset is designed to be **privacy-safe by construction**"""


def _ethical_considerations() -> str:
    return """## Ethical Considerations

- This dataset is intended for **research and development** purposes
- Generated code should be **reviewed by humans** before use in production
- The dataset should **not be used** to train models that generate harmful or malicious code
- Users should ensure compliance with applicable **laws and regulations** in their jurisdiction
- The synthetic nature of the data should be **clearly disclosed** in any downstream applications"""


def _citation(ctx: DocContext) -> str:
    return f"""## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{devsynth_{ctx.version.replace('.', '_')},
  title   = {{{ctx.dataset_name}: Synthetic Developer Conversation Dataset}},
  version = {{{ctx.version}}},
  year    = {{{ctx.created_at[:4]}}},
  license = {{{ctx.license}}},
  url     = {{https://github.com/your-org/devsynth-generator}}
}}
```"""
