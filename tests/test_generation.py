from collections import Counter

from devsynth_generator.generator import ConversationGenerator, ScenarioGenerator
from devsynth_generator.models import CodeSnippet, Conversation, ConversationDataset, GeneratorInfo, Message
from devsynth_generator.prompts import PromptBuilder, PromptTemplateError
from devsynth_generator.taxonomy import default_taxonomy
from devsynth_generator.validator import ConversationValidator


def test_generated_conversations_are_valid():
    conversations = ConversationGenerator(seed=7).generate(3)
    records = [conversation.to_dict() for conversation in conversations]

    errors = ConversationValidator().validate_many(records)

    assert errors == []


def test_validator_rejects_unknown_taxonomy_values():
    record = ConversationGenerator(seed=7).generate_one().to_dict()
    record["task_type"] = "not_real"
    record["difficulty"] = "legendary"
    record["language"] = "cobol"
    record["messages"][0]["role"] = "critic"
    record["tools"] = ["shell", "unknown_tool"]
    record["interaction_pattern"] = "unknown_pattern"

    errors = ConversationValidator().validate_record(record)

    assert {error.field for error in errors} == {
        "task_type",
        "difficulty",
        "language",
        "messages[0].role",
        "tools[1]",
        "interaction_pattern",
    }


def test_pydantic_models_support_code_and_generator_metadata():
    conversation = Conversation(
        id="conv-test",
        task_type="bug_fix",
        difficulty="easy",
        language="python",
        tools=["shell"],
        interaction_pattern="implementation_with_tests",
        messages=[
            Message(
                role="assistant",
                content="Patch the failing branch.",
                code_snippets=[
                    CodeSnippet(language="python", filename="app.py", code="print('ok')", purpose="example")
                ],
            )
        ],
        generator=GeneratorInfo(seed=123, prompt_template="conversation_seed.txt"),
    )
    dataset = ConversationDataset(conversations=[conversation], generator=GeneratorInfo(seed=123))

    assert conversation.metadata.turn_count == 1
    assert dataset.validate_taxonomy(default_taxonomy()) == {}
    assert dataset.to_dict()["conversations"][0]["generator"]["seed"] == 123


def test_dataset_taxonomy_validation_reports_code_snippet_languages():
    conversation = Conversation(
        id="conv-bad-code",
        task_type="bug_fix",
        difficulty="easy",
        language="python",
        messages=[
            Message(
                role="assistant",
                content="Here is a patch.",
                code_snippets=[CodeSnippet(language="brainfuck", code="+++")],
            )
        ],
    )

    errors = ConversationDataset(conversations=[conversation]).validate_taxonomy(default_taxonomy())

    assert errors == {"conv-bad-code": ["messages[0].code_snippets[0].language"]}


def test_scenario_generator_evenly_distributes_coverage_dimensions():
    taxonomy = default_taxonomy()
    conversations = ScenarioGenerator(taxonomy=taxonomy, seed=11).generate(84)

    for field in (
        "category",
        "subcategory",
        "intent",
        "difficulty",
        "learning_stage",
        "conversation_length",
    ):
        counts = Counter(getattr(conversation, field) for conversation in conversations)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_scenario_generator_outputs_valid_conversations():
    conversations = ScenarioGenerator(seed=11).generate(12)
    records = [conversation.to_dict() for conversation in conversations]

    errors = ConversationValidator().validate_many(records)

    assert errors == []
    assert {conversation.metadata.turn_count for conversation in conversations} <= {2, 4, 6}


def test_prompt_builder_injects_scenario_and_schema():
    scenario = ScenarioGenerator(seed=11).generate(1)[0]

    prompt = PromptBuilder().build(scenario)

    assert f"Category: {scenario.category}" in prompt
    assert f"Intent: {scenario.intent}" in prompt
    assert '"title": "SyntheticDeveloperConversation"' in prompt
    assert "{output_schema}" not in prompt


def test_prompt_builder_rejects_missing_template_fields(tmp_path):
    template_dir = tmp_path / "prompts"
    schema_dir = tmp_path / "schemas"
    template_dir.mkdir()
    schema_dir.mkdir()
    (template_dir / "bad.txt").write_text("Unknown: {missing_field}", encoding="utf-8")
    (schema_dir / "schema.json").write_text('{"type": "object"}', encoding="utf-8")

    builder = PromptBuilder(template_dir=template_dir, schema_dir=schema_dir)

    try:
        builder.build({"task_type": "bug_fix"}, template_name="bad.txt", schema_name="schema.json")
    except PromptTemplateError as error:
        assert "missing_field" in str(error)
    else:
        raise AssertionError("Expected PromptTemplateError")
