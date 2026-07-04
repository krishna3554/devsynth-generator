from devsynth_generator.generator import ConversationGenerator
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
