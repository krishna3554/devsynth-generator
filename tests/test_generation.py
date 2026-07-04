from devsynth_generator.generator import ConversationGenerator
from devsynth_generator.validator import ConversationValidator


def test_generated_conversations_are_valid():
    conversations = ConversationGenerator(seed=7).generate(3)
    records = [conversation.to_dict() for conversation in conversations]

    errors = ConversationValidator().validate_many(records)

    assert errors == []
