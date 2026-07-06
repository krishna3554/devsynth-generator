from devsynth_generator.generator import ScenarioGenerator
from devsynth_generator.validator import ValidationPipeline


def valid_record():
    return ScenarioGenerator(seed=1).generate(1)[0].to_dict()


def fields(result):
    return {error.field for error in result.errors}


def test_validation_pipeline_accepts_valid_generated_record():
    record = valid_record()

    result = ValidationPipeline().validate_record(record)

    assert result.is_valid


def test_validation_pipeline_reports_schema_errors():
    record = valid_record()
    record.pop("messages")

    result = ValidationPipeline().validate_record(record)

    assert not result.is_valid
    assert "messages" in fields(result)


def test_validation_pipeline_detects_common_pii_patterns():
    record = valid_record()
    record["messages"][0]["content"] = "Email me at dev@example.com or call 555-123-4567."

    result = ValidationPipeline().validate_record(record)

    messages = [error.message for error in result.errors]
    assert "Possible PII detected: email" in messages
    assert "Possible PII detected: phone" in messages


def test_validation_pipeline_enforces_conversation_length_constraints():
    record = valid_record()
    record["conversation_length"] = "short"
    record["messages"].append({"role": "user", "content": "One more turn."})

    result = ValidationPipeline().validate_record(record)

    assert "messages" in fields(result)


def test_validation_pipeline_verifies_metadata_turn_count():
    record = valid_record()
    record["metadata"]["turn_count"] = 99

    result = ValidationPipeline().validate_record(record)

    assert "metadata.turn_count" in fields(result)


def test_validation_pipeline_verifies_metadata_coverage_consistency():
    record = valid_record()
    record["metadata"]["coverage"]["category"] = "code_review"

    result = ValidationPipeline().validate_record(record)

    assert "metadata.coverage.category" in fields(result)


def test_validation_pipeline_verifies_generator_coverage_consistency():
    record = valid_record()
    record["generator"]["parameters"]["coverage_matrix"]["difficulty"] = "hard"

    result = ValidationPipeline().validate_record(record)

    assert "generator.parameters.coverage_matrix.difficulty" in fields(result)
