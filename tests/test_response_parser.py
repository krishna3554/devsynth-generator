import logging

import pytest
from pydantic import BaseModel, Field

from devsynth_generator.parser import ModelResponseParser, ResponseParseError


class ParsedExample(BaseModel):
    name: str
    count: int = Field(..., ge=1)


def test_response_parser_extracts_fenced_json():
    text = """Here is the result:

```json
{"name": "alpha", "count": 2}
```
"""

    parsed = ModelResponseParser().parse_model(text, ParsedExample)

    assert parsed == ParsedExample(name="alpha", count=2)


def test_response_parser_extracts_balanced_json_from_text():
    text = 'prefix {"name": "alpha with } in string", "count": 3} suffix'

    parsed = ModelResponseParser().parse_model(text, ParsedExample)

    assert parsed.count == 3


def test_response_parser_retries_generation_and_logs_failures(caplog):
    responses = iter(["not json", '{"name": "alpha", "count": 2}'])
    parser = ModelResponseParser(max_parse_retries=1)

    with caplog.at_level(logging.WARNING):
        parsed = parser.parse_with_retries(lambda: next(responses), ParsedExample)

    assert parsed.name == "alpha"
    assert "Model response parse failed attempt=1/2" in caplog.text


def test_response_parser_retries_pydantic_validation_failures(caplog):
    responses = iter(['{"name": "alpha", "count": 0}', '{"name": "alpha", "count": 1}'])
    parser = ModelResponseParser(max_parse_retries=1)

    with caplog.at_level(logging.WARNING):
        parsed = parser.parse_with_retries(lambda: next(responses), ParsedExample)

    assert parsed.count == 1
    assert "Model response parse failed attempt=1/2" in caplog.text


def test_response_parser_raises_after_retry_budget():
    parser = ModelResponseParser(max_parse_retries=1)

    with pytest.raises(ResponseParseError):
        parser.parse_with_retries(lambda: "not json", ParsedExample)
