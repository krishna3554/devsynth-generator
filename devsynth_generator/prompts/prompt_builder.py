"""Build generation prompts from file templates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Formatter
from typing import Any, Mapping

from devsynth_generator.config import PACKAGE_ROOT

LOGGER = logging.getLogger(__name__)


class PromptTemplateError(ValueError):
    """Raised when a prompt template cannot be rendered."""


class PromptBuilder:
    """Load prompt templates and inject scenario data plus an output schema."""

    def __init__(
        self,
        template_dir: Path | None = None,
        schema_dir: Path | None = None,
        default_template: str = "conversation_seed.txt",
        default_schema: str = "conversation.schema.json",
    ) -> None:
        self.template_dir = template_dir or PACKAGE_ROOT / "prompts"
        self.schema_dir = schema_dir or PACKAGE_ROOT / "schemas"
        self.default_template = default_template
        self.default_schema = default_schema

    def build(
        self,
        scenario: Any,
        *,
        template_name: str | None = None,
        schema_name: str | None = None,
        extra_context: Mapping[str, Any] | None = None,
    ) -> str:
        template = self.load_template(template_name or self.default_template)
        context = self._scenario_context(scenario)
        context["output_schema"] = self.load_schema(schema_name or self.default_schema)
        if extra_context:
            context.update(extra_context)
        self._ensure_required_fields(template, context)
        return template.format(**context)

    def load_template(self, template_name: str) -> str:
        path = self._resolve(self.template_dir, template_name)
        LOGGER.debug("Loading prompt template from %s", path)
        return path.read_text(encoding="utf-8")

    def load_schema(self, schema_name: str) -> str:
        path = self._resolve(self.schema_dir, schema_name)
        LOGGER.debug("Loading output schema from %s", path)
        schema = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(schema, indent=2, sort_keys=True)

    def _scenario_context(self, scenario: Any) -> dict[str, Any]:
        if hasattr(scenario, "to_dict"):
            data = scenario.to_dict()
        elif hasattr(scenario, "model_dump"):
            data = scenario.model_dump(mode="json")
        elif isinstance(scenario, Mapping):
            data = dict(scenario)
        else:
            data = vars(scenario)

        metadata = data.get("metadata") or {}
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump(mode="json")

        context = {
            "id": data.get("id", ""),
            "task_type": data.get("task_type") or data.get("category", ""),
            "category": data.get("category") or data.get("task_type", ""),
            "subcategory": data.get("subcategory", ""),
            "intent": data.get("intent", ""),
            "difficulty": data.get("difficulty", ""),
            "learning_stage": data.get("learning_stage", ""),
            "conversation_length": data.get("conversation_length", ""),
            "language": data.get("language", ""),
            "interaction_pattern": data.get("interaction_pattern", ""),
            "tools": ", ".join(data.get("tools", [])),
            "turn_count": metadata.get("turn_count", ""),
        }
        context["scenario_json"] = json.dumps(data, indent=2, sort_keys=True)
        return context

    def _ensure_required_fields(self, template: str, context: Mapping[str, Any]) -> None:
        missing = sorted(
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name and field_name not in context
        )
        if missing:
            raise PromptTemplateError(f"Missing prompt context fields: {', '.join(missing)}")

    def _resolve(self, root: Path, filename: str) -> Path:
        path = (root / filename).resolve()
        root = root.resolve()
        if root not in path.parents and path != root:
            raise PromptTemplateError(f"Template path escapes root: {filename}")
        if not path.exists():
            raise PromptTemplateError(f"Prompt resource not found: {path}")
        return path
