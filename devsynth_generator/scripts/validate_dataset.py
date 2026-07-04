"""Validate a generated JSONL conversation dataset."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from devsynth_generator.config import configure_logging, load_settings
from devsynth_generator.validator import ConversationValidator

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a JSONL dataset.")
    return parser


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()

    records = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = ConversationValidator().validate_many(records)
    if errors:
        for error in errors:
            LOGGER.error("%s %s: %s", error.record_id, error.field, error.message)
        raise SystemExit(1)
    print(f"Validated {len(records)} records")


if __name__ == "__main__":
    main()
