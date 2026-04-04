#!/usr/bin/env python3
"""Validate funding.json against the schema declared in its $schema field."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.validators import validator_for


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_remote_json(url: str) -> object:
    with urllib.request.urlopen(url) as response:  # noqa: S310
        return json.load(response)


def validate_file(path: Path) -> int:
    try:
        instance = load_json(path)
    except json.JSONDecodeError as exc:
        print(f"{path}: invalid JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(instance, dict):
        print(f"{path}: top-level value must be a JSON object", file=sys.stderr)
        return 1

    schema_url = instance.get("$schema")
    if not isinstance(schema_url, str) or not schema_url:
        print(f"{path}: missing or invalid $schema URL", file=sys.stderr)
        return 1

    try:
        schema = load_remote_json(schema_url)
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        validator.validate(instance)
    except OSError as exc:
        print(f"{path}: failed to load schema {schema_url}: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(
            f"{path}: schema at {schema_url} did not return valid JSON: {exc}",
            file=sys.stderr,
        )
        return 1
    except jsonschema_exceptions.SchemaError as exc:
        print(f"{path}: invalid JSON Schema from {schema_url}: {exc}", file=sys.stderr)
        return 1
    except jsonschema_exceptions.ValidationError as exc:
        location = "$"
        if exc.absolute_path:
            location = "$." + ".".join(str(part) for part in exc.absolute_path)
        print(f"{path}: schema validation failed at {location}: {exc.message}", file=sys.stderr)
        return 1

    print(f"{path}: schema validation passed")
    return 0


def main(argv: list[str]) -> int:
    filenames = argv[1:] or ["funding.json"]
    status = 0
    for filename in filenames:
        status |= validate_file(Path(filename))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
