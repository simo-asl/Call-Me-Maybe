"""Loading and writing JSON files used by the command-line application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.errors import InputError


def read_json(path_text: str) -> Any:
    """Read JSON from an existing regular file with useful error messages."""

    path = Path(path_text)
    if not path.is_file():
        raise InputError(f"Input file does not exist or is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)
    except json.JSONDecodeError as error:
        message = f"Invalid JSON in '{path}' at line {error.lineno}, column {error.colno}."
        raise InputError(message) from error
    except OSError as error:
        raise InputError(f"Cannot read '{path}': {error.strerror or error}") from error


def write_results(path_text: str, results: list[dict[str, Any]]) -> None:
    """Write results as formatted JSON, creating the output directory if needed."""

    path = Path(path_text)
    if path.exists() and not path.is_file():
        raise InputError(f"Output path is not a regular file: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=4, ensure_ascii=False)
            output_file.write("\n")
    except OSError as error:
        raise InputError(f"Cannot write '{path}': {error.strerror or error}") from error
