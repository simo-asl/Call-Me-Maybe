"""Command-line interface for the Call Me Maybe project."""

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.errors import CallMeMaybeError
from src.input_loader import read_json, write_results
from src.models import validate_functions, validate_prompts
from src.service import generate_calls


def parse_arguments() -> argparse.Namespace:
    """Parse supported project command-line options."""

    parser = argparse.ArgumentParser(
        prog="Call Me Maybe",
        description="Generate schema-constrained LLM function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="JSON file containing function definitions.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="JSON file containing input prompts.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path for the generated result JSON.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model identifier accepted by the provided llm_sdk.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the application and convert expected failures to a clear exit status."""

    arguments = parse_arguments()
    started_at = perf_counter()
    try:
        functions = validate_functions(
            read_json(arguments.functions_definition))
        prompts = validate_prompts(read_json(arguments.input))
        model = Small_LLM_Model(model_name=arguments.model)
        results = generate_calls(model, functions, prompts)
        write_results(arguments.output, results)
    except CallMeMaybeError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except (OSError, RuntimeError, ValueError) as error:
        print(
                f"Error while loading or running the LLM: {error}",
                file=sys.stderr
                )
        raise SystemExit(1) from error
    elapsed = perf_counter() - started_at
    print(f"Wrote {len(results)} result(s) in {elapsed:.1f} seconds.")
