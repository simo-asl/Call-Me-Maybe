"""Command-line interface for the Call Me Maybe project."""

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.errors import CallMeMaybeError
from src.input_loader import read_json, write_results
from src.models import validate_functions, validate_prompts
from src.printing import configure_output
from src.service import generate_calls


def parse_arguments() -> argparse.Namespace:
    """Parse the supported command-line options.

    Returns:
        Namespace containing input, output, model, and display settings.
    """

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
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument(
        "--visual",
        action="store_true",
        help="Show the constrained generation process step by step.",
    )
    diagnostics.add_argument(
        "--debug_tokens",
        action="store_true",
        help="Show compact token-selection diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    """Run generation, write results, and report expected errors to stderr.

    Raises:
        SystemExit: With status 1 when an expected input or generation error
            occurs.
    """

    arguments = parse_arguments()
    configure_output(
        visual=arguments.visual,
        debug_tokens=arguments.debug_tokens,
    )
    started_at = perf_counter()
    try:
        functions = validate_functions(
            read_json(arguments.functions_definition)
        )
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
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    elapsed = perf_counter() - started_at
    print(f"Wrote {len(results)} result(s) in {elapsed:.1f} seconds.")
