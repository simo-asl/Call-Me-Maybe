"""Command-line entry point for the Call Me Maybe program."""

import argparse
import time
import json
import os

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.constrained_llm import LLM
from src.function_caller import CallMeMaybe
from src.token_encoder import Encoder
from src.parser import load_prompts


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
ALLOWED_MODELS = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B",
)


def parse_args() -> argparse.Namespace:
    """Parse model, function-definition, input, and output options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--functions_definition',
        default='data/input/functions_definition.json'
    )
    parser.add_argument(
        '--input',
        default='data/input/function_calling_tests.json'
    )
    parser.add_argument(
        '--output',
        default='data/output/function_calling_results.json'
    )
    parser.add_argument(
        '--model',
        choices=ALLOWED_MODELS,
        default=DEFAULT_MODEL,
        help=f'Hugging Face model name (default: {DEFAULT_MODEL})'
    )
    return parser.parse_args()


def create_encoder(vocab_path: str) -> Encoder:
    """Load the vocabulary JSON and create the project encoder."""
    with open(vocab_path, 'r', encoding='utf-8') as file:
        tokens = json.load(file)
    return Encoder(tokens)


if __name__ == "__main__":
    try:
        args = parse_args()
        llm_model = Small_LLM_Model(model_name=args.model)
        encoder = create_encoder(llm_model.get_path_to_vocab_file())
        llm = LLM(llm_model, encoder)
        call_me_maybe = CallMeMaybe(llm, args.functions_definition)

        prompts = load_prompts(args.input)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = []

        start_time = time.perf_counter()
        for index, prompt in enumerate(prompts):
            print(f"\n{index}. Processing {prompt!r}...")

            result = call_me_maybe.process_func(prompt)
            results.append(json.loads(result))

        with open(args.output, 'w', encoding='utf-8') as output:
            json.dump(
                results,
                output,
                ensure_ascii=False,
                indent=2,
            )
        elapsed = time.perf_counter() - start_time
        print('Finished.')
        print(f"Generation time: {elapsed:.2f} seconds")

    except FileNotFoundError as error:
        print(f"File not found: {error.filename}")

    except json.JSONDecodeError as error:
        print(
            f"Error decoding JSON: {error.msg} "
            f"at line {error.lineno} column {error.colno}"
        )

    except Exception as error:
        print(f"An unexpected error occurred: {str(error)}")
