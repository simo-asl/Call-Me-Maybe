"""Focused regression tests for the core function-calling pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from src.decoder import ConstrainedDecoder
from src.errors import GenerationError, InputError
from src.models import FunctionDefinition, validate_functions
from src.printing import Colors


class FakeModel:
    """Minimal model interface used to test decoder setup without inference."""

    def __init__(self, vocab_path: str) -> None:
        self.vocab_path = vocab_path

    def encode(self, text: str) -> list[int]:
        return [0]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        return [0.0] * 8

    def get_path_to_vocab_file(self) -> str:
        return self.vocab_path


class CoreProjectTests(unittest.TestCase):
    """Verify the stable guarantees provided by the project."""

    def _title(self, text: str) -> None:
        print(f"\n{Colors.YELLOW}=== {text} ==={Colors.RESET}")

    def _pass(self) -> None:
        print(f"{Colors.GREEN}[PASS]{Colors.RESET}")

    def test_schema_validation_rejects_duplicate_functions(self) -> None:
        self._title("TEST 3: SCHEMA VALIDATION")
        print("Input: two function definitions named 'add'")
        print("Expected: duplicate function names must be rejected")

        function = {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"},
        }
        with self.assertRaises(InputError):
            validate_functions([function, function])

        print("Result: InputError raised correctly")
        self._pass()

    def test_decoder_builds_schema_cache(self) -> None:
        self._title("TEST 1: CONSTRAINED DECODER CACHE")
        print("Function: add")
        print("Schema: a -> number, b -> number")
        print("Action: build ConstrainedDecoder cache from the schema")

        function = FunctionDefinition.model_validate({
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "returns": {"type": "number"},
        })
        with tempfile.TemporaryDirectory() as directory:
            vocab_path = Path(directory) / "vocab.json"
            vocab_path.write_text(
                json.dumps({"dummy": 0, "{": 1, "}": 2, '"': 3,
                            "name": 4, "parameters": 5, "0": 6, ".": 7}),
                encoding="utf-8",
            )
            decoder = ConstrainedDecoder(
                FakeModel(str(vocab_path)), [function])

            self.assertEqual(decoder._cache.allowed_fn, ["add"])
            self.assertEqual(decoder._cache.func_params["add"], 2)
            self.assertEqual(decoder._cache.param_types["add"]["a"], "number")
            self.assertEqual(decoder._cache.param_types["add"]["b"], "number")

            print("Decoder detected:")
            print(f"  allowed function : {decoder._cache.allowed_fn[0]}")
            print(f"  parameter count  : {decoder._cache.func_params['add']}")
            print("  parameter a type : number")
            print("  parameter b type : number")

        print("Result: decoder cache matches the function schema")
        self._pass()

    def test_decoder_rejects_empty_prompt(self) -> None:
        self._title("TEST 2: EMPTY PROMPT SAFETY")
        print("Input prompt: whitespace only")
        print("Expected: controlled GenerationError, no unexpected crash")

        function = FunctionDefinition.model_validate({
            "name": "ping",
            "description": "Simple function",
            "parameters": {},
            "returns": {"type": "null"},
        })
        with tempfile.TemporaryDirectory() as directory:
            vocab_path = Path(directory) / "vocab.json"
            vocab_path.write_text(json.dumps({"dummy": 0}), encoding="utf-8")
            decoder = ConstrainedDecoder(
                FakeModel(str(vocab_path)), [function])

            with self.assertRaises(GenerationError):
                decoder.generate("   ")

        print("Result: GenerationError raised correctly")
        self._pass()


if __name__ == "__main__":
    unittest.main(verbosity=2)
