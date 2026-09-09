"""Focused regression tests for the core function-calling pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from src.decoder import ConstrainedDecoder
from src.errors import GenerationError, InputError
from src.models import FunctionDefinition, validate_functions


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

    def test_schema_validation_rejects_duplicate_functions(self) -> None:
        function = {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"},
        }
        with self.assertRaises(InputError):
            validate_functions([function, function])

    def test_decoder_builds_schema_cache(self) -> None:
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
            decoder = ConstrainedDecoder(FakeModel(str(vocab_path)), [function])

            self.assertEqual(decoder._cache.allowed_fn, ["add"])
            self.assertEqual(decoder._cache.func_params["add"], 2)
            self.assertEqual(decoder._cache.param_types["add"]["a"], "number")

    def test_decoder_rejects_empty_prompt(self) -> None:
        function = FunctionDefinition.model_validate({
            "name": "ping",
            "description": "Simple function",
            "parameters": {},
            "returns": {"type": "null"},
        })
        with tempfile.TemporaryDirectory() as directory:
            vocab_path = Path(directory) / "vocab.json"
            vocab_path.write_text(json.dumps({"dummy": 0}), encoding="utf-8")
            decoder = ConstrainedDecoder(FakeModel(str(vocab_path)), [function])

            with self.assertRaises(GenerationError):
                decoder.generate("   ")


if __name__ == "__main__":
    unittest.main()
