"""Small regression tests for input validation and JSON file handling."""

import json
import tempfile
import unittest
from pathlib import Path

from src.errors import InputError
from src.input_loader import read_json, write_results
from src.models import validate_functions, validate_prompts


class BasicProjectTests(unittest.TestCase):
    """Cover stable validation and file-handling behavior."""

    def test_validate_prompts_accepts_valid_input(self) -> None:
        prompts = validate_prompts([{"prompt": "Add 2 and 3"}])
        self.assertEqual(prompts[0].prompt, "Add 2 and 3")

    def test_validate_prompts_rejects_duplicates(self) -> None:
        with self.assertRaises(InputError):
            validate_prompts([{"prompt": "hello"}, {"prompt": "hello"}])

    def test_validate_functions_rejects_duplicate_names(self) -> None:
        function = {
            "name": "add",
            "description": "Add numbers",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"},
        }
        with self.assertRaises(InputError):
            validate_functions([function, function])

    def test_read_json_rejects_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaises(InputError):
                read_json(str(path))

    def test_write_and_read_results(self) -> None:
        results = [
            {"prompt": "Add 2 and 3", "name": "add", "parameters": {"a": 2}}
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output" / "results.json"
            write_results(str(path), results)
            self.assertEqual(read_json(str(path)), results)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), results)


if __name__ == "__main__":
    unittest.main()
