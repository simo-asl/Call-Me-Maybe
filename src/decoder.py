"""Token-by-token constrained decoding for function-call JSON."""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Protocol
from src.printing import GenerationVisualizer

import numpy as np

from src.errors import GenerationError, InputError
from src.models import FunctionDefinition


class LlmModel(Protocol):
    """Public subset of ``llm_sdk.Small_LLM_Model`` used by this project."""

    def encode(self, text: str) -> object:
        """Encode text as a tensor-like value containing input token ids."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for the supplied token ids."""

    def get_path_to_vocab_file(self) -> str:
        """Return the path of the tokenizer vocabulary JSON file."""


@dataclass
class MaskCache:
    """Cache structure matching the original project logic."""

    raw_functions: list[dict[str, object]]
    allowed_fn: list[str]
    func_params: dict[str, int]
    param_types: dict[str, dict[str, object]]
    vocab_dict: dict[int, str]
    clean_dict_items: list[tuple[int, str]]
    mini_dict: list[tuple[int, str]]
    p4_mask: np.ndarray
    p4_numbers_only: np.ndarray
    p4_no_comma: np.ndarray


class ConstrainedDecoder:
    """Generate valid JSON function calls using token-level constraints."""

    def __init__(
        self,
        model: LlmModel,
        functions: list[FunctionDefinition],
    ) -> None:
        self._model = model
        self._functions = functions
        self._cache = self._build_cache()
        self._visualizer = GenerationVisualizer()

    def _build_cache(self) -> MaskCache:
        """Pre-compute masks and dictionaries exactly like original logic."""
        try:
            with open(
                self._model.get_path_to_vocab_file(),
                encoding="utf-8",
            ) as vocab_file:
                raw_vocab = json.load(vocab_file)
        except (OSError, json.JSONDecodeError) as error:
            raise InputError(
                f"Cannot load model vocabulary: {error}"
            ) from error

        vocab_dict: dict[int, str] = {
            v: k.replace("Ġ", " ")
            for k, v in raw_vocab.items()
            if isinstance(v, int)
        }

        printable_set = set(string.printable)
        valid_ids = [
            token_id
            for token_id, token_str in vocab_dict.items()
            if token_str and all(c in printable_set for c in token_str)
        ]

        clean_dict_items = [
            (k, v)
            for k, v in vocab_dict.items()
            if v and all(c in printable_set for c in v)
        ]

        raw_functions = [
            {
                "name": f.name,
                "description": f.description,
                "parameters": {
                    k: v.model_dump() for k, v in f.parameters.items()
                },
            }
            for f in self._functions
        ]

        allowed_fn = [f.name for f in self._functions]
        func_params: dict[str, int] = {}
        param_types: dict[str, dict[str, object]] = {}

        for fn in raw_functions:
            name = str(fn["name"])
            params = fn.get("parameters", {})
            if isinstance(params, dict):
                func_params[name] = len(params)
                param_types[name] = {}
                for param_key, details in params.items():
                    if isinstance(details, dict):
                        param_types[name][param_key] = details.get("type")

        dummy_ids = self._encode("dummy")
        vocab_size = len(
            self._model.get_logits_from_input_ids(dummy_ids)
        )

        p4_mask = np.zeros(vocab_size, dtype=bool)
        p4_mask[valid_ids] = True

        p4_numbers_only = np.zeros(vocab_size, dtype=bool)
        allowed_math_chars = set("0123456789.-, }")
        for i, s in clean_dict_items:
            if all(char in allowed_math_chars for char in s) or s == "null":
                p4_numbers_only[i] = True

        p4_no_comma = p4_mask.copy()
        for i, s in clean_dict_items:
            if "," in s:
                p4_no_comma[i] = False

        target_phrases = allowed_fn + ['{"name":"', '","parameters":{', "}"]
        mini_dict = [
            (i, s)
            for i, s in clean_dict_items
            if any(s in phrase for phrase in target_phrases)
        ]

        return MaskCache(
            raw_functions=raw_functions,
            allowed_fn=allowed_fn,
            func_params=func_params,
            param_types=param_types,
            vocab_dict=vocab_dict,
            clean_dict_items=clean_dict_items,
            mini_dict=mini_dict,
            p4_mask=p4_mask,
            p4_numbers_only=p4_numbers_only,
            p4_no_comma=p4_no_comma,
        )

    def generate(self, user_prompt: str) -> dict[str, object]:
        """Generate one structurally valid function call matching logic."""
        if not user_prompt.strip():
            raise GenerationError("User prompt cannot be empty.")

        optimized_schemas = []
        for f in self._cache.raw_functions:
            optimized_schemas.append(
                {
                    "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": f.get("parameters", {}),
                }
            )

        schema_hints = json.dumps(
            optimized_schemas,
            separators=(",", ":"),
        )

        prompt = (
            f"System: You are a strict API. Output ONLY valid JSON matching "
            f"these schemas: {schema_hints}\n"
            r"Rule: For the regex field, NEVER output literal matches. "
            r"Always use proper regex sets "
            r"(e.g. '[aeiouAEIOU]', '[0-9]+', '\\bword\\b'). "
            r"For replacement, if asked for a character (e.g. asterisks), "
            r"output EXACTLY ONE character (e.g. '*')."
            "\n"
            f"User: {user_prompt}\n"
            "Tool Call: "
        )

        input_ids = self._encode(prompt)
        vocab_size = len(
            self._model.get_logits_from_input_ids(input_ids)
        )

        prefix = '{"name":"'
        current_str = prefix
        input_ids.extend(self._encode(prefix))
        bridge_injected = False
        max_tokens = 150

        step = 1
        while (
            not current_str.replace(" ", "").replace("\n", "").endswith("}}")
            and len(input_ids) < len(prompt) + max_tokens
        ):
            # Teleport
            if prefix in current_str and '","parameters":{' not in current_str:
                after_prefix = current_str.split(prefix)[1]
                possible_names = [
                    n
                    for n in self._cache.allowed_fn
                    if n.startswith(after_prefix)
                ]

                if (
                    len(possible_names) == 1
                    and possible_names[0] != after_prefix
                ):
                    remainder = possible_names[0][len(after_prefix):] + '"'
                    current_str += remainder
                    input_ids.extend(self._encode(remainder))
                    continue

            # Bridge Fast-Forward
            if (
                current_str.endswith('"')
                and not bridge_injected
                and prefix in current_str
                and len(current_str) > len(prefix)
            ):
                bridge = ',"parameters":{'
                current_str += bridge
                input_ids.extend(self._encode(bridge))
                bridge_injected = True

                func_name = current_str.split('"name":"')[1].split('"')[0]
                if self._cache.func_params.get(func_name, 99) == 0:
                    current_str += "}}"
                    break

                active_schema = next(
                    (
                        f
                        for f in self._cache.raw_functions
                        if f["name"] == func_name
                    ),
                    None,
                )

                if active_schema:
                    tiny_schema = json.dumps(
                        [
                            {
                                "name": active_schema["name"],
                                "description": active_schema.get(
                                    "description",
                                    "",
                                ),
                                "parameters": active_schema.get(
                                    "parameters",
                                    {},
                                ),
                            }
                        ],
                        separators=(",", ":"),
                    )

                    tiny_prompt = (
                        f"System: Output valid JSON matching this schema: "
                        f"{tiny_schema}\n"
                        r"Rule: For the regex field, NEVER output "
                        r"literal matches. "
                        r'Always use proper regex sets '
                        r'(e.g. "[aeiouAEIOU]", "[0-9]+", "\\bword\\b"). '
                        r"For replacement, if asked for a character "
                        r"(e.g. asterisks), output EXACTLY ONE character "
                        r"(e.g. '*')."
                        "\n"
                        f"User: {user_prompt}\n"
                        f"Tool Call: {current_str}"
                    )

                    input_ids = self._encode(tiny_prompt)
                else:
                    input_ids.extend(self._encode(bridge))

                continue

            rules = self._get_allowed_chars(
                current_str,
                self._cache.allowed_fn,
            )
            logits = np.array(
                self._model.get_logits_from_input_ids(input_ids)
            )
            mask = np.zeros(vocab_size, dtype=bool)

            if len(rules) > 10:
                # PHASE 4: THE QUOTA & TYPE SHIELD
                match = re.search(
                    r'"name"\s*:\s*"([^"]+)',
                    current_str,
                )
                func_name = match.group(1) if match else ""

                params_str = (
                    current_str.split('"parameters"')[1]
                    if '"parameters"' in current_str
                    else ""
                )

                if params_str:
                    in_string = False
                    last_structural_colon = -1
                    last_structural_comma = -1
                    last_structural_brace = -1

                    for i, char in enumerate(params_str):
                        if char == '"':
                            if i == 0 or params_str[i - 1] != "\\":
                                in_string = not in_string
                        elif not in_string:
                            if char == ":":
                                last_structural_colon = i
                            elif char == ",":
                                last_structural_comma = i
                            elif char == "}":
                                last_structural_brace = i

                    is_inside_value = (
                        last_structural_colon > last_structural_comma
                        and last_structural_colon > last_structural_brace
                    )

                    active_key = ""
                    if is_inside_value:
                        keys_found = re.findall(
                            r'"([^"]+)"\s*:',
                            params_str,
                        )
                        if keys_found:
                            active_key = keys_found[-1]

                    expected_type = self._cache.param_types.get(
                        func_name,
                        {},
                    ).get(active_key, "Any")

                    param_count = len(
                        re.findall(r'"([^"]+)"\s*:', params_str)
                    )
                    target_count = self._cache.func_params.get(
                        func_name,
                        99,
                    )

                    # THE MASK ROUTER
                    if is_inside_value and expected_type == "number":
                        mask = self._cache.p4_numbers_only.copy()

                        current_value = params_str[
                            last_structural_colon + 1:
                        ].strip()

                        has_decimal_point = "." in current_value

                        if not has_decimal_point:
                            for i, s in self._cache.clean_dict_items:
                                if "," in s or "}" in s:
                                    mask[i] = False

                        elif param_count == target_count:
                            for i, s in self._cache.clean_dict_items:
                                if "," in s:
                                    mask[i] = False

                    elif is_inside_value and in_string:
                        mask = self._cache.p4_mask.copy()
                        if param_count == target_count:
                            for i, s in self._cache.clean_dict_items:
                                if '",' in s.replace(" ", ""):
                                    mask[i] = False

                        if active_key == "regex":
                            for i, s in self._cache.clean_dict_items:
                                if " " in s:
                                    mask[i] = False

                            if re.search(
                                r'"regex"\s*:\s*"$',
                                params_str,
                            ):
                                for i, s in self._cache.clean_dict_items:
                                    if not any(
                                        s.startswith(c)
                                        for c in ["[", "\\"]
                                    ):
                                        mask[i] = False

                    elif param_count == target_count:
                        clean_str = current_str.strip()
                        if clean_str.endswith('"'):
                            current_str = clean_str + "}}"
                            break
                        elif clean_str.endswith("}"):
                            current_str = clean_str + "}"
                            break
                        elif clean_str.endswith(","):
                            current_str = clean_str[:-1] + "}}"
                            break
                        else:
                            mask = self._cache.p4_no_comma.copy()

                    else:
                        mask = self._cache.p4_mask.copy()
                        is_expecting_key = (
                            params_str.strip().endswith("{")
                            or params_str.strip().endswith(",")
                        )
                        if is_expecting_key:
                            for i, s in self._cache.clean_dict_items:
                                cleaned = s.strip()
                                if not (
                                    cleaned.startswith('"') or not cleaned
                                ):
                                    mask[i] = False
            else:
                for i, s in self._cache.mini_dict:
                    if any(rule.startswith(s) for rule in rules):
                        mask[i] = True

            logits[~mask] = -np.inf
            best_id = int(np.argmax(logits))

            selected_token = self._cache.vocab_dict.get(
                best_id,
                "",
            )

            current_str += selected_token
            input_ids.append(best_id)

            self._visualizer.show_step(
                step=step,
                token=selected_token,
                allowed=int(np.count_nonzero(mask)),
                vocab_size=vocab_size,
                current=current_str,
            )

            step += 1

            if (
                current_str.endswith('"')
                and not bridge_injected
                and prefix in current_str
            ):
                bridge = ',"parameters":{'
                current_str += bridge
                input_ids.extend(self._encode(bridge))
                bridge_injected = True

        try:
            extracted_dict = json.loads(current_str)
        except json.JSONDecodeError as error:
            raise GenerationError(
                f"Generated string could not be parsed: {current_str}"
            ) from error

        return extracted_dict

    @staticmethod
    def _get_allowed_chars(
        current_str: str,
        allowed_names: list[str],
    ) -> list[str]:
        prefix = '{"name":"'
        if len(current_str) < len(prefix):
            return [prefix[len(current_str):]]

        after_prefix = current_str[len(prefix):]
        if '"' not in after_prefix:
            return [
                name[len(after_prefix):] + '"'
                for name in allowed_names
                if name.startswith(after_prefix)
            ]

        func_name = after_prefix.split('"')[0]
        target = prefix + func_name + '","parameters":{'
        if len(current_str) < len(target):
            return [target[len(current_str):]]

        return list(string.printable)

    def _encode(self, text: str) -> list[int]:
        encoded = self._model.encode(text)
        try:
            return encoded.tolist()[0]
        except (AttributeError, IndexError, TypeError):
            return list(encoded)
