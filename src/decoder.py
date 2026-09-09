"""Token-by-token constrained decoding for function-call JSON."""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

from src.errors import GenerationError, InputError
from src.generation_utils import (
    inject_bridge_after_token,
    select_token,
    try_teleport,
    ParameterState,
    analyze_parameters,
    build_prompt,
    build_tiny_prompt,
    get_allowed_chars,
)
from src.models import FunctionDefinition
from src.printing import GenerationVisualizer


class LlmModel(Protocol):
    """Public subset of ``llm_sdk.Small_LLM_Model`` used by this project."""

    def encode(self, text: str) -> object:
        """Encode text as a tensor-like value containing input token IDs.

        Args:
            text: Text to tokenize.

        Returns:
            A tensor-like object or iterable containing token IDs.
        """

    def get_logits_from_input_ids(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Return next-token logits for the supplied token IDs.

        Args:
            input_ids: Existing token-ID sequence.

        Returns:
            One logit for each tokenizer vocabulary entry.
        """

    def get_path_to_vocab_file(self) -> str:
        """Return the path to the tokenizer vocabulary JSON file."""


@dataclass
class MaskCache:
    """Store vocabulary data and reusable masks for constrained generation.

    Attributes:
        raw_functions: Serialized definitions used in generation prompts.
        allowed_fn: Function names permitted in generated calls.
        func_params: Expected parameter count indexed by function name.
        param_types: Parameter types indexed by function and parameter name.
        vocab_dict: Token text indexed by vocabulary ID.
        clean_dict_items: Printable vocabulary ID and text pairs.
        mini_dict: Vocabulary entries relevant to JSON structure generation.
        p4_mask: Mask for all printable vocabulary entries.
        p4_numbers_only: Mask for numeric-value generation.
        p4_no_comma: Printable-entry mask with comma-containing tokens removed.
    """

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
    """Generate valid JSON function calls with token-level constraints.

    Attributes:
        _model: Language-model interface used to score next tokens.
        _functions: Validated definitions available to the model.
        _cache: Derived vocabulary data and token masks.
        _visualizer: Terminal renderer for each generated token.
    """

    def __init__(
        self,
        model: LlmModel,
        functions: list[FunctionDefinition],
    ) -> None:
        """Initialize the decoder and build its reusable vocabulary cache.

        Args:
            model: Model exposing the required public SDK operations.
            functions: Validated functions that may be selected.
        """
        self._model = model
        self._functions = functions
        self._cache = self._build_cache()
        self._visualizer = GenerationVisualizer()

    def _build_cache(self) -> MaskCache:
        """Build vocabulary lookups and masks used while decoding.

        Returns:
            Cached function metadata, vocabulary fragments, and masks.

        Raises:
            InputError: If the model vocabulary cannot be read as JSON.
        """
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
            value: key.replace("Ġ", " ")
            for key, value in raw_vocab.items()
            if isinstance(value, int)
        }

        printable_set = set(string.printable)

        valid_ids = [
            token_id
            for token_id, token_str in vocab_dict.items()
            if token_str
            and all(char in printable_set for char in token_str)
        ]

        clean_dict_items = [
            (token_id, token_str)
            for token_id, token_str in vocab_dict.items()
            if token_str
            and all(char in printable_set for char in token_str)
        ]

        raw_functions = cast(list[dict[str, object]], [
            {
                "name": function.name,
                "description": function.description,
                "parameters": {
                    key: value.model_dump()
                    for key, value in function.parameters.items()
                },
            }
            for function in self._functions
        ])

        allowed_fn = [function.name for function in self._functions]
        func_params: dict[str, int] = {}
        param_types: dict[str, dict[str, object]] = {}

        for function in raw_functions:
            name = str(function["name"])
            params = function.get("parameters", {})

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

        for token_id, token_str in clean_dict_items:
            if (
                all(
                    char in allowed_math_chars
                    for char in token_str
                )
                or token_str == "null"
            ):
                p4_numbers_only[token_id] = True

        p4_no_comma = p4_mask.copy()

        for token_id, token_str in clean_dict_items:
            if "," in token_str:
                p4_no_comma[token_id] = False

        target_phrases = (
            allowed_fn
            + ['{"name":"', '","parameters":{', "}"]
        )

        mini_dict = [
            (token_id, token_str)
            for token_id, token_str in clean_dict_items
            if any(
                token_str in phrase
                for phrase in target_phrases
            )
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

    def _handle_bridge_fast_forward(
        self,
        current_str: str,
        input_ids: list[int],
        user_prompt: str,
        prefix: str,
        bridge_injected: bool,
    ) -> tuple[str, list[int], bool, bool, bool]:
        """Insert the parameter bridge after a completed function name.

        Args:
            current_str: Function-call JSON generated so far.
            input_ids: Token IDs representing the active model context.
            user_prompt: Original prompt being converted to a function call.
            prefix: Fixed JSON prefix preceding the function name.
            bridge_injected: Whether the parameter bridge was already inserted.

        Returns:
            Updated text and IDs, followed by bridge, completion, and
            step-skipping flags.
        """
        if (
            not current_str.endswith('"')
            or bridge_injected
            or prefix not in current_str
            or len(current_str) <= len(prefix)
        ):
            return current_str, input_ids, bridge_injected, False, False

        bridge = ',"parameters":{'
        current_str += bridge
        input_ids.extend(self._encode(bridge))
        bridge_injected = True

        func_name = current_str.split('"name":"')[1].split('"')[0]

        if self._cache.func_params.get(func_name, 99) == 0:
            current_str += "}}"
            return current_str, input_ids, bridge_injected, True, False

        active_schema = next(
            (
                function
                for function in self._cache.raw_functions
                if function["name"] == func_name
            ),
            None,
        )

        if active_schema:
            tiny_prompt = build_tiny_prompt(
                active_schema,
                user_prompt,
                current_str,
            )
            input_ids = self._encode(tiny_prompt)
        else:
            input_ids.extend(self._encode(bridge))

        return current_str, input_ids, bridge_injected, False, True

    def _build_number_mask(self, state: ParameterState) -> np.ndarray:
        """Create a mask appropriate for the current numeric parameter.

        Args:
            state: Parsed state of the parameters object.

        Returns:
            Boolean vocabulary mask for numeric fragments.
        """
        mask = self._cache.p4_numbers_only.copy()
        current_value = state.params_str[
            state.last_structural_colon + 1:
        ].strip()
        has_decimal_point = "." in current_value

        if not has_decimal_point:
            for token_id, token_str in self._cache.clean_dict_items:
                if "," in token_str or "}" in token_str:
                    mask[token_id] = False
        elif state.param_count == state.target_count:
            for token_id, token_str in self._cache.clean_dict_items:
                if "," in token_str:
                    mask[token_id] = False

        return mask

    def _build_string_mask(self, state: ParameterState) -> np.ndarray:
        """Create a mask appropriate for the current string parameter.

        Args:
            state: Parsed state of the parameters object.

        Returns:
            Boolean vocabulary mask for string fragments.
        """
        mask = self._cache.p4_mask.copy()
        if not state.in_string:
            for token_id, token_str in self._cache.clean_dict_items:
                if not token_str.strip().startswith('"'):
                    mask[token_id] = False

        if state.param_count == state.target_count:
            for token_id, token_str in self._cache.clean_dict_items:
                cleaned = token_str.replace(" ", "")
                if '",' in cleaned or '"},' in cleaned:
                    mask[token_id] = False

        if state.active_key == "regex":
            for token_id, token_str in self._cache.clean_dict_items:
                if " " in token_str:
                    mask[token_id] = False

            if re.search(
                r'"regex"\s*:\s*"$',
                state.params_str,
            ):
                for token_id, token_str in self._cache.clean_dict_items:
                    if not any(
                        token_str.startswith(char)
                        for char in ["[", "\\"]
                    ):
                        mask[token_id] = False

        return mask

    def _finish_parameter_mask(
        self,
        current_str: str,
    ) -> tuple[np.ndarray, str, bool]:
        """Complete a parameter object once all expected keys are present.

        Args:
            current_str: Function-call JSON generated so far.

        Returns:
            A comma-free mask, possibly completed JSON text, and whether
            generation should stop.
        """
        clean_str = current_str.strip()

        if clean_str.endswith('"'):
            return self._cache.p4_no_comma.copy(), clean_str + "}}", True

        if clean_str.endswith("}"):
            return self._cache.p4_no_comma.copy(), clean_str + "}", True

        if clean_str.endswith(","):
            return self._cache.p4_no_comma.copy(), clean_str[:-1] + "}}", True

        return self._cache.p4_no_comma.copy(), current_str, False

    def _build_parameter_key_mask(
        self,
        state: ParameterState,
        func_name: str,
    ) -> np.ndarray:
        """Create a mask limited to valid unused parameter keys.

        Args:
            state: Parsed state of the parameters object.
            func_name: Name of the selected function.

        Returns:
            Boolean vocabulary mask for valid parameter-key continuations.
        """
        mask = np.zeros_like(
                self._cache.p4_mask,
                dtype=bool,
            )
        allowed_keys = [
            key
            for key in self._cache.param_types.get(func_name, {})
            if key not in state.used_keys
        ]

        params_str = state.params_str
        key_start = max(
            params_str.rfind("{"),
            params_str.rfind(","),
        )

        current_fragment = params_str[key_start + 1:].strip()

        targets = [
            f'"{key}":'
            for key in allowed_keys
        ]

        for token_id, token_str in self._cache.clean_dict_items:
            candidate = current_fragment + token_str

            if any(
                target.startswith(candidate)
                for target in targets
            ):
                mask[token_id] = True

        return mask

    def _build_structural_mask(
        self,
        rules: list[str],
        vocab_size: int,
    ) -> np.ndarray:
        """Create a mask for fixed JSON and function-name prefixes.

        Args:
            rules: Remaining valid structural strings.
            vocab_size: Number of entries in the model vocabulary.

        Returns:
            Boolean vocabulary mask for prefixes matching a rule.
        """
        mask = np.zeros(vocab_size, dtype=bool)

        for token_id, token_str in self._cache.mini_dict:
            if any(
                rule.startswith(token_str)
                for rule in rules
            ):
                mask[token_id] = True

        return mask

    def _build_generation_mask(
        self,
        current_str: str,
        rules: list[str],
        vocab_size: int,
    ) -> tuple[np.ndarray, str, bool]:
        """Build the token mask for the current generation state.

        Args:
            current_str: JSON string generated so far.
            rules: Structural generation rules currently being followed.
            vocab_size: Number of tokens in the model vocabulary.

        Returns:
            A tuple containing the token mask, current generated string,
            and whether generation should stop.
        """
        if len(rules) <= 10:
            return (
                self._build_structural_mask(rules, vocab_size),
                current_str,
                False,
            )

        mask = np.zeros(vocab_size, dtype=bool)
        match = re.search(
            r'"name"\s*:\s*"([^"]+)',
            current_str,
        )
        func_name = match.group(1) if match else ""

        state = analyze_parameters(
            current_str,
            func_name,
            self._cache.param_types,
            self._cache.func_params,
        )

        if not state:
            return mask, current_str, False

        if state.is_inside_value and state.expected_type == "number":
            return self._build_number_mask(state), current_str, False

        current_value = state.params_str[
            state.last_structural_colon + 1:
        ].strip()

        if (
            state.is_inside_value
            and state.expected_type == "string"
            and (
                state.in_string
                or not current_value
            )
        ):
            return self._build_string_mask(state), current_str, False

        if state.param_count == state.target_count:
            return self._finish_parameter_mask(current_str)

        return (
            self._build_parameter_key_mask(state, func_name),
            current_str,
            False,
        )

    def generate(self, user_prompt: str) -> dict[str, object]:
        """Generate one parseable JSON function-call object for a prompt.

        Args:
            user_prompt: Natural-language request to convert into a call.

        Returns:
            Parsed JSON object generated under the decoder's constraints.

        Raises:
            GenerationError: If the prompt is empty or generated text is not
                parseable JSON.
        """
        if not user_prompt.strip():
            raise GenerationError("User prompt cannot be empty.")

        prompt = build_prompt(
            self._cache.raw_functions,
            user_prompt,
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
            current_str, teleported = try_teleport(
                self,
                current_str,
                input_ids,
                prefix,
            )

            if teleported:
                continue

            (
                current_str,
                input_ids,
                bridge_injected,
                finished,
                skip_step,
            ) = self._handle_bridge_fast_forward(
                current_str,
                input_ids,
                user_prompt,
                prefix,
                bridge_injected,
            )

            if finished:
                break

            if skip_step:
                continue

            rules = get_allowed_chars(
                current_str,
                self._cache.allowed_fn,
            )

            mask, current_str, finished = self._build_generation_mask(
                current_str,
                rules,
                vocab_size,
            )

            if finished:
                break

            best_id, selected_token = select_token(
                self,
                input_ids,
                mask,
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

            current_str, bridge_injected = inject_bridge_after_token(
                self,
                current_str,
                input_ids,
                prefix,
                bridge_injected,
            )

        try:
            return cast(dict[str, object], json.loads(current_str))
        except json.JSONDecodeError as error:
            raise GenerationError(
                f"Generated string could not be parsed: {current_str}"
            ) from error

    def _encode(self, text: str) -> list[int]:
        """Normalize the model's encoded representation to a token-ID list.

        Args:
            text: Text to encode with the configured model.

        Returns:
            Token IDs extracted from the model's return value.
        """
        encoded = cast(Any, self._model.encode(text))

        try:
            return cast(list[int], encoded.tolist()[0])
        except (AttributeError, IndexError, TypeError):
            return cast(list[int], list(encoded))
