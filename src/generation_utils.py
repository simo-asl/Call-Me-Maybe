"""Generation helper functions."""

import json
import re
import string
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ParameterState:
    params_str: str
    in_string: bool
    is_inside_value: bool
    active_key: str
    expected_type: object
    param_count: int
    target_count: int
    last_structural_colon: int


def build_prompt(
    raw_functions: list[dict[str, object]],
    user_prompt: str,
) -> str:
    schema_hints = json.dumps(
        raw_functions,
        separators=(",", ":"),
    )

    return (
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


def build_tiny_prompt(
    schema: dict[str, object],
    user_prompt: str,
    current_str: str,
) -> str:
    tiny_schema = json.dumps(
        [schema],
        separators=(",", ":"),
    )

    return (
        f"System: Output valid JSON matching this schema: "
        f"{tiny_schema}\n"
        r"Rule: For the regex field, NEVER output literal matches. "
        r"Always use proper regex sets "
        r'(e.g. "[aeiouAEIOU]", "[0-9]+", "\\bword\\b"). '
        r"For replacement, if asked for a character "
        r"(e.g. asterisks), output EXACTLY ONE character "
        r"(e.g. '*')."
        "\n"
        f"User: {user_prompt}\n"
        f"Tool Call: {current_str}"
    )


def get_allowed_chars(
    current_str: str,
    allowed_names: list[str],
) -> list[str]:
    prefix = '{"name":"'
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


def analyze_parameters(
    current_str: str,
    func_name: str,
    param_types: dict[str, dict[str, object]],
    func_params: dict[str, int],
) -> ParameterState | None:
    if '"parameters"' not in current_str:
        return None

    params_str = current_str.split('"parameters"')[1]

    if not params_str:
        return None

    in_string = False
    last_colon = -1
    last_comma = -1
    last_brace = -1

    for i, char in enumerate(params_str):
        if char == '"':
            backslashes = 0
            j = i - 1

            while j >= 0 and params_str[j] == "\\":
                backslashes += 1
                j -= 1

            if backslashes % 2 == 0:
                in_string = not in_string

        elif not in_string:
            if char == ":":
                last_colon = i
            elif char == ",":
                last_comma = i
            elif char == "}":
                last_brace = i

    is_inside_value = (
        last_colon > last_comma
        and last_colon > last_brace
    )

    keys = re.findall(
        r'"([^"]+)"\s*:',
        params_str,
    )

    active_key = (
        keys[-1]
        if is_inside_value and keys
        else ""
    )

    expected_type = param_types.get(
        func_name,
        {},
    ).get(active_key, "Any")

    return ParameterState(
        params_str=params_str,
        in_string=in_string,
        is_inside_value=is_inside_value,
        active_key=active_key,
        expected_type=expected_type,
        param_count=len(keys),
        target_count=func_params.get(func_name, 99),
        last_structural_colon=last_colon,
    )


def try_teleport(
    self: Any,
    current_str: str,
    input_ids: list[int],
    prefix: str,
) -> tuple[str, bool]:
    if (
        prefix not in current_str
        or '","parameters":{' in current_str
    ):
        return current_str, False

    after_prefix = current_str.split(prefix)[1]

    possible_names = [
        name
        for name in self._cache.allowed_fn
        if name.startswith(after_prefix)
    ]

    if (
        len(possible_names) != 1
        or possible_names[0] == after_prefix
    ):
        return current_str, False

    remainder = possible_names[0][len(after_prefix):] + '"'

    current_str += remainder
    input_ids.extend(self._encode(remainder))

    return current_str, True


def inject_bridge_after_token(
    self: Any,
    current_str: str,
    input_ids: list[int],
    prefix: str,
    bridge_injected: bool,
) -> tuple[str, bool]:
    if (
        not current_str.endswith('"')
        or bridge_injected
        or prefix not in current_str
    ):
        return current_str, bridge_injected

    bridge = ',"parameters":{'

    current_str += bridge
    input_ids.extend(self._encode(bridge))

    return current_str, True


def select_token(
    self: Any,
    input_ids: list[int],
    mask: np.ndarray,
) -> tuple[int, str]:
    logits = np.array(
        self._model.get_logits_from_input_ids(input_ids)
    )

    logits[~mask] = -np.inf
    best_id = int(np.argmax(logits))

    selected_token = self._cache.vocab_dict.get(
        best_id,
        "",
    )

    return best_id, selected_token
