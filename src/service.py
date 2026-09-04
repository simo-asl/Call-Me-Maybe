"""High-level orchestration of generation and output validation."""

from __future__ import annotations

from typing import Any

from src.decoder import ConstrainedDecoder, LlmModel
from src.errors import GenerationError
from src.models import FunctionCall, FunctionDefinition, PromptInput


def generate_calls(
    model: LlmModel,
    functions: list[FunctionDefinition],
    prompts: list[PromptInput],
) -> list[dict[str, Any]]:
    """Generate and validate one schema-conforming call for every input prompt."""

    decoder = ConstrainedDecoder(model, functions)
    by_name = {function.name: function for function in functions}
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        generated = decoder.generate(prompt.prompt)
        name = generated["name"]
        parameters = generated["parameters"]
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise GenerationError("Decoder returned an invalid function-call structure.")
        _validate_call_against_schema(name, parameters, by_name)
        result = FunctionCall(
            prompt=prompt.prompt,
            name=name,
            parameters=parameters,
        )
        results.append(result.model_dump())
    return results


def _validate_call_against_schema(
    name: str,
    parameters: dict[str, object],
    functions: dict[str, FunctionDefinition],
) -> None:
    """Defensively verify keys and Python values before writing the result."""

    function = functions.get(name)
    if function is None:
        raise GenerationError(f"Generated unknown function name '{name}'.")
    if set(parameters) != set(function.parameters):
        raise GenerationError("Generated parameters do not exactly match the selected schema.")
    for key, definition in function.parameters.items():
        if not _value_matches_type(parameters[key], definition.type):
            message = f"Generated parameter '{key}' does not match type '{definition.type}'."
            raise GenerationError(message)


def _value_matches_type(value: object, expected: str) -> bool:
    """Return whether a decoded JSON value has the expected schema type."""

    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return value is None
