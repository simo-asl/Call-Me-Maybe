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
    """Generate and validate one schema-conforming call per prompt.

    Args:
        model: Model used by the constrained decoder.
        functions: Validated definitions available for selection.
        prompts: Validated natural-language requests to process.

    Returns:
        JSON-ready function-call results in prompt order.

    Raises:
        GenerationError: If a generated call has an invalid structure, name,
            parameter set, or parameter value type.
    """

    decoder = ConstrainedDecoder(model, functions)
    by_name = {function.name: function for function in functions}
    results: list[dict[str, Any]] = []
    for prompt in prompts:
        generated = decoder.generate(prompt.prompt)
        name = generated["name"]
        parameters = generated["parameters"]
        if not isinstance(name, str) or not isinstance(parameters, dict):
            raise GenerationError(
                "Decoder returned an invalid function-call structure.")
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
    """Verify a generated call exactly matches its selected schema.

    Args:
        name: Name of the selected function.
        parameters: Generated arguments for that function.
        functions: Validated definitions indexed by function name.

    Raises:
        GenerationError: If the name, keys, or value types are invalid.
    """

    function = functions.get(name)
    if function is None:
        raise GenerationError(f"Generated unknown function name '{name}'.")
    if set(parameters) != set(function.parameters):
        raise GenerationError(
            "Generated parameters do not exactly match the selected schema.")
    for key, definition in function.parameters.items():
        if not _value_matches_type(parameters[key], definition.type):
            message = f"Generated parameter '{
                key}' does not match type '{definition.type}'."
            raise GenerationError(message)


def _value_matches_type(value: object, expected: str) -> bool:
    """Return whether a decoded JSON value matches the expected schema type.

    Args:
        value: Decoded JSON value to check.
        expected: Supported schema type name.

    Returns:
        Whether the value conforms to the expected type.
    """

    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return value is None
