"""Pydantic models and validation helpers for project data."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.errors import InputError


SupportedType = Literal["string", "number", "integer", "boolean", "null"]


class ParameterDefinition(BaseModel):
    """Describe one parameter accepted by a callable function."""

    model_config = ConfigDict(extra="forbid")

    type: SupportedType


class FunctionDefinition(BaseModel):
    """Describe a function that the model is allowed to call."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, Any]


class PromptInput(BaseModel):
    """Represent one natural-language function-calling request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class FunctionCall(BaseModel):
    """The exact structure written for each successful request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


def validate_functions(raw_data: Any) -> list[FunctionDefinition]:
    """Validate function definitions and reject duplicate names and keys."""

    if not isinstance(raw_data, list) or not raw_data:
        raise InputError("Function definitions must be a non-empty JSON list.")
    try:
        functions = [FunctionDefinition.model_validate(
            item) for item in raw_data]
    except ValidationError as error:
        raise InputError(
            format_validation_error("function definitions", error))
    names = [function.name for function in functions]
    if len(names) != len(set(names)):
        raise InputError(
            "Function definitions contain duplicate function names.")
    for function in functions:
        if any(not key for key in function.parameters):
            raise InputError(
                f"Function '{function.name}' has an empty parameter name.")
    return functions


def validate_prompts(raw_data: Any) -> list[PromptInput]:
    """Validate the input prompt list and reject duplicate prompts."""

    if not isinstance(raw_data, list):
        raise InputError("Prompt input must be a JSON list.")

    try:
        prompts = [PromptInput.model_validate(item) for item in raw_data]
    except ValidationError as error:
        raise InputError(format_validation_error("prompt input", error))

    prompt_values = [prompt.prompt for prompt in prompts]
    if len(prompt_values) != len(set(prompt_values)):
        raise InputError("Prompt input contains duplicate prompts.")

    return prompts


def format_validation_error(label: str, error: ValidationError) -> str:
    """Turn Pydantic errors into compact user-facing messages."""

    messages = []

    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        messages.append(f"{location}: {item['msg']}")

    return f"Invalid {label}: " + "; ".join(messages)
