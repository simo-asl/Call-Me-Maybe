"""Pydantic models and validation helpers for project data."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.errors import InputError


SupportedType = Literal["string", "number", "integer", "boolean", "null"]


class ParameterDefinition(BaseModel):
    """Describe one parameter accepted by a callable function."""

    model_config = ConfigDict(extra="ignore")

    type: SupportedType


class FunctionDefinition(BaseModel):
    """Describe a function that the model is allowed to call."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, Any]


class PromptInput(BaseModel):
    """Represent one natural-language function-calling request."""

    model_config = ConfigDict(extra="ignore")

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
        functions = [FunctionDefinition.model_validate(item) for item in raw_data]
    except ValidationError as error:
        raise InputError(format_validation_error("function definitions", error)) from error
    names = [function.name for function in functions]
    if len(names) != len(set(names)):
        raise InputError("Function definitions contain duplicate function names.")
    for function in functions:
        if any(not key for key in function.parameters):
            raise InputError(f"Function '{function.name}' has an empty parameter name.")
    return functions


def validate_prompts(raw_data: Any) -> list[PromptInput]:
    """Validate the input prompt list."""

    if not isinstance(raw_data, list):
        raise InputError("Prompt input must be a JSON list.")
    try:
        return [PromptInput.model_validate(item) for item in raw_data]
    except ValidationError as error:
        raise InputError(format_validation_error("prompt input", error)) from error


def format_validation_error(label: str, error: ValidationError) -> str:
    """Turn the first Pydantic error into a compact user-facing message."""

    first_error = error.errors()[0]
    location = ".".join(str(part) for part in first_error["loc"])
    return f"Invalid {label} at {location}: {first_error['msg']}"
