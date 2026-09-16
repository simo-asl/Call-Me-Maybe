"""Parse and validate JSON input files used by Call Me Maybe.

This module defines the Pydantic schemas for user prompts and function
definitions. It also provides helpers for loading JSON files while
rejecting duplicate keys before validating their structure.
"""

import json
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
)


Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[a-zA-Z_]\w*$"),
]

NonEmptyString = Annotated[
    str,
    StringConstraints(pattern=r"\S+"),
]


class PromptModel(BaseModel):
    """Represent and validate one user prompt.

    The prompt must be a string containing at least one non-whitespace
    character. Extra fields are not accepted.
    """

    prompt: NonEmptyString

    model_config = ConfigDict(extra="forbid")


class PromptsModel(RootModel[list[PromptModel]]):
    """Represent and validate the complete list of user prompts.

    The input must contain at least one valid prompt.
    """

    root: Annotated[
        list[PromptModel],
        Field(min_length=1),
    ]


class TypeModel(BaseModel):
    """Represent a supported function parameter or return type.

    Only the types required by the project are accepted.
    Extra fields are not accepted.
    """

    type: Literal[
        "number",
        "string",
        "boolean",
        "integer",
    ]

    model_config = ConfigDict(extra="forbid")


class FunctionModel(BaseModel):
    """Represent and validate one function definition.

    A function must contain a valid name, a non-empty description,
    its parameter definitions, and its return type. Extra fields are
    not accepted.
    """

    name: Identifier
    description: NonEmptyString
    parameters: dict[Identifier, TypeModel]
    returns: TypeModel

    model_config = ConfigDict(extra="forbid")


class FunctionsModel(RootModel[list[FunctionModel]]):
    """Represent and validate the complete list of function definitions.

    The definitions file must contain at least one valid function.
    """

    root: Annotated[
        list[FunctionModel],
        Field(min_length=1),
    ]


def reject_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys.

    Args:
        pairs: Ordered key-value pairs produced by the JSON decoder.

    Returns:
        A dictionary containing the parsed key-value pairs.

    Raises:
        ValueError: If the same key appears more than once.
    """
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key: '{key}'")
        result[key] = value

    return result


def load_prompts(path: str) -> list[str]:
    """Load and validate prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        A list containing the validated prompt strings.

    Raises:
        ValueError: If duplicate JSON keys are found.
        pydantic.ValidationError: If the input does not match the
            expected prompt schema.
        json.JSONDecodeError: If the file does not contain valid JSON.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(
            file,
            object_pairs_hook=reject_duplicates,
        )

    validated = PromptsModel.model_validate(data)

    return [
        item.prompt
        for item in validated.root
    ]


def load_functions(path: str) -> list[FunctionModel]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the function definitions JSON file.

    Returns:
        A list containing the validated function definitions.

    Raises:
        ValueError: If duplicate JSON keys are found.
        pydantic.ValidationError: If a definition does not match the
            expected function schema.
        json.JSONDecodeError: If the file does not contain valid JSON.
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(
            file,
            object_pairs_hook=reject_duplicates,
        )

    validated = FunctionsModel.model_validate(data)

    return validated.root
