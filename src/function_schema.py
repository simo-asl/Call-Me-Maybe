"""Function schema representation used during constrained generation."""

import json
from typing import Any

from pydantic import BaseModel, PrivateAttr

from src.token_encoder import Encoder


class Function(BaseModel):
    """Store one function definition and its encoded token data."""

    _name: str = PrivateAttr()
    _t_name: list[int] = PrivateAttr()
    _description: str = PrivateAttr()
    _params: dict[str, str] = PrivateAttr()
    _t_definition: list[int] = PrivateAttr()

    def __init__(self, function: dict[str, Any], encoder: Encoder):
        """Build a function object from one JSON definition."""
        super().__init__()

        self._name = function['name']
        self._t_name = encoder.encode(self._name)
        self._description = function.get('description', '')

        self._params = {
            name: schema['type']
            for name, schema in function['parameters'].items()
        }

        self._t_definition = encoder.encode(
            self._to_tool_schema()
        )

    def _to_tool_schema(self) -> str:
        """Return the function definition as a JSON tool schema string."""
        properties = {
            name: {'type': param_type}
            for name, param_type in self._params.items()
        }

        return json.dumps({
            'name': self._name,
            'description': self._description,
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': list(self._params),
            },
        })

    @property
    def name(self) -> str:
        """Return the function name."""
        return self._name

    @property
    def t_name(self) -> list[int]:
        """Return encoded function-name tokens."""
        return self._t_name

    @property
    def description(self) -> str:
        """Return the function description."""
        return self._description

    @property
    def params(self) -> dict[str, str]:
        """Return parameter names mapped to their declared types."""
        return self._params

    @property
    def param_names(self) -> list[str]:
        """Return parameter names in declaration order."""
        return list(self._params)

    @property
    def t_definition(self) -> list[int]:
        """Return encoded tool-schema tokens."""
        return self._t_definition
