"""Constrained function selection and argument generation."""

import json

from pydantic import BaseModel

from src.constrained_llm import LLM
from src.function_schema import Function
from src.token_encoder import Encoder
from src.parser import load_functions


class CallMeMaybe(BaseModel):
    """Select a function and generate schema-compatible arguments."""

    llm: LLM
    encoder: Encoder
    functions: dict[str, Function]
    definition_tokens: list[int]
    instruction_prefix: list[int]
    instruction_suffix: list[int]

    def __init__(self, llm: LLM, function_definitions: str) -> None:
        """Load function definitions and prepare the tool instruction."""
        encoder = llm.encoder
        functions: dict[str, Function] = {}

        for definition in load_functions(function_definitions):
            function = Function(
                definition.model_dump(),
                encoder,
            )

            if function.name in functions:
                raise ValueError(
                    f"Duplicate function name: '{function.name}'"
                )

            functions[function.name] = function

        definition_tokens = [
            token
            for function in functions.values()
            for token in function.t_definition
        ]

        instruction_prefix = encoder.encode(
            '<|im_start|>system\n'
            'You are provided with function signatures '
            'within <tools></tools> XML tags:\n'
            '<tools>\n'
        )

        instruction_suffix = encoder.encode(
            '</tools>\n'
            'For each function call, return a json '
            'object within <tool_call></tool_call> tags:\n'
            '<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            '</tool_call>\n'
            '<|im_end|>\n'
        )

        super().__init__(
            llm=llm,
            encoder=encoder,
            functions=functions,
            definition_tokens=definition_tokens,
            instruction_prefix=instruction_prefix,
            instruction_suffix=instruction_suffix,
        )

    def set_tools(self, function: Function | None = None) -> None:
        """Set all tools, or one selected tool, in the LLM instruction."""
        definitions = (
            function.t_definition
            if function is not None
            else self.definition_tokens
        )

        self.llm.set_instruction(
            self.instruction_prefix
            + definitions
            + self.instruction_suffix
        )

    def generate_number(
        self,
        tokens: list[int],
        integer: bool = False,
        max_tokens: int = 128,
    ) -> list[int]:
        """Generate a JSON number while masking invalid token choices."""
        chars = '-0123456789' + ('' if integer else '.')

        stop_ids = {
            self.encoder.encode(char)[0]
            for char in ',}'
        }

        allowed_ids = {
            self.encoder.encode(char)[0]
            for char in chars
        }

        result: list[int] = []
        has_digit = False
        dot_used = False

        for _ in range(max_tokens):
            mask = set(allowed_ids)

            if result:
                mask.discard(self.encoder.encode('-')[0])

            if not has_digit:
                mask -= stop_ids

                if not integer:
                    mask.discard(self.encoder.encode('.')[0])
            else:
                mask |= stop_ids

            if dot_used and not integer:
                mask.discard(self.encoder.encode('.')[0])

            token = self.llm.next_token(tokens + result, mask)
            text = self.encoder.decode(token)

            if token in stop_ids:
                break

            result.append(token)
            has_digit = has_digit or text.isdigit()
            dot_used = dot_used or text == '.'

        if not has_digit:
            raise ValueError('Could not generate a valid number')

        return result

    def argument_context(
        self,
        function: Function,
        prompt: str,
        arg_name: str,
        arg_type: str,
        tokens: list[int],
    ) -> list[int]:
        """Build focused context for one function argument."""
        instruction = (
            'You are completing a function call.\n'
            'For each parameter:\n'
            '1. Read the parameter name and function description.\n'
            '2. Search the user request for an explicit value.\n'
            '3. If an explicit value exists, copy it exactly.\n'
            '4. Never invent information.\n\n'
            f'Function: {function.name}\n'
            f'Description: {function.description}\n'
            f'User request: {prompt}\n'
            f'Parameter: {arg_name}\n'
            f'Type: {arg_type}\n\n'
            'Answer:\n'
        )

        return self.encoder.encode(instruction) + tokens

    def regex_argument_context(
        self,
        function: Function,
        prompt: str,
        arg_name: str,
        arg_type: str,
        tokens: list[int],
    ) -> list[int]:
        """Build focused context for regex substitution arguments."""
        instruction = (
            'You are completing a regex substitution function call.\n'
            'Generate only the requested parameter value.\n'
            'Use the function description and the user request.\n'
        )

        if arg_name == 'regex':
            instruction += (
                'If the user explicitly provides literal text or a word to '
                'match, preserve that literal value exactly.\n'
                'Only generate a general regular-expression pattern when the '
                'user describes a class or category of values.\n'
                'Do not combine separate occurrences from the source string '
                'into a larger pattern.\n'
            )
        elif arg_name == 'replacement':
            instruction += (
                'Use the replacement requested by the user.\n'
                'Preserve explicit replacement values exactly.\n'
            )

        instruction += (
            f'\nFunction: {function.name}\n'
            f'Description: {function.description}\n'
            f'User request: {prompt}\n'
            f'Parameter: {arg_name}\n'
            f'Type: {arg_type}\n\n'
            'Answer:\n'
        )

        return self.encoder.encode(instruction) + tokens

    def generate_string(
        self,
        tokens: list[int],
        max_tokens: int = 128,
    ) -> str:
        """Generate JSON string content until an unescaped quote."""
        context = tokens + self.encoder.encode('"')
        value = ''

        for _ in range(max_tokens):
            token = self.llm.next_token(context)
            text = self.encoder.decode(token)
            candidate = value + text

            quote = candidate.find('"')

            while quote != -1:
                backslashes = 0
                index = quote - 1

                while index >= 0 and candidate[index] == '\\':
                    backslashes += 1
                    index -= 1

                if backslashes % 2 == 0:
                    return candidate[:quote]

                quote = candidate.find('"', quote + 1)

            value = candidate
            context.append(token)

        return value

    def add_args(
        self,
        function: Function,
        tokens: list[int],
        text: str,
    ) -> list[int]:
        """Generate each function argument with schema constraints."""
        for index, arg_name in enumerate(function.param_names):
            arg_type = function.params[arg_name]

            if index:
                tokens += self.encoder.encode(', ')

            tokens += self.encoder.encode(f'"{arg_name}": ')

            if arg_type == 'boolean':
                options = [
                    self.encoder.encode('true'),
                    self.encoder.encode('false'),
                ]
                tokens += self.llm.next_option(tokens, options)

            elif arg_type in ('number', 'integer'):
                selected = self.generate_number(
                    tokens,
                    integer=arg_type == 'integer',
                )

                if arg_type in ('number'):
                    value = self.encoder.decode(selected)

                    if '.' not in value:
                        selected += self.encoder.encode('.0')

                tokens += selected

            elif arg_type == 'string':
                if arg_name in ('regex', 'replacement'):
                    context = self.regex_argument_context(
                        function,
                        text,
                        arg_name,
                        arg_type,
                        tokens,
                    )
                else:
                    context = self.argument_context(
                        function,
                        text,
                        arg_name,
                        arg_type,
                        tokens,
                    )

                self.llm.set_instruction([])
                raw_value = self.generate_string(context)
                self.set_tools(function)

                try:
                    value = json.loads(f'"{raw_value}"')
                except json.JSONDecodeError:
                    value = raw_value

                tokens += self.encoder.encode(json.dumps(value))
            else:
                raise ValueError(
                    f"Unsupported argument type '{arg_type}'"
                )

        tokens += self.encoder.encode('}\n')

        return tokens

    def process_func(self, prompt: str) -> str:
        """Generate one constrained function call for a user prompt."""
        original_prompt = prompt

        text = (
            '<|im_start|>user\n'
            + original_prompt
            + '\n<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<tool_call>\n'
            '{"name": "'
        )

        tokens = self.encoder.encode(text)

        self.set_tools()

        function_names = [
            function.t_name
            for function in self.functions.values()
        ]

        selected_name = self.llm.next_option(
            tokens,
            function_names,
            terminator=self.encoder.encode('"'),
        )

        function = self.functions[
            self.encoder.decode(selected_name)
        ]

        tokens += function.t_name
        tokens += self.encoder.encode(
            '", "arguments": {'
        )

        self.set_tools(function)

        tokens = self.add_args(
            function,
            tokens,
            original_prompt,
        )

        tokens += self.encoder.encode('}')

        raw = self.encoder.decode(tokens)
        tool_json = raw[raw.find('{"name":'):]

        print(repr(tool_json))

        data = json.loads(tool_json)

        result = {
            'prompt': original_prompt,
            'name': data['name'],
            'parameters': data['arguments'],
        }

        return json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
