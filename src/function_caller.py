"""Constrained function selection and argument generation."""

import json
import re

from pydantic import BaseModel

from src.constrained_llm import LLM
from src.function_schema import Function
from src.token_encoder import Encoder


REGEX_MAPPING = [
    (['vowel', 'vowels'], r'[aeiouAEIOU]'),
    (['consonant', 'consonants'],
     r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]'),
    (['digit', 'digits', 'number', 'numbers'], r'\\d+'),
    (['uppercase', 'upper', 'capital'], r'[A-Z]+'),
    (['lowercase', 'lower'], r'[a-z]+'),
    (['letter', 'letters', 'alphabetic'], r'[a-zA-Z]+'),
    (['space', 'spaces', 'whitespace'], r'\\s+'),
    (['punctuation', 'special'], r'[^\w\s]'),
    (['alphanumeric'], r'\\w+'),
    (['newline', 'newlines'], r'\\n+'),
    (['tab', 'tabs'], r'\\t+'),
]

NUMBER_PATTERN = re.compile(
    r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
)


def escape(text: str) -> str:
    """Escape backslashes and quotes before embedding text in JSON."""
    return text.replace('\\', '\\\\').replace('"', '\\"')


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
        with open(function_definitions, 'r', encoding='utf-8') as file:
            for definition in json.load(file):
                function = Function(definition, encoder)
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
            self.instruction_prefix + definitions + self.instruction_suffix
        )

    def regex_pattern(self, text: str) -> list[int]:
        """Infer a supported regex pattern from words in the prompt."""
        words = {word.strip('\'\".,!?').lower() for word in text.split()}
        for keywords, pattern in REGEX_MAPPING:
            if words & set(keywords):
                return self.encoder.encode(pattern)

        match = re.search(r"['\"](\w+)['\"]", text)
        if match:
            return self.encoder.encode(match.group(1))
        return self.encoder.encode(r'\\w+')

    def path_options(self, text: str) -> list[list[int]]:
        """Return Unix or Windows filesystem paths found in the prompt."""
        paths = re.findall(r'(?:[A-Za-z]:\\\\[^\s]+|/[^\s]+)', text)
        return [
            self.encoder.encode(path.strip('"\''))
            for path in paths
        ]

    def number_options(self, text: str) -> list[list[int]]:
        """Return numeric values found in the prompt as token sequences."""
        return [self.encoder.encode(
            value) for value in NUMBER_PATTERN.findall(text)]

    def compatible_functions(self, text: str) -> list[Function]:
        """Return functions whose argument
        types can be formed from the prompt."""
        has_number = bool(NUMBER_PATTERN.search(text))
        compatible = [
            function
            for function in self.functions.values()
            if has_number
            or not any(
                param_type in ('number', 'float')
                for param_type in function.params.values()
            )
        ]
        return compatible or list(self.functions.values())

    def add_args(
        self,
        function: Function,
        tokens: list[int],
        text: str,
    ) -> list[int]:
        """Generate each function argument from constrained candidates."""
        for index, arg_name in enumerate(function.param_names):
            arg_type = function.params[arg_name]
            if index:
                tokens += self.encoder.encode(', ')
            tokens += self.encoder.encode(f'"{arg_name}": ')

            if arg_name == 'regex':
                tokens += self.encoder.encode('"')
                tokens += self.regex_pattern(text)
                tokens += self.encoder.encode('"')
                continue

            if arg_type == 'boolean':
                options = [
                    self.encoder.encode('true'),
                    self.encoder.encode('false'),
                ]
            elif arg_type in ('number', 'float'):
                options = self.number_options(text)
            elif arg_name == 'path':
                options = self.path_options(text)
                if not options:
                    options = self.encoder.encode_words_separated(text)
            else:
                options = self.encoder.encode_words_separated(text)

            if not options:
                raise ValueError(
                    f"No valid {arg_type} candidate for argument '{arg_name}'"
                )

            if arg_type == 'string':
                tokens += self.encoder.encode('"')

            selected = self.llm.next_option(tokens, options)
            if arg_type in ('number', 'float'):
                value = self.encoder.decode(selected)
                if value.isdigit():
                    selected += self.encoder.encode('.0')
            tokens += selected

            if arg_type == 'string':
                tokens += self.encoder.encode('"')

        tokens += self.encoder.encode('}\n')
        return tokens

    def process_func(self, prompt: str) -> str:
        """Generate one constrained function call for a user prompt."""
        prompt = escape(prompt)
        text = (
            '<|im_start|>user\n'
            + prompt
            + '\n<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<tool_call>\n'
            '{"name": "'
        )
        tokens = self.encoder.encode(text)

        self.set_tools()
        candidates = self.compatible_functions(prompt)
        function_names = [function.t_name for function in candidates]
        selected_name = self.llm.next_option(tokens, function_names)
        function = self.functions[self.encoder.decode(selected_name)]

        tokens += function.t_name
        tokens += self.encoder.encode('", "arguments": {')
        self.set_tools(function)
        tokens = self.add_args(function, tokens, prompt)
        tokens += self.encoder.encode('}')

        raw = self.encoder.decode(tokens)
        tool_json = raw[raw.find('{"name":'):]
        print(repr(tool_json))
        data = json.loads(tool_json)

        return (
            '\t{\n'
            f'\t\t"prompt": "{prompt}",\n'
            f'\t\t"name": "{data["name"]}",\n'
            f'\t\t"parameters": {json.dumps(data["arguments"])}\n'
            '\t}'
        )
