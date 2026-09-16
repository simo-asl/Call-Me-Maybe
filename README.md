*This project has been created as part of the 42 curriculum by mel-asla.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling project built around a small local language model. Its goal is to translate natural-language requests into structured function calls without executing the functions themselves.

The program receives two JSON inputs: a list of available function definitions and a list of prompts. For every prompt, it must choose the correct function and generate arguments that match the function schema. The result is written to `data/output/function_calling_results.json` using exactly three fields: `prompt`, `name`, and `parameters`.

The main difficulty of the project is that a small language model cannot be trusted to freely generate correct structured output every time. The implementation therefore uses **constrained decoding**: raw next-token logits are obtained from the provided `llm_sdk`, invalid choices are masked, and the model is only allowed to choose among tokens that are valid for the current decision.

The required and default model is:

```text
Qwen/Qwen3-0.6B
```

Function definitions are loaded dynamically from the input file. Function selection is made from the LLM logits and is not based on hardcoded function names or keyword matching.

## Project structure

```text
src/
├── __init__.py
├── __main__.py
├── constrained_llm.py
├── function_caller.py
├── function_schema.py
├── parser.py
└── token_encoder.py
```

### `__main__.py`

This is the command-line entry point. It parses the input paths and model option, initializes the SDK model and encoder, reads the prompts, creates the output directory when necessary, processes every request, and writes the final JSON array.

It also handles missing files, malformed JSON, and unexpected runtime errors with readable messages instead of an uncontrolled traceback.

### `token_encoder.py`

This module implements the tokenizer used by the project from the vocabulary file exposed by the SDK.

It builds:

- a trie used for longest-match encoding;
- a reverse vocabulary used for decoding token IDs;
- conversions between normal whitespace and the vocabulary markers `Ġ`, `Ċ`, and `ĉ`.

Keeping encoding in the project makes the relation between text, token IDs, and constrained generation easier to control.

### `function_schema.py`

`Function` represents one function definition. It stores the function name, description, parameter names and types, encoded name, and encoded tool schema.

The currently supported parameter types are:

```text
number
integer
string
boolean
```

Unsupported parameter types are rejected instead of being generated incorrectly.

### `constrained_llm.py`

`LLM` is a small wrapper around `Small_LLM_Model`. It is responsible for retrieving logits, applying token masks, selecting the highest-scoring allowed token, and selecting one complete sequence from a list of valid token sequences.

Only public SDK functionality is used to interact with the model.

### `parser.py`

parser.py validates prompt and function-definition JSON with Pydantic
and rejects duplicate JSON keys before they can be overwritten.

### `function_caller.py`

`CallMeMaybe` contains the main function-calling pipeline. It loads the available tools, prepares the tool instruction, selects a function, generates each argument according to its declared type, and builds the final JSON object.

## Instructions

The project requires Python 3.10 or later and uses `uv` for dependency management.

Install the environment:

```bash
make install
```

or directly:

```bash
uv sync
```

Run the project with the default files:

```bash
make run
```

which executes:

```bash
uv run python -m src
```

The default paths are:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
data/output/function_calling_results.json
```

Custom paths can be provided with the subject command-line interface:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The project also accepts the implemented alternative model option:

```bash
uv run python -m src --model Qwen/Qwen2.5-0.5B
```

The required `Qwen/Qwen3-0.6B` remains the default model.

Useful Makefile targets are:

```bash
make install
make run
make debug
make clean
make lint
```

`make lint` runs both `flake8` and `mypy` with the flags required by the subject while excluding the provided SDK and virtual environment.

## Example usage

Given a definition such as:

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": {"type": "number"},
    "b": {"type": "number"}
  }
}
```

and this request:

```text
What is the sum of 2 and 3?
```

the generated output item is:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The program generates the call only; it does not execute `fn_add_numbers`.

## Constrained decoding algorithm

### 1. Read and encode the function definitions

Each function from `functions_definition.json` is converted to a tool-style schema. Its name and definition are encoded once and stored as token IDs.

The available functions are therefore determined by the input file rather than by the source code.

### 2. Build the tool context

The encoded function definitions are inserted into a Qwen-style system/tool instruction. The user prompt is then encoded as a chat message followed by the beginning of a tool call:

```text
<tool_call>
{"name": "
```

The model is not asked to freely generate the complete JSON response.

### 3. Select the function with the LLM

Every available function name is represented as a token sequence. `next_option()` performs prefix-constrained selection over these sequences.

At each generation step:

1. collect the next token of every function name that is still possible;
2. use those token IDs as the allowed mask;
3. ask the LLM for logits;
4. choose the highest-logit allowed token;
5. discard function-name sequences that no longer match;
6. repeat until one complete function name has been generated.

For example, with these available functions:

```text
fn_greet
fn_add_numbers
fn_reverse_string
```

the model can only follow token paths belonging to one of those names. It cannot invent a fourth function name.

This keeps function selection model-driven while enforcing the list of functions supplied by the schema.

### 4. Mask invalid logits

When a token mask is provided, the full logit vector is copied and every disallowed token is replaced by negative infinity. The original score is preserved for allowed tokens.

Conceptually:

```text
                 token A   token B   token C
model logits       9.1       7.4       5.0
allowed             no       yes       yes
masked logits      -inf       7.4       5.0
```

`argmax` then chooses token B because it is the highest-scoring valid choice.

### 5. Generate schema-compatible arguments

After function selection, argument generation follows the parameter order and type from the selected function definition.

#### Boolean

Boolean generation is constrained to the encoded sequences for:

```text
true
false
```

The LLM chooses between these two valid JSON values.

#### Number and integer

Numeric generation uses a restricted character set. Only digits and the characters valid for the requested numeric type are available. The generator also tracks whether a digit has already appeared and whether a decimal point has already been used.

For `integer`, the decimal point is not allowed. For `number`, a generated integer value is normalized with `.0` when necessary.

This prevents a numeric parameter from becoming arbitrary text.

#### String

Strings need semantic generation, so they are generated token by token from a focused argument context. That context contains the selected function, its description, the original user request, the current parameter name, and its type.

The string generator stops when it finds an unescaped closing quote. A quote preceded by an odd number of backslashes is treated as escaped content rather than the end of the string.

This matters for inputs containing quotes, backslashes, templates, SQL queries, regular expressions, and Windows paths.

For `regex` and `replacement` parameters, the focused context also clarifies the role of the parameter: a regex should represent the pattern described by the request, while a replacement should represent the actual requested replacement text or symbol. These instructions are general and do not contain answers for individual test prompts.

### 6. Serialize and validate the result

Generated string values are passed through JSON decoding/encoding so quotes and backslashes are represented safely. The complete internal tool-call fragment is parsed with `json.loads()` before the final object is returned.

The final output format is:

```json
{
  "prompt": "original request",
  "name": "selected_function",
  "parameters": {}
}
```

This separates structural validity from semantic accuracy: constrained decoding protects the structure and types, while the LLM is still responsible for understanding what value the user intended.

## Design decisions

### Dynamic function selection

Function names are never selected through a chain of `if` statements or keyword rules. The definitions are loaded at runtime and the LLM chooses among their encoded names using logits. This is important because evaluation can replace both the prompts and the function set.

### Greedy decoding

The highest-scoring allowed token is selected with `argmax`. Greedy decoding keeps the behavior simple and deterministic for a fixed model and context and avoids introducing sampling randomness into constrained choices.

### Separate structural and semantic constraints

Different argument types need different levels of control. Booleans and numbers have small, well-defined valid token spaces and can be strongly constrained. Free-form strings cannot be reduced to a small fixed list without losing information, so their generation uses the LLM with a focused context and JSON-safe serialization.

### Custom vocabulary encoder

The project reads the vocabulary path through the public SDK and implements its own encoding and decoding helpers. A trie provides longest vocabulary matches without depending on tokenizer internals.

### Small modules with clear responsibilities

Tokenization, raw-logit handling, function schema representation, function-call generation, and command-line I/O are kept in separate modules. This makes each stage of the generation pipeline easier to inspect and test independently.

## Challenges faced

### Keeping JSON valid while using a language model

Free generation can easily produce extra prose, malformed quotes, or invalid values. The JSON skeleton is therefore built by the program and the LLM is used only where a decision or value is required. Function names, booleans, and numbers are constrained instead of being freely generated.

### Multi-token function names

A function name is not guaranteed to be represented by one token. Comparing only the first token is not sufficient when names share prefixes or are split by the vocabulary. Prefix-constrained option selection was used so complete encoded sequences can be compared token by token.

### Numeric generation

Numbers need to remain valid JSON while still allowing negative values and decimals. The numeric generator tracks the state of the value so a minus sign cannot appear in the middle of a number and a decimal point cannot be repeated.

### Quotes and backslashes in strings

A simple search for the next `"` is not enough because the quote may be escaped. The string generator counts the consecutive backslashes before a quote to decide whether it closes the string.

JSON serialization is then used to preserve values such as Windows paths, regex escapes, and quoted templates.

### Small-model semantic accuracy

Structural constraints do not automatically make the model understand every request correctly. During testing, the model sometimes produced a semantically plausible but incorrect string value. Focused per-argument context was introduced to give the model the function description, request, parameter name, and type without hardcoding the expected answer.

Two difficult examples remain useful demonstrations of this limitation:

- a Unix path can lose its leading `/` even though the rest of the path is correct;
- a request to replace vowels with asterisks can produce a semantically related regex or the word `asterisk` instead of the exact expected pattern and `*` replacement.

These cases are valid structurally but show that semantic extraction with a 0.6B model is a separate problem from JSON validity.

## Performance analysis

The main startup cost is loading the language model. After that, each constrained generation step requires a logits call, so runtime depends on the number of prompts and the number of generated tokens.

The implementation does not use batching or logits caching. The priority for the mandatory part is keeping the decoding logic understandable and reliable.

With the required `Qwen/Qwen3-0.6B` model, the latest local Moulinette runs produced:

```text
Public set:  10 / 11 correct (90.9%)
Private set: 10 / 11 correct (90.9%)
```

All generated result files in these runs were parseable JSON. The remaining failures were argument-extraction errors rather than failures to produce the required JSON structure.

The tests also complete comfortably within the subject's five-minute limit on the development machine.

Accuracy can still depend on the prompt and on the model. The constraints guarantee what the decoder explicitly restricts; they do not turn semantic interpretation into a deterministic parser.

## Testing strategy

Testing was done incrementally instead of changing several parts of the decoder at once.

The main checks were:

1. run `uv sync` from a clean environment;
2. run the default public prompts with `uv run python -m src`;
3. parse and inspect `data/output/function_calling_results.json`;
4. verify function names and parameter types against the definitions;
5. run the supplied Moulinette on the public set;
6. run the supplied Moulinette on the private set;
7. retest previously passing cases after changes to string generation;
8. run `make lint` to check `flake8` and `mypy`.

Special attention was given to:

- integers and decimal numbers;
- multiple numeric arguments;
- quoted strings;
- SQL queries;
- escaped quotes;
- Unix and Windows paths;
- regex strings and backslashes;
- templates containing braces;
- function names represented by multiple tokens.

The output file is also parsed again inside the program before each final result is returned, which catches malformed internal tool-call JSON early.

## Error handling

The command-line entry point handles missing files and malformed JSON explicitly. It also catches unexpected exceptions and prints a readable error message.

Function definitions with unsupported parameter types or duplicate names are rejected instead of silently producing an invalid call.

The output directory is created automatically when needed.

## Resources

References used while working on the project:

- Call Me Maybe subject and evaluation sheet
- Python documentation, especially `json`, file handling, and `argparse`
- Pydantic documentation
- NumPy documentation
- Qwen model/tokenizer documentation
- Hugging Face model documentation
- `uv` documentation
- the project-provided `llm_sdk`

### AI usage

AI tools were used as a development aid for technical explanations, comparing possible approaches, reviewing debugging output, and improving documentation. Generated suggestions were checked against the subject, the code, and local test results before being kept. The implementation and its behavior were reviewed and tested locally so that every retained part could be explained during evaluation.
