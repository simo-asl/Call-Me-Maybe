*This project has been created as part of the 42 curriculum by mel-asla.*

# Call Me Maybe

## Description

Call Me Maybe is an introduction to LLM function calling. It reads function
definitions and user prompts, asks the supplied small LLM to select a function
and its arguments, then writes structured function calls to JSON. It does not
execute the selected functions.

The default model is `Qwen/Qwen3-0.6B`, accessed exclusively through the
provided `llm_sdk` public API.

## Instructions

Requirements are Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
make install
make run
```

The default command is equivalent to:

```bash
uv run python -m src
```

It reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, then creates
`data/output/function_calling_results.json`.

Input and output paths can be changed:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Useful development commands are `make debug`, `make lint`, and `make clean`.

## Example usage

For a prompt such as `What is the sum of 40 and 2?`, a result has this shape:

```json
{
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40, "b": 2}
}
```

Every output item has exactly `prompt`, `name`, and `parameters` keys.

## Design and constrained decoding

`src/input_loader.py` is responsible only for JSON files, `src/models.py`
validates external data with Pydantic, and `src/service.py` coordinates one
call per prompt. `src/decoder.py` is intentionally the only module that knows
about LLM tokens and constraints.

The decoder builds an instruction containing the dynamically loaded schemas.
For every generated token, it obtains logits through
`get_logits_from_input_ids` and considers only vocabulary tokens that preserve
the current grammar state. The fixed JSON object structure and parameter keys
are emitted from the chosen schema. Function-name tokens must be a prefix of a
loaded function name, so the LLM selects from real definitions rather than a
keyword rule. String tokens are checked for valid JSON escaping; number tokens
must remain a valid JSON-number prefix; booleans are restricted to `true` or
`false`. This prevents extra keys, incorrect function names, invalid types,
and prose from being generated. A final schema validation is a defensive
second check before output is written.

This keeps the approach explicit rather than relying on “please output JSON”
prompting. It also means input definitions are never hardcoded.

## Performance, accuracy, speed, and reliability

The decoder uses greedy selection among schema-valid tokens. Loading the model
is normally the dominant cost; generation also requires one logits call per
selected token. Exact timing and task-accuracy benchmarks have not been
measured in this repository, so no numerical performance or accuracy claims
are made here.

Reliability is addressed structurally: invalid JSON syntax and invalid schema
paths are masked before a model token is chosen, and generated values are
checked again before serialization. Semantic quality—choosing the intended
function and arguments—still depends on the model and prompt clarity.

## Challenges and testing strategy

The main challenge is that tokenizer entries can include several characters,
so each candidate token is checked as a complete fragment rather than assuming
one token equals one character. JSON string escapes and number endings require
separate lexical state handling.

The intended lightweight checks are:

```bash
uv sync
make lint
uv run python -m src --help
```

The program also reports missing files, malformed JSON, malformed definitions,
unsupported types, vocabulary problems, and generation failures as clear
command-line errors instead of writing invalid output.

## Resources

- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [uv documentation](https://docs.astral.sh/uv/)
- The project-provided `llm_sdk` API and the Call Me Maybe subject

## AI usage

AI was used as a programming assistant to analyse the provided read-only
reference behaviour, propose a clearer file layout, review implementation
details, and help write this documentation. The final implementation was
reviewed locally with the project checks listed above.
