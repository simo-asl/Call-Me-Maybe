*This project has been created as part of the 42 curriculum by mel-asla.*

# Call Me Maybe

## Description

Call Me Maybe is a small function-calling system built around a local language
model. The program reads a list of available function definitions and a list of
natural-language prompts. For every prompt, the LLM chooses a function and its
arguments, and the program writes the result as JSON.

The project focuses on **constrained decoding**. Instead of asking the model to
freely generate JSON and hoping that the result is valid, the program reads the
model's raw next-token logits and masks choices that are not allowed. The model
therefore makes decisions only between valid candidates.

The default model is `Qwen/Qwen3-0.6B`, accessed through the provided
`llm_sdk`. The selected functions are described by the input JSON file; their
names are not hardcoded in the function-selection logic.

The program generates function calls only. It does not execute the functions.

## Project structure

```text
src/
├── __init__.py
├── __main__.py
├── constrained_llm.py
├── function_caller.py
├── function_schema.py
└── token_encoder.py
```

- `__main__.py` parses arguments, loads the model and input files, and writes
  the output file.
- `token_encoder.py` builds a trie from the model vocabulary and provides the
  encoding/decoding operations used by the project.
- `function_schema.py` stores a function definition and pre-encodes its name,
  description, parameter types, and tool representation.
- `constrained_llm.py` obtains raw logits from `llm_sdk`, masks forbidden token
  IDs, and selects the highest-scoring allowed token.
- `function_caller.py` selects the function, builds constrained argument
  candidates, and produces the final function-call JSON.

## Instructions

The project requires Python 3.10 or newer and `uv`.

Install dependencies:

```bash
make install
```

Run with the default files and the subject model:

```bash
make run
```

This is equivalent to:

```bash
uv run python -m src
```

By default the program reads:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
```

and generates:

```text
data/output/function_calling_results.json
```

Custom paths can be supplied from the command line:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The default model is `Qwen/Qwen3-0.6B`. A compatible model can also be selected
with:

```bash
uv run python -m src --model Qwen/Qwen2.5-0.5B
```

The alternative model option is useful for experimentation; semantic accuracy
can differ between models.

Other useful commands:

```bash
make lint
make debug
make clean
```

## Example usage

Given a function definition similar to:

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

and the prompt:

```text
What is the sum of 2 and 3?
```

the generated item has the following form:

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

Every output object contains exactly `prompt`, `name`, and `parameters`.

## Constrained decoding algorithm

### 1. Load and encode the tools

The function definitions are read from JSON. Each definition is converted into
a tool-style schema and encoded with the project encoder. Function names are
also encoded separately because they become constrained generation choices.

### 2. Build the model context

The prompt and available tool definitions are placed in a Qwen-style chat
context. The program does not ask the SDK to generate a complete answer.
Instead, it calls `get_logits_from_input_ids()` whenever a model decision is
needed.

### 3. Constrain function selection

Every available function name is represented as a token sequence. At each
position, only token IDs that can continue at least one remaining function name
are allowed.

For example, if the valid choices are conceptually:

```text
fn_greet
fn_add_numbers
fn_reverse_string
```

only next tokens belonging to one of these sequences remain selectable. After
the model chooses a token, incompatible sequences are removed. This continues
until one complete function name remains.

The actual model decision is still made from its logits; the constraint only
prevents it from selecting a function that is not available.

### 4. Mask invalid logits

`constrained_llm.py` receives the full next-token logits from the SDK. Every
disallowed token receives negative infinity, while allowed logits keep their
original scores. `argmax` then returns the highest-scoring valid token.

Conceptually:

```text
model logits:       A=9.1   B=7.4   C=5.0
allowed tokens:       no     yes     yes
masked logits:      A=-inf  B=7.4   C=5.0
selected token:             B
```

### 5. Constrain argument values

After a function is selected, its parameter names and types come from its
schema. Candidate values are extracted from the prompt and encoded. The same
`next_option()` mechanism lets the LLM select only between those candidates.

The implementation includes small type-aware rules:

- booleans are restricted to `true` or `false`;
- number/float arguments are restricted to numeric values found in the prompt;
- filesystem paths are extracted as complete Unix or Windows path candidates;
- regex arguments use a small mapping for common requested character classes;
- string candidates preserve quoted text and text following a colon, which is
  useful for queries and templates containing spaces or special characters.

### 6. Produce JSON

The JSON structure, selected function name, and constrained values are combined
into the final call. The generated tool-call fragment is parsed with
`json.loads()` before the final result is written, so malformed JSON is not
silently accepted.

## Design decisions

The implementation intentionally uses a small number of modules. The goal was
to keep the constrained-decoding mechanism visible and easy to explain rather
than hide it behind a large framework.

A custom vocabulary encoder is used by the main project. It loads the model's
vocabulary JSON, builds a trie, and performs longest-match tokenization. This
also makes the relationship between text, token IDs, logits, and constrained
decoding explicit.

Function selection remains model-driven: the LLM chooses between dynamically
loaded function names using its logits. Small extraction rules are used for
argument candidates so that generated values remain compatible with the
selected parameter types.

Greedy selection (`argmax`) was chosen because it is deterministic and simple:
for a fixed model and context, the highest-scoring allowed token is selected.

## Performance analysis

Model loading is the largest startup cost. During generation, every constrained
choice requires a call to obtain next-token logits, so runtime grows with the
number of tokens needed to select function names and argument values.

The implementation favors simplicity and reliability over generation speed. It
does not use batching or logits caching.

With the required `Qwen/Qwen3-0.6B` model, the implementation was tested against
the provided private grading set and produced **11 valid results out of 11**.
This result applies to that tested set and is not a claim that every possible
natural-language prompt will be interpreted correctly.

The `--model` option was also used to run the program with
`Qwen/Qwen2.5-0.5B`. The program remained capable of producing constrained JSON,
although the smaller/different model made more semantic function-selection
mistakes. This illustrates the difference between structural validity and
semantic model accuracy.

## Challenges faced

### JSON escaping

Regex values such as `\d+` must be escaped correctly inside JSON. An early
version produced an invalid JSON escape. Keeping the regex representation JSON
safe fixed the issue.

### Multi-word string arguments

Treating every word as an independent candidate loses values such as SQL
queries and templates. Candidate extraction therefore also preserves quoted
content and complete text after a colon.

### Filesystem paths

A Unix path such as `/home/user/data.json` must be treated as one value rather
than allowing the model to select a nearby word such as `file`. Path extraction
keeps Unix and Windows paths as complete candidates.

### Type safety for numeric arguments

Allowing arbitrary prompt words for numeric parameters could create invalid JSON
such as an unquoted word in a numeric field. Numeric parameters are therefore
restricted to values matching a numeric pattern before constrained selection.

## Testing strategy

The implementation was tested in several stages:

1. Run the supplied demonstration prompts and inspect the generated JSON.
2. Check function names and argument values against their definitions.
3. Test strings containing quotes, regex escapes, multiple words, Unix paths,
   Windows paths, integers, and decimal numbers.
4. Run the private project grader with the required `Qwen/Qwen3-0.6B` model.
5. Re-run the grader after fixes to ensure previous valid cases were not broken.

The final private grading run passed all 11 tests.

Static checks can be run with:

```bash
make lint
```

which runs `flake8` and `mypy` on `src`.

## Error handling

The command-line entry point reports missing files and malformed JSON with clear
messages. Unexpected runtime errors are also caught and printed instead of
producing a partial successful result without explanation.

The caller rejects argument generation when no candidate compatible with the
required type can be found.

## Resources

- Python documentation
- Pydantic documentation
- NumPy documentation
- Hugging Face Transformers documentation
- Hugging Face Hub documentation
- `uv` documentation
- The project-provided `llm_sdk`
- The Call Me Maybe subject

## AI usage

AI tools were used during development to help compare implementation ideas,
identify edge cases, debug JSON escaping and path handling, refactor file names
and documentation, and explain transformer/logit concepts. The implementation
was then run locally against the project inputs and private grader, and the
observed failures were used to make targeted fixes.
