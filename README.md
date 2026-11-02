# bielik-tools

A collection of supplementary tools for working with **Bielik** language models.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/speakleash/bielik-tools.git
cd bielik-tools
```

Explore available tools and examples in their respective subdirectories.

## Structured Outputs

Bielik models have been trained to generate structured outputs. Once the model is running in [vLLM](https://github.com/vllm-project/vllm), you can try the [structured_output.py](https://github.com/speakleash/bielik-tools/blob/main/examples/structured_output.py) example to generate structured outputs using OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat](https://platform.openai.com/docs/api-reference/chat) APIs.

## Tool Calling

To use function/tool calling, you need to enable the extended chat template. This can be done using the provided [advanced chat template](https://github.com/speakleash/bielik-tools/blob/main/tools/bielik_advanced_chat_template.jinja) and [tool parser](https://github.com/speakleash/bielik-tools/blob/main/tools/bielik_vllm_tool_parser.py). Start vLLM with the following command:

```bash
vllm serve Bielik-11B-v2.5-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./bielik-tools/tools/bielik_vllm_tool_parser.py \
    --tool-call-parser bielik \
    --chat-template ./bielik-tools/tools/bielik_advanced_chat_template.jinja
```

Choose the tool parser file matching your vLLM version:

| vLLM version | Tool parser | Reasoning parser |
|---|---|---|
| ≥ 0.24.0 | `bielik_vllm_tool_parser.py` | `bielik_vllm_reasoning_parser.py` |
| 0.15.0 – 0.23.x | `bielik_vllm_tool_parser_v0.15.0.py` | `bielik_vllm_reasoning_parser.py` |
| 0.13.0 – 0.14.x | `bielik_vllm_tool_parser_v0.13.0.py` | `bielik_vllm_reasoning_parser_v0.13.0.py` |
| ≤ 0.12.0 | `bielik_vllm_tool_parser_v0.12.0.py` | — |

All tool parser files share one implementation, `bielik_vllm_tool_parser.py`,
which resolves vLLM imports and the `ToolParser` constructor signature at load
time. The versioned files are thin wrappers kept for backwards compatibility;
they must stay in the same directory as `bielik_vllm_tool_parser.py`. They
differ only in how `tool_choice="required"` is handled:

* `bielik_vllm_tool_parser.py`, `_v0.15.0.py` and `_v0.13.0.py` redirect it
  to `auto` and force a `<tool_call>` block with a guided regex;
* `_v3.1.py` does the same with a more permissive regex (text around calls);
* `_v0.12.0.py` leaves it to vLLM's built-in handling, as before.

The same applies to the reasoning parsers: `bielik_vllm_reasoning_parser.py`
holds the implementation and `_v0.13.0.py` is a wrapper with identical
behaviour.

For the Bielik v3.1 chat template use `bielik_vllm_tool_parser_v3.1.py` and
`bielik_vllm_reasoning_parser_v3.1.py` (vLLM ≥ 0.24). They allow free text
around tool calls when `tool_choice="required"` and enable thinking via
`reasoning_effort` in `chat_template_kwargs`. Load only one Bielik tool parser
and one reasoning parser plugin per server — all of them register as `bielik`.

The unit tests (see [Tests](#tests)) run against vLLM 0.25.1; older vLLM
versions are handled by import fallbacks but are not covered by automated
tests.

Then, run [tool\_calling.py](https://github.com/speakleash/bielik-tools/blob/main/examples/tool_calling.py) or [tool\_calling\_streaming.py](https://github.com/speakleash/bielik-tools/blob/main/examples/tool_calling_streaming.py) to see how tool calling works in practice.

## Reasoning

Reasoning is available in Bielik 11B v2.5 Instruct and Bielik v3.0 Instruct models. Enabling reasoning allows the model to better handle complex questions by expanding its reasoning capabilities. To try it out, start vLLM with the following command:

```bash
vllm serve speakleash/Bielik-11B-v3.0-Instruct \
    --chat-template ./bielik-tools/tools/bielik_advanced_chat_template.jinja \
    --reasoning-parser-plugin ./bielik-tools/tools/bielik_vllm_reasoning_parser.py \
    --reasoning-parser bielik
```

Then, run [reasoning\_streaming.py](https://github.com/speakleash/bielik-tools/blob/main/examples/reasoning_streaming.py) to see how the model performs in reasoning mode.

## Tests

Unit tests for the parsers live in `tests/`. They exercise every tool parser file, load each tool and reasoning parser file the way vLLM does (`--tool-parser-plugin` / `--reasoning-parser-plugin`) and import vLLM, so run them in an environment where vLLM (>= 0.24) is installed; no GPU or model download is needed (a fake tokenizer is used):

```bash
pip install pytest
python -m pytest tests -q
```

## Multi-Agent with CrewAI

For this example to work you need Tavily API key. Create `.env` file with contents:

```
BASE_URL=http://0.0.0.0:8000/v1
MODEL_NAME=path_or_hf_repo
API_KEY=token-abc123
TAVILY_API_KEY=tvly-apikey123123123123
```

Run vllm

```bash
vllm serve speakleash/Bielik-11B-v3.0-Instruct \
    --enable-auto-tool-choice \
    --tool-parser-plugin ./bielik-tools/tools/bielik_vllm_tool_parser.py \
    --tool-call-parser bielik \
    --chat-template ./bielik-tools/tools/bielik_advanced_chat_template.jinja \
    --port 8000 --host 0.0.0.0 \
    --api-key token-abc123
```

Then, run [crewai_to_file.py](https://github.com/speakleash/bielik-tools/blob/main/examples/crewai_to_file.py)  
Final contens of report will be placed in `bielik_output/atrakcje.md` 
