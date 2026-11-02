# bielik-tools

A collection of supplementary tools for working with **Bielik** language models.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/speakleash/bielik-tools.git
cd bielik-tools
```

Explore available tools and examples in their respective subdirectories.

## Install as a package

The parsers can also be installed as a vLLM plugin into the same environment as vLLM
(vLLM itself is not installed by this package):

```bash
pip install git+https://github.com/speakleash/bielik-tools
```

The plugin registers the `bielik` tool and reasoning parsers automatically and picks the
parser files matching the installed vLLM version (see the table below), so no
`--tool-parser-plugin` / `--reasoning-parser-plugin` paths are needed. The advanced chat
template ships with the package:

```bash
vllm serve speakleash/Bielik-11B-v3.0-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser bielik \
    --reasoning-parser bielik \
    --chat-template "$(python -m bielik_tools template-path)"
```

`python -m bielik_tools info` shows the detected vLLM version and the selected parser files.
Passing `--tool-parser-plugin` / `--reasoning-parser-plugin` explicitly still works and takes
precedence over the packaged parsers.

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

The 0.24.0 tool-parser cut is the vLLM `ToolParser` API change
(`__init__(tokenizer, tools=…)` plus `supports_required_and_named`).
Reasoning parsers are unchanged across 0.15+.

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
