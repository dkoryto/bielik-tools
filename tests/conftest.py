"""Shared fixtures for Bielik vLLM parser tests.

The parsers are vLLM plugins (standalone files loaded via
``--tool-parser-plugin``), so they are imported here by file path.
Tests are skipped when vLLM is not installed.
"""
import importlib.util
from pathlib import Path

import pytest

try:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
except ImportError:
    # vLLM is optional: skip only the parser tests, keep other tests runnable.
    ChatCompletionRequest = None
    collect_ignore = ["test_bielik_vllm_tool_parser.py", "test_plugin_loading.py"]

TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"

TOOL_CALL_START_ID = 1
TOOL_CALL_END_ID = 2
TEXT_TOKEN_ID = 100

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


class FakeTokenizer:
    """Minimal tokenizer: the parser only needs ``get_vocab()``."""

    def __init__(self, vocab=None):
        self._vocab = vocab if vocab is not None else {
            "<tool_call>": TOOL_CALL_START_ID,
            "</tool_call>": TOOL_CALL_END_ID,
        }

    def get_vocab(self):
        return self._vocab


def _load_module(filename):
    path = TOOLS_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Every tool parser plugin file users may pass to --tool-parser-plugin, with
# the tool_choice="required" behaviour it is expected to keep:
#   "default" - redirect to "auto" + guided regex (single <tool_call> block)
#   "v3.1"    - redirect to "auto" + permissive v3.1 regex
#   "builtin" - leave "required" to vLLM's built-in handling
TOOL_PARSER_FILES = {
    "bielik_vllm_tool_parser.py": "default",
    "bielik_vllm_tool_parser_v0.15.0.py": "default",
    "bielik_vllm_tool_parser_v0.13.0.py": "default",
    "bielik_vllm_tool_parser_v0.12.0.py": "builtin",
    "bielik_vllm_tool_parser_v3.1.py": "v3.1",
}


@pytest.fixture(scope="session", params=list(TOOL_PARSER_FILES))
def tool_parser_file(request):
    return request.param


@pytest.fixture(scope="session")
def required_mode(tool_parser_file):
    return TOOL_PARSER_FILES[tool_parser_file]


@pytest.fixture(scope="session")
def tool_parser_module(tool_parser_file):
    return _load_module(tool_parser_file)


@pytest.fixture
def parser(tool_parser_module):
    return tool_parser_module.BielikToolParser(FakeTokenizer())


@pytest.fixture
def make_request():
    def _make(tools=(WEATHER_TOOL,), tool_choice=None, **kwargs):
        params = {
            "model": "speakleash/Bielik-11B-v3.0-Instruct",
            "messages": [{"role": "user", "content": "Jaka jest pogoda?"}],
            **kwargs,
        }
        if tools:
            params["tools"] = list(tools)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        return ChatCompletionRequest(**params)

    return _make


def stream_chunks(parser, request, chunks):
    """Feed text chunks to ``extract_tool_calls_streaming`` like vLLM does.

    ``<tool_call>`` / ``</tool_call>`` must be passed as separate chunks so
    they map to their special token ids. Returns the list of deltas.
    """
    previous_text, previous_ids, deltas = "", [], []
    for chunk in chunks:
        if chunk == "<tool_call>":
            delta_ids = [TOOL_CALL_START_ID]
        elif chunk == "</tool_call>":
            delta_ids = [TOOL_CALL_END_ID]
        else:
            delta_ids = [TEXT_TOKEN_ID] * max(1, len(chunk) // 3)
        current_text = previous_text + chunk
        current_ids = previous_ids + delta_ids
        deltas.append(
            parser.extract_tool_calls_streaming(
                previous_text, current_text, chunk,
                previous_ids, current_ids, delta_ids, request,
            ))
        previous_text, previous_ids = current_text, current_ids
    return deltas


def collect_stream(deltas):
    """Merge streamed deltas into ``(content, {index: {id, name, arguments}})``."""
    content, calls = "", {}
    for delta in deltas:
        if delta is None:
            continue
        if delta.content:
            content += delta.content
        for tc in delta.tool_calls or []:
            call = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id:
                call["id"] = tc.id
            if tc.function is not None:
                call["name"] += tc.function.name or ""
                call["arguments"] += tc.function.arguments or ""
    return content, calls
