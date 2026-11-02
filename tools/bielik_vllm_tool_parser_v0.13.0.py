"""Bielik tool parser plugin for vLLM 0.13.0 - 0.14.x (compatibility shim).

The implementation lives in bielik_vllm_tool_parser.py (same directory);
this file loads it by path and registers ``BielikPre015ToolParser`` as ``bielik``.
Keeps this file's historical behaviour: tool_choice="required" is left to
vLLM's built-in handling (no guided-regex redirect).
"""
import importlib.util
import sys
from pathlib import Path

_SHARED_FILE = Path(__file__).resolve().parent / "bielik_vllm_tool_parser.py"
# Must match SHARED_MODULE_NAME in bielik_vllm_tool_parser.py: loading the
# file under this name skips its own registration.
_SHARED_MODULE_NAME = "_bielik_vllm_tool_parser_shared"


def _load_shared():
    module = sys.modules.get(_SHARED_MODULE_NAME)
    if module is not None:
        return module
    if not _SHARED_FILE.is_file():
        raise ImportError(
            f"{Path(__file__).name} needs {_SHARED_FILE.name} in the same "
            f"directory (not found: {_SHARED_FILE})")
    spec = importlib.util.spec_from_file_location(_SHARED_MODULE_NAME, _SHARED_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the Bielik tool parser from {_SHARED_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SHARED_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_SHARED_MODULE_NAME]
        raise
    return module


_shared = _load_shared()

BielikToolParser = _shared.register_tool_parser(_shared.BielikPre015ToolParser)
