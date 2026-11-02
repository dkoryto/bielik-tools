"""vLLM general plugin registering the Bielik tool and reasoning parsers.

Exposed through the ``vllm.general_plugins`` entry point. vLLM may call
:func:`register` in several processes (API server, engine core, workers) and
more than once per process, so registration is idempotent.

On vLLM versions whose parser managers provide ``register_lazy_module`` the
parsers are registered lazily: nothing is imported until vLLM actually asks
for the ``bielik`` parser. Older versions fall back to importing the parser
file eagerly. A parser file passed explicitly via ``--tool-parser-plugin`` /
``--reasoning-parser-plugin`` is loaded after general plugins and therefore
still takes precedence.
"""

from __future__ import annotations

import logging

from bielik_tools import select_parser_files, vllm_version

logger = logging.getLogger(__name__)

PARSER_NAME = "bielik"
TOOL_PARSER_MODULE = "bielik_tools.tool_parser"
TOOL_PARSER_CLASS = "BielikToolParser"
REASONING_PARSER_MODULE = "bielik_tools.reasoning_parser"
REASONING_PARSER_CLASS = "BielikReasoningParser"


def _tool_parser_manager(version: tuple[int, int, int]):
    if version >= (0, 13, 0):
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    else:
        from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
            ToolParserManager,
        )
    return ToolParserManager


def _register(manager, registry_attr: str, module_path: str, class_name: str) -> None:
    # Already registered in this process (by an earlier call, or by a parser
    # file the user loaded explicitly): keep the existing registration.
    if PARSER_NAME in getattr(manager, registry_attr, {}):
        return
    if PARSER_NAME in getattr(manager, "lazy_parsers", {}):
        return
    if hasattr(manager, "register_lazy_module"):
        manager.register_lazy_module(PARSER_NAME, module_path, class_name)
        return
    import importlib

    parser_cls = getattr(importlib.import_module(module_path), class_name)
    manager.register_module(PARSER_NAME, module=parser_cls)


def register() -> None:
    """Register the ``bielik`` tool and reasoning parsers with vLLM."""
    version = vllm_version()
    if version is None:
        logger.warning("bielik-tools: vLLM is not installed; nothing to register.")
        return

    _, reasoning_file = select_parser_files(version)

    try:
        _register(
            _tool_parser_manager(version),
            "tool_parsers",
            TOOL_PARSER_MODULE,
            TOOL_PARSER_CLASS,
        )
    except Exception:
        logger.exception("bielik-tools: failed to register the Bielik tool parser.")

    if reasoning_file is None:
        logger.debug(
            "bielik-tools: no reasoning parser available for vLLM %s.",
            ".".join(map(str, version)),
        )
        return

    try:
        from vllm.reasoning import ReasoningParserManager

        _register(
            ReasoningParserManager,
            "reasoning_parsers",
            REASONING_PARSER_MODULE,
            REASONING_PARSER_CLASS,
        )
    except Exception:
        logger.exception(
            "bielik-tools: failed to register the Bielik reasoning parser."
        )
