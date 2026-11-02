"""Reasoning parser for Bielik models, usable as a vLLM ``--reasoning-parser-plugin``.

This file holds the single implementation of the Bielik reasoning parser. It
is self-contained and resolves vLLM imports from the newest known module
location first, falling back to older locations only when that specific
module is missing.

The versioned files next to this one (``bielik_vllm_reasoning_parser_v*.py``)
are thin shims kept for backwards compatibility; they load this file and
register the matching variant (see ``BielikV31ReasoningParser`` below).

The parser is registered under the name ``bielik``.
"""
import importlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike

# Module name used by the versioned shims when they load this file. When
# loaded under this name the module only defines the parser classes and
# leaves registration to the shim, so "bielik" is registered exactly once.
SHARED_MODULE_NAME = "_bielik_vllm_reasoning_parser_shared"


# --------------------------------------------------------------------------
# vLLM compatibility layer
# --------------------------------------------------------------------------

class _CompatImportError(ImportError):
    """None of the candidate modules provides the requested names."""


def _import_from(candidates: Sequence[str], *names: str) -> tuple:
    """Import ``names`` from the first module in ``candidates`` providing them.

    Candidates are tried newest-first. A candidate is skipped only when the
    candidate module itself (or one of its parent packages) does not exist,
    or when it lacks one of the names; any other import failure is re-raised.
    """
    last_error: Optional[BaseException] = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if not missing or not (module_name == missing
                                   or module_name.startswith(missing + ".")):
                raise
            last_error = exc
            continue
        if all(hasattr(module, name) for name in names):
            return tuple(getattr(module, name) for name in names)
    raise _CompatImportError(
        f"Could not import {', '.join(names)} from any of: "
        f"{', '.join(candidates)}") from last_error


# vllm.entrypoints.openai.protocol was split up in vLLM 0.15; newer
# versions move the shared response types to vllm.entrypoints.generate.
(DeltaMessage,) = _import_from(
    ("vllm.entrypoints.generate.base.protocol",
     "vllm.entrypoints.openai.engine.protocol",
     "vllm.entrypoints.openai.protocol"),
    "DeltaMessage")


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

class BielikReasoningParser(ReasoningParser):
    """
    Reasoning parser for Bielik models.

    Delegates to DeepSeekR1ReasoningParser when thinking is enabled
    (``enable_thinking=True`` in ``chat_template_kwargs``), and to
    IdentityReasoningParser otherwise — so that the entire model output
    is returned as content, not reasoning.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args: Any, **kwargs: Any):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        enable_thinking = self.thinking_enabled(chat_kwargs)

        if enable_thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    @staticmethod
    def thinking_enabled(chat_kwargs: dict) -> bool:
        """Whether ``chat_template_kwargs`` turn thinking on."""
        return bool(chat_kwargs.get("enable_thinking", False))

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self,
        model_output: str,
        request: Union["ChatCompletionRequest", "ResponsesRequest"],
    ) -> tuple[Optional[str], Optional[str]]:
        return self._parser.extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> Optional[DeltaMessage]:
        return self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )


# --------------------------------------------------------------------------
# Variants kept for the versioned plugin files
# --------------------------------------------------------------------------

class BielikV31ReasoningParser(BielikReasoningParser):
    """
    Reasoning parser for Bielik models with the v3.1 chat template.

    Thinking is enabled when ``chat_template_kwargs`` has
    ``reasoning_effort`` set to anything other than ``"none"``. If
    ``reasoning_effort`` is absent, falls back to the legacy
    ``enable_thinking=True`` flag (advanced template).
    """

    @staticmethod
    def thinking_enabled(chat_kwargs: dict) -> bool:
        effort = chat_kwargs.get("reasoning_effort")
        if effort is not None:
            return str(effort).lower() != "none"
        return bool(chat_kwargs.get("enable_thinking", False))


def register_reasoning_parser(parser_cls: type, name: str = "bielik") -> type:
    """Register ``parser_cls`` with vLLM under ``name``."""
    return ReasoningParserManager.register_module(name)(parser_cls)


if __name__ != SHARED_MODULE_NAME:
    register_reasoning_parser(BielikReasoningParser)
