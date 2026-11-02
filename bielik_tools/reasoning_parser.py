"""Importable entry point for the Bielik reasoning parser.

Importing this module loads the reasoning parser file matching the installed
vLLM version and re-exports its ``BielikReasoningParser`` class. vLLM imports
it lazily (the first time ``--reasoning-parser bielik`` is requested).
"""

from bielik_tools import reasoning_parser_path
from bielik_tools._loader import load_file_module

_path = reasoning_parser_path()
if _path is None:
    raise ImportError(
        "No Bielik reasoning parser is available for the installed vLLM version."
    )

_impl = load_file_module(f"{__name__}._impl", _path)

BielikReasoningParser = _impl.BielikReasoningParser

__all__ = ["BielikReasoningParser"]
