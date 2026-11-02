"""Importable entry point for the Bielik tool parser.

Importing this module loads the tool parser file matching the installed vLLM
version and re-exports its ``BielikToolParser`` class. vLLM imports it lazily
(the first time ``--tool-call-parser bielik`` is requested).
"""

from bielik_tools import tool_parser_path
from bielik_tools._loader import load_file_module

_impl = load_file_module(f"{__name__}._impl", tool_parser_path())

BielikToolParser = _impl.BielikToolParser

__all__ = ["BielikToolParser"]
