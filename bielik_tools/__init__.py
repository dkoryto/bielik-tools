"""Bielik tool-calling and reasoning parsers packaged as a vLLM plugin.

Installing this package registers the ``bielik`` tool parser and the
``bielik`` reasoning parser with vLLM through the ``vllm.general_plugins``
entry point, so ``--tool-parser-plugin`` / ``--reasoning-parser-plugin`` are
no longer required. The parser implementation matching the installed vLLM
version is selected automatically from the files in ``tools/``.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "chat_template_path",
    "data_dir",
    "reasoning_parser_path",
    "select_parser_files",
    "tool_parser_path",
    "vllm_version",
]

__version__ = "0.1.0"

CHAT_TEMPLATE_FILE = "bielik_advanced_chat_template.jinja"


def data_dir() -> Path:
    """Return the directory holding the parser files and the chat template.

    In a wheel install the files from ``tools/`` are shipped inside the
    package (``bielik_tools/tools``); in a source checkout or editable install
    the repository's top-level ``tools/`` directory is used.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / "tools", here.parent / "tools"):
        if (candidate / CHAT_TEMPLATE_FILE).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the bielik-tools data directory "
        f"(looked in {here / 'tools'} and {here.parent / 'tools'})."
    )


def chat_template_path() -> Path:
    """Return the path of the advanced Bielik chat template."""
    return data_dir() / CHAT_TEMPLATE_FILE


def _parse_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    if not match:
        raise ValueError(f"Unrecognised vLLM version string: {version!r}")
    return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]


def vllm_version() -> tuple[int, int, int] | None:
    """Return the installed vLLM version as ``(major, minor, patch)``.

    Returns ``None`` when vLLM is not installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        raw = version("vllm")
    except PackageNotFoundError:
        try:
            import vllm  # noqa: PLC0415
        except ImportError:
            return None
        raw = getattr(vllm, "__version__", "0")
    return _parse_version(raw)


def select_parser_files(
    version: tuple[int, int, int],
) -> tuple[str, str | None]:
    """Return ``(tool_parser_file, reasoning_parser_file)`` for a vLLM version.

    Mirrors the compatibility table in the README. The reasoning parser file is
    ``None`` for vLLM versions without a compatible reasoning parser.
    """
    if version >= (0, 24, 0):
        return "bielik_vllm_tool_parser.py", "bielik_vllm_reasoning_parser.py"
    if version >= (0, 15, 0):
        return (
            "bielik_vllm_tool_parser_v0.15.0.py",
            "bielik_vllm_reasoning_parser.py",
        )
    if version >= (0, 13, 0):
        return (
            "bielik_vllm_tool_parser_v0.13.0.py",
            "bielik_vllm_reasoning_parser_v0.13.0.py",
        )
    return "bielik_vllm_tool_parser_v0.12.0.py", None


def _require_version() -> tuple[int, int, int]:
    version = vllm_version()
    if version is None:
        raise RuntimeError("vLLM is not installed in this environment.")
    return version


def tool_parser_path(version: tuple[int, int, int] | None = None) -> Path:
    """Return the tool parser file matching the (installed) vLLM version."""
    tool_file, _ = select_parser_files(version or _require_version())
    return data_dir() / tool_file


def reasoning_parser_path(version: tuple[int, int, int] | None = None) -> Path | None:
    """Return the reasoning parser file for the (installed) vLLM version.

    Returns ``None`` if no reasoning parser supports that version.
    """
    _, reasoning_file = select_parser_files(version or _require_version())
    return data_dir() / reasoning_file if reasoning_file else None
