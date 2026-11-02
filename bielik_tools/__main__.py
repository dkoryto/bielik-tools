"""Command-line helpers.

Usage::

    python -m bielik_tools template-path   # path of the advanced chat template
    python -m bielik_tools info            # selected parser files for this vLLM
"""

from __future__ import annotations

import argparse
import sys

from bielik_tools import (
    __version__,
    chat_template_path,
    data_dir,
    select_parser_files,
    vllm_version,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bielik_tools")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template-path", help="print the advanced chat template path")
    sub.add_parser("info", help="print the parser files selected for vLLM")
    args = parser.parse_args(argv)

    if args.command == "template-path":
        print(chat_template_path())
        return 0

    version = vllm_version()
    print(f"bielik-tools {__version__}")
    print(f"data dir:          {data_dir()}")
    print(f"chat template:     {chat_template_path()}")
    if version is None:
        print("vLLM:              not installed")
        return 0
    tool_file, reasoning_file = select_parser_files(version)
    print(f"vLLM:              {'.'.join(map(str, version))}")
    print(f"tool parser:       {tool_file}")
    print(f"reasoning parser:  {reasoning_file or '(none for this vLLM version)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
