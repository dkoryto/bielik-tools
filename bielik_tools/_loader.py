"""Load a parser implementation from a file shipped in ``tools/``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_file_module(module_name: str, path: Path) -> ModuleType:
    """Import ``path`` as ``module_name``, reusing an already-loaded module.

    The module is placed in ``sys.modules`` before execution so that vLLM's
    ``register_module`` decorator (which records ``cls.__module__`` for lazy
    lookup in recent versions) can resolve it by name later on.
    """
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load parser module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
