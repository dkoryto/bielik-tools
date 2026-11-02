"""Load each parser file the way vLLM does (``--*-parser-plugin <path>``).

vLLM imports a plugin with ``import_from_path(<file stem>, <path>)`` and then
looks the parser up by name, so every file must register ``bielik`` exactly
once, on its own, with the expected implementation.
"""
import sys

import pytest

from conftest import TOOLS_DIR, TOOL_PARSER_FILES, FakeTokenizer

from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

TOOL_PARSER_CLASSES = {
    "default": "BielikToolParser",
    "builtin": "BielikV012ToolParser",
    "v3.1": "BielikV31ToolParser",
}

REASONING_PARSER_FILES = {
    "bielik_vllm_reasoning_parser.py": "BielikReasoningParser",
    "bielik_vllm_reasoning_parser_v0.13.0.py": "BielikReasoningParser",
    "bielik_vllm_reasoning_parser_v3.1.py": "BielikV31ReasoningParser",
}

SHARED_MODULES = (
    "_bielik_vllm_tool_parser_shared",
    "_bielik_vllm_reasoning_parser_shared",
)


@pytest.fixture
def isolated_registry(monkeypatch):
    """Fresh vLLM parser registries and module cache; counts registrations."""
    calls = []
    for manager, attr in ((ToolParserManager, "tool_parsers"),
                          (ReasoningParserManager, "reasoning_parsers")):
        monkeypatch.setattr(manager, attr, dict(getattr(manager, attr)))
        monkeypatch.setattr(manager, "lazy_parsers", dict(manager.lazy_parsers))
        for registry in (getattr(manager, attr), manager.lazy_parsers):
            registry.pop("bielik", None)

        original = manager.register_module.__func__

        def spy(cls, name=None, force=True, module=None, _original=original):
            calls.append((cls.__name__, name))
            return _original(cls, name, force, module)

        monkeypatch.setattr(manager, "register_module", classmethod(spy))

    for name in SHARED_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield calls
    for path in TOOLS_DIR.glob("bielik_vllm_*_parser*.py"):
        sys.modules.pop(path.stem, None)


@pytest.mark.parametrize("filename", list(TOOL_PARSER_FILES))
def test_tool_parser_plugin(filename, isolated_registry):
    ToolParserManager.import_tool_parser(str(TOOLS_DIR / filename))

    assert isolated_registry == [("ToolParserManager", "bielik")]
    parser_cls = ToolParserManager.get_tool_parser("bielik")
    assert parser_cls.__name__ == TOOL_PARSER_CLASSES[TOOL_PARSER_FILES[filename]]
    assert parser_cls.supports_required_and_named is False

    # vLLM >= 0.24 passes tools= (positionally or by keyword); older only
    # the tokenizer.
    for args, kwargs in (((), {}), ((None,), {}), ((), {"tools": None})):
        parser = parser_cls(FakeTokenizer(), *args, **kwargs)
        assert parser.tool_call_start_token_id == 1


@pytest.mark.parametrize("filename", list(REASONING_PARSER_FILES))
def test_reasoning_parser_plugin(filename, isolated_registry):
    ReasoningParserManager.import_reasoning_parser(str(TOOLS_DIR / filename))

    assert isolated_registry == [("ReasoningParserManager", "bielik")]
    parser_cls = ReasoningParserManager.get_reasoning_parser("bielik")
    assert parser_cls.__name__ == REASONING_PARSER_FILES[filename]


def test_shims_load_shared_implementation_once(isolated_registry):
    for filename in TOOL_PARSER_FILES:
        if filename != "bielik_vllm_tool_parser.py":
            ToolParserManager.import_tool_parser(str(TOOLS_DIR / filename))
    shared = sys.modules["_bielik_vllm_tool_parser_shared"]
    for filename in TOOL_PARSER_FILES:
        module = sys.modules.get(TOOLS_DIR.joinpath(filename).stem)
        if module is not None and filename != "bielik_vllm_tool_parser.py":
            assert issubclass(module.BielikToolParser, shared.BielikToolParser)


# --------------------------------------------------------------------------
# Reasoning parser behaviour
# --------------------------------------------------------------------------

THINK_VOCAB = {"<think>": 10, "</think>": 11}


@pytest.fixture(params=list(REASONING_PARSER_FILES))
def reasoning_variant(request, isolated_registry):
    ReasoningParserManager.import_reasoning_parser(str(TOOLS_DIR / request.param))
    return ReasoningParserManager.get_reasoning_parser("bielik")


@pytest.mark.parametrize("chat_kwargs, legacy, v31", [
    (None, False, False),
    ({}, False, False),
    ({"enable_thinking": True}, True, True),
    ({"enable_thinking": False}, False, False),
    ({"reasoning_effort": "high"}, False, True),
    ({"reasoning_effort": "none"}, False, False),
    ({"reasoning_effort": "None", "enable_thinking": True}, True, False),
])
def test_reasoning_delegate(reasoning_variant, chat_kwargs, legacy, v31):
    from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
    from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

    parser = reasoning_variant(FakeTokenizer(vocab=THINK_VOCAB),
                               chat_template_kwargs=chat_kwargs)
    thinking = v31 if reasoning_variant.__name__ == "BielikV31ReasoningParser" else legacy
    expected = DeepSeekR1ReasoningParser if thinking else IdentityReasoningParser
    assert type(parser._parser) is expected


# --------------------------------------------------------------------------
# Compatibility import helper
# --------------------------------------------------------------------------

@pytest.fixture(params=["bielik_vllm_tool_parser.py", "bielik_vllm_reasoning_parser.py"])
def compat_module(request, isolated_registry):
    from conftest import _load_module
    return _load_module(request.param)


def test_import_from_skips_missing_candidates(compat_module):
    import json
    (dumps,) = compat_module._import_from(
        ("bielik_nonexistent_pkg.protocol", "json"), "dumps")
    assert dumps is json.dumps


def test_import_from_skips_candidates_without_names(compat_module):
    import json
    (dumps,) = compat_module._import_from(("os", "json"), "dumps")
    assert dumps is json.dumps


def test_import_from_reraises_unrelated_failures(compat_module, tmp_path, monkeypatch):
    (tmp_path / "bielik_broken_mod.py").write_text("import bielik_missing_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError) as excinfo:
        compat_module._import_from(("bielik_broken_mod", "json"), "dumps")
    assert excinfo.value.name == "bielik_missing_dependency"


def test_import_from_reports_all_candidates(compat_module):
    with pytest.raises(compat_module._CompatImportError, match="bielik_a, bielik_b"):
        compat_module._import_from(("bielik_a", "bielik_b"), "x")
