from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_streamlit_stub() -> None:
    if "streamlit" in sys.modules:
        return

    st_mod = types.ModuleType("streamlit")

    def _cache_resource(*args, **kwargs):  # type: ignore[no-untyped-def]
        def _decorator(fn):  # type: ignore[no-untyped-def]
            return fn

        return _decorator

    st_mod.cache_resource = _cache_resource  # type: ignore[attr-defined]
    sys.modules["streamlit"] = st_mod


def _install_langchain_prompt_stub() -> None:
    required = ["langchain_core", "langchain_core.prompts"]
    if all(name in sys.modules for name in required):
        return

    core_mod = types.ModuleType("langchain_core")
    prompts_mod = types.ModuleType("langchain_core.prompts")

    class _DummyPromptTemplate:
        def __init__(self, *args, **kwargs) -> None:
            self.template = kwargs.get("template", "")

        def format(self, **kwargs) -> str:
            return str(self.template)

    prompts_mod.PromptTemplate = _DummyPromptTemplate  # type: ignore[attr-defined]
    sys.modules["langchain_core"] = core_mod
    sys.modules["langchain_core.prompts"] = prompts_mod


_install_streamlit_stub()
_install_langchain_prompt_stub()

from apps import testcase_platform as tp  # noqa: E402
from src.rag.analysis.badcase_loop import save_badcase_rule_templates  # noqa: E402


class TestTaskQueryCompose(unittest.TestCase):
    def test_default_limit_no_longer_hardcoded_12000(self) -> None:
        long_text = "A" * 13000
        with patch.dict(os.environ, {}, clear=False):
            merged, warnings, _ = tp._compose_task_query(
                core_text=long_text,
                extra_text="",
                links_text="",
                files=[],
                max_chars=None,
            )

        self.assertIn(long_text, merged)
        self.assertFalse(any("已截断" in w for w in warnings))

    def test_compose_uses_env_limit_when_max_chars_not_provided(self) -> None:
        long_text = "B" * 17000
        with patch.dict(os.environ, {"TASK_QUERY_MAX_CHARS": "15000"}, clear=False):
            merged, warnings, _ = tp._compose_task_query(
                core_text=long_text,
                extra_text="",
                links_text="",
                files=[],
                max_chars=None,
            )

        self.assertIn("[提示] 输入过长，已自动截断。", merged)
        self.assertTrue(any("15000" in w for w in warnings))

    def test_compose_respects_explicit_limit_override(self) -> None:
        long_text = "C" * 23000
        with patch.dict(os.environ, {"TASK_QUERY_MAX_CHARS": "60000"}, clear=False):
            merged, warnings, _ = tp._compose_task_query(
                core_text=long_text,
                extra_text="",
                links_text="",
                files=[],
                max_chars=20000,
            )

        self.assertIn("[提示] 输入过长，已自动截断。", merged)
        self.assertTrue(any("20000" in w for w in warnings))


class TestTaskQueryLimitNormalize(unittest.TestCase):
    def test_normalize_task_query_max_chars_bounds(self) -> None:
        self.assertEqual(tp._normalize_task_query_max_chars("1000"), 12000)
        self.assertEqual(tp._normalize_task_query_max_chars("300000"), 200000)
        self.assertEqual(tp._normalize_task_query_max_chars("not-int"), 60000)


class TestRiskGate(unittest.TestCase):
    def test_extract_blocking_p0_risks_only_returns_p0(self) -> None:
        risk_report = {
            "items": [
                {"id": "RISK-001", "severity": "P0", "title": "命中排除项", "category": "constraint_violation"},
                {"id": "RISK-002", "severity": "P1", "title": "跨端链路证据偏弱", "category": "traceability"},
                {"id": "RISK-003", "severity": "p0", "title": "检索上下文为空", "category": "retrieval_quality"},
            ]
        }
        blocked = tp._extract_blocking_p0_risks(risk_report)
        self.assertEqual(len(blocked), 2)
        self.assertEqual({item.get("id") for item in blocked}, {"RISK-001", "RISK-003"})

    def test_extract_blocking_risks_should_follow_template_severities(self) -> None:
        risk_report = {
            "items": [
                {"id": "RISK-010", "severity": "P1", "title": "跨端链路证据偏弱", "category": "traceability"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            save_badcase_rule_templates(
                {
                    "templates": {
                        "p0_blocking": {"enabled": True, "severities": ["P1"]},
                    }
                },
                project_root=root,
            )
            with patch.object(tp, "PROJECT_ROOT", root):
                blocked = tp._extract_blocking_p0_risks(risk_report)
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].get("id"), "RISK-010")


if __name__ == "__main__":
    unittest.main(verbosity=2)
