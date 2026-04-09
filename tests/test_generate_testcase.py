from __future__ import annotations

import os
import sys
import types
import unittest

RAG_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "rag")
RAG_DIR = os.path.abspath(RAG_DIR)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)


def _install_fake_langchain_modules() -> None:
    lc_community = types.ModuleType("langchain_community")
    lc_embeddings = types.ModuleType("langchain_community.embeddings")
    lc_vectorstores = types.ModuleType("langchain_community.vectorstores")
    lc_openai = types.ModuleType("langchain_openai")
    lc_core = types.ModuleType("langchain_core")
    lc_core_prompts = types.ModuleType("langchain_core.prompts")

    class _DummyHFEmbeddings:
        def __init__(self, *args, **kwargs) -> None:
            return

    class _DummyChroma:
        def __init__(self, *args, **kwargs) -> None:
            return

    class _DummyChatOpenAI:
        def __init__(self, *args, **kwargs) -> None:
            return

        def invoke(self, prompt: str):
            class _Resp:
                content = ""

            return _Resp()

    class _DummyPromptTemplate:
        def __init__(self, *args, **kwargs) -> None:
            self.template = kwargs.get("template", "")

        def format(self, **kwargs) -> str:
            return self.template

    lc_embeddings.HuggingFaceEmbeddings = _DummyHFEmbeddings
    lc_vectorstores.Chroma = _DummyChroma
    lc_openai.ChatOpenAI = _DummyChatOpenAI
    lc_core_prompts.PromptTemplate = _DummyPromptTemplate

    sys.modules["langchain_community"] = lc_community
    sys.modules["langchain_community.embeddings"] = lc_embeddings
    sys.modules["langchain_community.vectorstores"] = lc_vectorstores
    sys.modules["langchain_openai"] = lc_openai
    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.prompts"] = lc_core_prompts


os.environ.setdefault("OPENAI_API_KEY", "test-key")
_install_fake_langchain_modules()
if "httpx" not in sys.modules:
    httpx_stub = types.ModuleType("httpx")

    class _DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            return

    httpx_stub.Client = _DummyClient
    sys.modules["httpx"] = httpx_stub

import generate_testcase  # noqa: E402


class TestGenerateTestcaseFilters(unittest.TestCase):
    def test_is_doc_approved_unknown_false_even_legacy(self) -> None:
        metadata = {}
        self.assertFalse(generate_testcase._is_doc_approved(metadata, include_legacy_unlabeled=True))
        metadata = {"ext_status": "unknown"}
        self.assertFalse(generate_testcase._is_doc_approved(metadata, include_legacy_unlabeled=True))

    def test_is_doc_approved_accepts_chinese_approved(self) -> None:
        metadata = {"ext_status": "已审核"}
        self.assertTrue(generate_testcase._is_doc_approved(metadata, include_legacy_unlabeled=False))

    def test_module_filters_trim_case_insensitive(self) -> None:
        metadata = {"ext_module": " Core "}
        policy = {"approved_only": False, "modules": [" core "]}
        self.assertTrue(generate_testcase._passes_runtime_filters(metadata, policy))

    def test_filters_prefer_canonical_metadata_fields(self) -> None:
        metadata = {"approved": True, "release": "v2", "modules": ["订单", "支付"]}
        policy = {"approved_only": True, "release": "v2", "modules": ["支付"]}
        self.assertTrue(generate_testcase._passes_runtime_filters(metadata, policy))

    def test_release_filter_falls_back_to_legacy_field(self) -> None:
        metadata = {"approved": True, "ext_release": "v3"}
        policy = {"approved_only": True, "release": "v3", "modules": []}
        self.assertTrue(generate_testcase._passes_runtime_filters(metadata, policy))

    def test_is_doc_approved_from_canonical_status(self) -> None:
        metadata = {"status": "approved"}
        self.assertTrue(generate_testcase._is_doc_approved(metadata, include_legacy_unlabeled=False))

    def test_context_header_contains_doc_key_and_chunk_index(self) -> None:
        class _Doc:
            def __init__(self) -> None:
                self.page_content = "示例内容"
                self.metadata = {
                    "doc_key": "requirement|file_upload|req.md",
                    "chunk_index": 3,
                    "source_type": "requirement",
                    "source_name": "req.md",
                    "origin": "file_upload",
                    "ext_module": "订单",
                }

        candidates = [
            {
                "source_type": "requirement",
                "rank_key": (0, 0.1, 0, 0),
                "doc": _Doc(),
                "unique_key": "u1",
            }
        ]
        context = generate_testcase._build_context_from_candidates(candidates)
        self.assertIn("doc_key=requirement|file_upload|req.md", context)
        self.assertIn("chunk_index=3", context)


if __name__ == "__main__":
    unittest.main(verbosity=2)
