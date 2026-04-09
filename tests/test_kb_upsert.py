from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

RAG_DIR = Path(__file__).resolve().parents[1] / "src" / "rag"
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


def _install_fake_langchain_modules() -> None:
    lc_community = types.ModuleType("langchain_community")
    lc_embeddings = types.ModuleType("langchain_community.embeddings")
    lc_vectorstores = types.ModuleType("langchain_community.vectorstores")
    lc_core = types.ModuleType("langchain_core")
    lc_core_documents = types.ModuleType("langchain_core.documents")
    lc_splitters = types.ModuleType("langchain_text_splitters")

    class _DummyHFEmbeddings:
        def __init__(self, *args, **kwargs) -> None:
            return

    class _DummyCollection:
        def count(self) -> int:
            return 0

    class _DummyChroma:
        def __init__(self, *args, **kwargs) -> None:
            self._collection = _DummyCollection()

        def delete(self, ids=None) -> None:
            return

        def add_documents(self, documents=None, ids=None) -> None:
            return

    class _DummyDocument:
        def __init__(self, page_content: str = "", metadata=None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    class _DummySplitter:
        def __init__(self, *args, **kwargs) -> None:
            return

        def create_documents(self, texts):
            return [_DummyDocument(page_content=str(t)) for t in (texts or [])]

    lc_embeddings.HuggingFaceEmbeddings = _DummyHFEmbeddings
    lc_vectorstores.Chroma = _DummyChroma
    lc_core_documents.Document = _DummyDocument
    lc_splitters.RecursiveCharacterTextSplitter = _DummySplitter

    sys.modules["langchain_community"] = lc_community
    sys.modules["langchain_community.embeddings"] = lc_embeddings
    sys.modules["langchain_community.vectorstores"] = lc_vectorstores
    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.documents"] = lc_core_documents
    sys.modules["langchain_text_splitters"] = lc_splitters


_install_fake_langchain_modules()

import kb_upsert  # noqa: E402


class _FakeCollection:
    def __init__(self, n: int = 1) -> None:
        self._n = n

    def count(self) -> int:
        return self._n


class _FakeVectorDB:
    def __init__(self) -> None:
        self._collection = _FakeCollection(3)
        self.deleted_ids = []
        self.added = 0

    def delete(self, ids):
        self.deleted_ids.extend(ids or [])

    def add_documents(self, documents, ids):
        self.added += len(ids or [])


class _DummyDoc:
    def __init__(self) -> None:
        self.metadata = {}


class TestKbUpsert(unittest.TestCase):
    def test_ingest_assets_triggers_runtime_cache_invalidation(self) -> None:
        fake_db = _FakeVectorDB()
        assets = [
            {
                "source_type": "requirement",
                "origin": "file_upload",
                "source_name": "a.md",
                "suffix": ".md",
                "content_bytes": b"hello",
                "text": "",
                "external_ref": "",
                "metadata": {},
            }
        ]

        with patch.object(kb_upsert, "_load_index", return_value={"version": "1", "items": {}}), patch.object(
            kb_upsert, "_save_index", return_value=None
        ), patch.object(
            kb_upsert, "_init_vectordb", return_value=fake_db
        ), patch.object(
            kb_upsert, "_asset_to_text", return_value=("text body", [])
        ), patch.object(
            kb_upsert, "_build_chunks", return_value=([_DummyDoc()], ["cid1"])
        ), patch.object(
            kb_upsert, "_persist_raw_asset", return_value=""
        ), patch.object(
            kb_upsert, "_invalidate_retrieval_runtime_cache"
        ) as mocked_invalidate:
            result = kb_upsert.ingest_assets(assets=assets, mode="append")

        self.assertTrue(result.get("ok"))
        self.assertEqual(int(result.get("ingested_assets", 0)), 1)
        mocked_invalidate.assert_called_once()

    def test_delete_assets_triggers_runtime_cache_invalidation(self) -> None:
        fake_db = _FakeVectorDB()
        index_data = {
            "version": "1",
            "items": {
                "requirement|file_upload|a.md": {
                    "chunk_ids": ["cid1", "cid2"],
                    "raw_asset_path": "",
                }
            },
        }
        with patch.object(kb_upsert, "_load_index", return_value=index_data), patch.object(
            kb_upsert, "_init_vectordb", return_value=fake_db
        ), patch.object(
            kb_upsert, "_save_index", return_value=None
        ), patch.object(
            kb_upsert, "_invalidate_retrieval_runtime_cache"
        ) as mocked_invalidate:
            result = kb_upsert.delete_assets(
                doc_keys=["requirement|file_upload|a.md"],
                delete_raw_asset=False,
            )

        self.assertTrue(result.get("ok"))
        self.assertEqual(int(result.get("deleted_assets", 0)), 1)
        mocked_invalidate.assert_called_once()


class TestKbUpsertNormalization(unittest.TestCase):
    def test_normalize_status_priority_ingest_over_review(self) -> None:
        metadata = {"status": "approved", "review_status": "rejected", "ingest_status": "pending"}
        normalized = kb_upsert._normalize_asset_metadata(metadata)
        self.assertEqual(normalized.get("ingest_status"), "pending")
        self.assertEqual(normalized.get("review_status"), "rejected")
        self.assertEqual(normalized.get("status"), "pending")

    def test_normalize_status_synonym_chinese(self) -> None:
        metadata = {"status": "已审核"}
        normalized = kb_upsert._normalize_asset_metadata(metadata)
        self.assertEqual(normalized.get("status"), "approved")

    def test_normalize_modules_trim_and_dedup(self) -> None:
        metadata = {"modules": " 核心模块 , API ", "module": ["API", "支付 "]}
        normalized = kb_upsert._normalize_asset_metadata(metadata)
        self.assertEqual(normalized.get("modules"), ["核心模块", "API", "支付"])
        self.assertEqual(normalized.get("module"), "核心模块,API,支付")

    def test_normalize_release_trim(self) -> None:
        metadata = {"release": " v1.0 "}
        normalized = kb_upsert._normalize_asset_metadata(metadata)
        self.assertEqual(normalized.get("release"), "v1.0")

    def test_normalize_approved_from_status(self) -> None:
        metadata = {"status": "已审核"}
        normalized = kb_upsert._normalize_asset_metadata(metadata)
        self.assertTrue(normalized.get("approved"))

    def test_normalize_approved_respects_explicit_value(self) -> None:
        metadata = {"status": "approved", "approved": "false"}
        normalized = kb_upsert._normalize_asset_metadata(metadata)
        self.assertFalse(normalized.get("approved"))

    def test_normalize_trace_refs_split(self) -> None:
        metadata = {"trace_refs": "A,B;C"}
        normalized = kb_upsert._normalize_asset_metadata(metadata)
        self.assertEqual(normalized.get("trace_refs"), ["A", "B", "C"])

    def test_build_chunks_ext_meta_list_joined(self) -> None:
        base_metadata = {
            "doc_key": "req|file|a.md",
            "source_hash": "hash123",
            "extra_meta": {"modules": ["A ", "B"], "release": " v1 "},
        }
        docs, _ = kb_upsert._build_chunks(
            text="hello",
            base_metadata=base_metadata,
            chunk_size=200,
            chunk_overlap=0,
        )
        self.assertTrue(docs)
        metadata = docs[0].metadata
        self.assertEqual(metadata.get("modules"), ["A", "B"])
        self.assertEqual(metadata.get("release"), "v1")
        self.assertEqual(metadata.get("ext_modules"), "A,B")
        self.assertEqual(metadata.get("ext_release"), "v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
