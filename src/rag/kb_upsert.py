from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

# Keep the same offline defaults as existing pipeline.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from connectors.feishu import fetch_feishu_board_text, fetch_feishu_doc_text
from connectors.figma import fetch_figma_text
from parsers.document_text import extract_text_from_document_bytes
from parsers.image_ocr import extract_text_from_image_bytes
from parsers.xmind import parse_xmind_bytes

SyncMode = Literal["append", "replace_by_source", "rebuild_all"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIRECTORY = DATA_DIR / "chroma_db"
RAW_ASSET_DIR = DATA_DIR / "kb_assets"
INDEX_FILE = DATA_DIR / "kb_index.json"
INDEX_LOCK_FILE = DATA_DIR / "kb_index.lock"

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def _index_file_lock():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_fp = INDEX_LOCK_FILE.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl  # type: ignore

            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        except Exception:
            # 非 POSIX 环境降级为无锁，仍保持原子写。
            pass
        yield
    finally:
        try:
            import fcntl  # type: ignore

            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_fp.close()


def _load_index() -> Dict[str, Any]:
    with _index_file_lock():
        if not INDEX_FILE.exists():
            return {"version": "1", "updated_at": _utc_now_iso(), "items": {}}

        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"version": "1", "updated_at": _utc_now_iso(), "items": {}}


def _save_index(index_data: Dict[str, Any]) -> None:
    payload = json.dumps(index_data, ensure_ascii=False, indent=2)
    with _index_file_lock():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=DATA_DIR,
            suffix=".kb_index.tmp",
        )
        try:
            tmp_file.write(payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        finally:
            tmp_file.close()
        Path(tmp_file.name).replace(INDEX_FILE)


def _resolve_local_embedding_model_path() -> str:
    custom_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    if custom_model_path:
        custom_path = Path(custom_model_path).expanduser().resolve()
        if (custom_path / "config.json").exists():
            return str(custom_path)
        raise FileNotFoundError(f"EMBEDDING_MODEL_PATH 无效: {custom_path}（缺少 config.json）")

    model_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
    )
    snapshots_dir = model_root / "snapshots"
    refs_main = model_root / "refs" / "main"

    if refs_main.exists():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip()
        if snapshot_id:
            candidate = snapshots_dir / snapshot_id
            if (candidate / "config.json").exists():
                return str(candidate.resolve())

    if snapshots_dir.exists():
        for candidate in sorted(snapshots_dir.iterdir(), reverse=True):
            if candidate.is_dir() and (candidate / "config.json").exists():
                return str(candidate.resolve())

    raise FileNotFoundError(
        "未找到本地 Embedding 模型 all-MiniLM-L6-v2。"
        " 请先下载到 HuggingFace 缓存，或设置 EMBEDDING_MODEL_PATH。"
    )


def _normalize_status_value(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    mapping = {
        "approved": "approved",
        "published": "approved",
        "pass": "approved",
        "passed": "approved",
        "已审核": "approved",
        "已入库": "approved",
        "通过": "approved",
        "draft": "draft",
        "草稿": "draft",
        "pending": "pending",
        "待审核": "pending",
        "待处理": "pending",
        "rejected": "rejected",
        "驳回": "rejected",
        "拒绝": "rejected",
        "不通过": "rejected",
    }
    if lowered in mapping:
        return mapping[lowered]
    if text in mapping:
        return mapping[text]
    return "unknown"


def _split_multi_values(raw: Any) -> List[str]:
    if raw is None:
        return []
    values: List[str] = []
    items: List[str] = []
    if isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        items = [str(raw)]

    for item in items:
        for token in re.split(r"[,\n，、;/；|]+", item):
            value = token.strip().strip("[]【】()（）")
            if value and value not in values:
                values.append(value)
    return values


def _normalize_feature_key(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    lowered = re.sub(r"\s+", "_", text.lower())
    lowered = re.sub(r"[^a-z0-9_]+", "_", lowered).strip("_")
    return lowered


def _normalize_bool_value(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "on", "是", "已审核", "通过"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", "未审核", "不通过"}:
        return False
    return None


def _normalize_asset_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    raw = metadata if isinstance(metadata, dict) else {}
    normalized = dict(raw)

    ingest_status = _normalize_status_value(raw.get("ingest_status"))
    review_status = _normalize_status_value(raw.get("review_status"))
    status = _normalize_status_value(raw.get("status"))
    if ingest_status:
        normalized["ingest_status"] = ingest_status
    if review_status:
        normalized["review_status"] = review_status
    if ingest_status or review_status or status:
        normalized["status"] = ingest_status or review_status or status

    modules = []
    for key in (
        "modules",
        "module",
        "module_name",
        "business_module",
        "ext_modules",
        "ext_module",
        "ext_module_name",
        "ext_business_module",
        "ext_domain",
    ):
        modules.extend(_split_multi_values(raw.get(key)))
    deduped_modules: List[str] = []
    for item in modules:
        if item not in deduped_modules:
            deduped_modules.append(item)
    if deduped_modules:
        normalized["modules"] = deduped_modules
        normalized["module"] = ",".join(deduped_modules)

    release = str(raw.get("release") or raw.get("ext_release") or "").strip()
    if release:
        normalized["release"] = release

    approved = _normalize_bool_value(raw.get("approved"))
    if approved is None:
        status_candidate = _normalize_status_value(
            normalized.get("status")
            or raw.get("ext_status")
            or raw.get("ext_ingest_status")
            or raw.get("ext_review_status")
        )
        if status_candidate:
            approved = status_candidate == "approved"
    if approved is not None:
        normalized["approved"] = bool(approved)

    trace_refs = _split_multi_values(raw.get("trace_refs"))
    if trace_refs:
        normalized["trace_refs"] = trace_refs[:20]

    feature_key = _normalize_feature_key(raw.get("feature_key"))
    if feature_key:
        normalized["feature_key"] = feature_key

    business_domain = _split_multi_values(raw.get("business_domain"))
    if business_domain:
        normalized["business_domain"] = business_domain

    related_doc_keys = _split_multi_values(raw.get("related_doc_keys"))
    if related_doc_keys:
        normalized["related_doc_keys"] = related_doc_keys[:20]

    upstream_modules = _split_multi_values(raw.get("upstream_modules"))
    if upstream_modules:
        normalized["upstream_modules"] = upstream_modules

    downstream_modules = _split_multi_values(raw.get("downstream_modules"))
    if downstream_modules:
        normalized["downstream_modules"] = downstream_modules

    return normalized


def _decode_text_bytes(payload: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return payload.decode(encoding)
        except Exception:
            continue
    return payload.decode("utf-8", errors="ignore")


def _csv_bytes_to_text(payload: bytes) -> str:
    raw_text = _decode_text_bytes(payload)
    sio = io.StringIO(raw_text)
    reader = csv.DictReader(sio)

    if not reader.fieldnames:
        return raw_text

    lines: List[str] = []
    for idx, row in enumerate(reader, start=1):
        kv = [f"{k}: {str(v).strip()}" for k, v in row.items() if str(v).strip()]
        if kv:
            lines.append(f"Row {idx}: " + " | ".join(kv))
    return "\n".join(lines)


def _external_ref_to_text(origin: str, reference: str) -> str:
    if origin == "feishu_doc":
        return fetch_feishu_doc_text(reference)
    if origin == "feishu_board":
        return fetch_feishu_board_text(reference)
    if origin == "figma":
        return fetch_figma_text(reference)
    if origin in {"api_doc_link", "api_doc_url"}:
        # First stage fallback: keep the link as retrievable text until real fetch parser is integrated.
        return f"API 接口文档链接: {reference}"
    raise ValueError(f"未知外部来源 origin: {origin}")


def _asset_to_text(asset: Dict[str, Any]) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    source_name = asset.get("source_name", "unknown")

    text = str(asset.get("text") or "").strip()
    suffix = str(asset.get("suffix") or "").lower().strip()
    payload = asset.get("content_bytes")
    external_ref = str(asset.get("external_ref") or "").strip()
    origin = str(asset.get("origin") or "file_upload").strip()

    if external_ref:
        ext_text = _external_ref_to_text(origin, external_ref).strip()
        text = f"{text}\n\n{ext_text}".strip() if text else ext_text

    if isinstance(payload, bytes) and payload:
        if suffix in {".md", ".markdown", ".txt", ".log", ".json", ".yaml", ".yml"}:
            file_text = _decode_text_bytes(payload)
            text = f"{text}\n\n{file_text}".strip() if text else file_text
        elif suffix in {".pdf", ".doc", ".docx"}:
            doc_text, warning = extract_text_from_document_bytes(payload, suffix)
            if warning:
                warnings.append(f"{source_name}: {warning}")
            if doc_text.strip():
                text = f"{text}\n\n{doc_text}".strip() if text else doc_text
        elif suffix == ".csv":
            file_text = _csv_bytes_to_text(payload)
            text = f"{text}\n\n{file_text}".strip() if text else file_text
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            ocr_text, warning = extract_text_from_image_bytes(payload)
            if warning:
                warnings.append(f"{source_name}: {warning}")
            merged = ocr_text.strip()
            if merged:
                text = f"{text}\n\n{merged}".strip() if text else merged
        elif suffix == ".xmind":
            map_text = parse_xmind_bytes(payload)
            text = f"{text}\n\n{map_text}".strip() if text else map_text
        else:
            fallback = _decode_text_bytes(payload).strip()
            if fallback:
                text = f"{text}\n\n{fallback}".strip() if text else fallback

    if not text.strip():
        warnings.append(f"{source_name}: 未提取到有效文本，已降级写入文件名占位内容。")
        text = f"[{asset.get('source_type', 'unknown')}] {source_name}"

    return text.strip(), warnings


def _build_chunks(
    text: str,
    base_metadata: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Document], List[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n- ", "\n", "。", "；", "，", " ", ""],
    )
    docs = splitter.create_documents([text])

    chunk_ids: List[str] = []
    total = len(docs)
    for idx, doc in enumerate(docs):
        meta = dict(base_metadata)
        extra_meta = meta.pop("extra_meta", {}) or {}
        meta["chunk_index"] = idx
        meta["chunk_total"] = total

        if isinstance(extra_meta, dict):
            for k, v in extra_meta.items():
                key = str(k or "").strip()
                if not key:
                    continue
                canonical_key = key[4:] if key.startswith("ext_") else key
                if v is None:
                    continue
                value_text = ""
                if isinstance(v, list):
                    parts = [str(x).strip() for x in v if str(x).strip()]
                    if not parts:
                        continue
                    meta[canonical_key] = parts
                    value_text = ",".join(parts)
                elif isinstance(v, bool):
                    meta[canonical_key] = v
                    value_text = "true" if v else "false"
                elif isinstance(v, (int, float)):
                    meta[canonical_key] = v
                    value_text = str(v)
                else:
                    value_text = str(v).strip()
                    if not value_text:
                        continue
                    meta[canonical_key] = value_text
                if not value_text:
                    continue
                meta[f"ext_{canonical_key}"] = value_text[:1000]

        doc.metadata = meta
        raw_id = f"{base_metadata['doc_key']}|{base_metadata['source_hash']}|{idx}"
        chunk_ids.append(hashlib.sha256(raw_id.encode("utf-8")).hexdigest())

    return docs, chunk_ids


def _persist_raw_asset(asset: Dict[str, Any], source_hash: str) -> str:
    payload = asset.get("content_bytes")
    if not isinstance(payload, bytes) or not payload:
        return ""

    source_type = str(asset.get("source_type") or "unknown")
    origin = str(asset.get("origin") or "unknown")
    source_name = str(asset.get("source_name") or "unnamed")
    suffix = str(asset.get("suffix") or "").strip().lower()

    target_dir = RAW_ASSET_DIR / _safe_filename(source_type) / _safe_filename(origin)
    target_dir.mkdir(parents=True, exist_ok=True)

    ext = suffix if suffix.startswith(".") else (f".{suffix}" if suffix else "")
    filename = f"{source_hash[:12]}_{_safe_filename(source_name)}{ext}"
    target_file = target_dir / filename
    target_file.write_bytes(payload)
    return str(target_file)


def _init_vectordb(need_embedding: bool = True) -> Chroma:
    embeddings = None
    if need_embedding:
        model_path = _resolve_local_embedding_model_path()
        embeddings = HuggingFaceEmbeddings(model_name=model_path)
    return Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=embeddings)


def _delete_chunks(vectordb: Chroma, chunk_ids: List[str]) -> int:
    if not chunk_ids:
        return 0
    vectordb.delete(ids=chunk_ids)
    return len(chunk_ids)


def _delete_raw_asset_file(raw_asset_path: str) -> Tuple[bool, str]:
    raw = str(raw_asset_path or "").strip()
    if not raw:
        return False, ""
    path = Path(raw)

    if not path.exists():
        return False, f"原始文件不存在: {path}"

    if path.is_file():
        path.unlink(missing_ok=True)
        return True, ""

    return False, f"原始路径不是文件，已跳过: {path}"


def _invalidate_retrieval_runtime_cache() -> None:
    try:
        rag_runtime = importlib.import_module("generate_testcase")
    except Exception:
        return
    clear_fn = getattr(rag_runtime, "clear_retrieval_runtime_cache", None)
    if callable(clear_fn):
        try:
            clear_fn()
        except Exception:
            return


def delete_assets(doc_keys: List[str], delete_raw_asset: bool = True) -> Dict[str, Any]:
    normalized_doc_keys: List[str] = []
    seen = set()
    for key in doc_keys or []:
        value = str(key or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized_doc_keys.append(value)

    result: Dict[str, Any] = {
        "ok": True,
        "partial_success": False,
        "received_doc_keys": len(normalized_doc_keys),
        "deleted_assets": 0,
        "deleted_chunks": 0,
        "deleted_raw_files": 0,
        "missing_doc_keys": [],
        "deleted_doc_keys": [],
        "warnings": [],
        "errors": [],
        "db_count": 0,
    }

    if not normalized_doc_keys:
        result["ok"] = False
        result["errors"].append("未提供有效 doc_key。")
        return result

    index_data = _load_index()
    items = index_data.get("items") or {}

    try:
        vectordb = _init_vectordb(need_embedding=False)
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"向量库初始化失败: {exc}")
        return result

    for doc_key in normalized_doc_keys:
        item = items.get(doc_key)
        if not item:
            result["missing_doc_keys"].append(doc_key)
            continue

        chunk_ids = list(item.get("chunk_ids") or [])
        try:
            deleted_chunk_count = _delete_chunks(vectordb, chunk_ids)
        except Exception as exc:
            result["errors"].append(f"{doc_key}: 删除向量失败: {exc}")
            continue

        if delete_raw_asset:
            raw_asset_path = str(item.get("raw_asset_path") or "").strip()
            if raw_asset_path:
                try:
                    removed, warning = _delete_raw_asset_file(raw_asset_path)
                    if removed:
                        result["deleted_raw_files"] += 1
                    if warning:
                        result["warnings"].append(f"{doc_key}: {warning}")
                except Exception as exc:
                    result["warnings"].append(f"{doc_key}: 删除原始文件异常: {exc}")

        items.pop(doc_key, None)
        result["deleted_assets"] += 1
        result["deleted_chunks"] += deleted_chunk_count
        result["deleted_doc_keys"].append(doc_key)

    if result["deleted_assets"] > 0:
        index_data["items"] = items
        index_data["updated_at"] = _utc_now_iso()
        _save_index(index_data)
        _invalidate_retrieval_runtime_cache()

    try:
        result["db_count"] = int(vectordb._collection.count())
    except Exception:
        result["db_count"] = 0

    if result["errors"]:
        if result["deleted_assets"] > 0:
            result["ok"] = True
            result["partial_success"] = True
        else:
            result["ok"] = False

    return result


def ingest_assets(
    assets: List[Dict[str, Any]],
    mode: SyncMode = "append",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> Dict[str, Any]:
    if mode not in {"append", "replace_by_source", "rebuild_all"}:
        raise ValueError(f"不支持的 mode: {mode}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "rebuild_all" and PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    index_data = _load_index()
    if mode == "rebuild_all":
        index_data["items"] = {}

    vectordb = _init_vectordb()

    result: Dict[str, Any] = {
        "ok": True,
        "partial_success": False,
        "mode": mode,
        "received_assets": len(assets),
        "ingested_assets": 0,
        "skipped_assets": 0,
        "added_chunks": 0,
        "deleted_chunks": 0,
        "warnings": [],
        "errors": [],
        "updated_sources": [],
        "db_count": 0,
    }

    new_records: List[Dict[str, Any]] = []
    post_commit_replacements: List[Dict[str, Any]] = []

    for idx, asset in enumerate(assets, start=1):
        chunk_ids: List[str] = []
        raw_asset_path = ""
        try:
            source_type = str(asset.get("source_type") or "").strip().lower()
            origin = str(asset.get("origin") or "").strip().lower()
            source_name = str(asset.get("source_name") or "").strip()

            if not source_type or not origin or not source_name:
                raise ValueError("asset 缺少必填字段: source_type/origin/source_name")

            text, warnings = _asset_to_text(asset)
            result["warnings"].extend(warnings)

            hash_material = b""
            payload = asset.get("content_bytes")
            if isinstance(payload, bytes) and payload:
                hash_material += payload
            hash_material += text.encode("utf-8", errors="ignore")
            external_ref = str(asset.get("external_ref") or "").strip()
            if external_ref:
                hash_material += external_ref.encode("utf-8", errors="ignore")

            source_hash = _sha256_bytes(hash_material)
            doc_key = f"{source_type}|{origin}|{source_name}"
            old_item = (index_data.get("items") or {}).get(doc_key)
            old_chunk_ids = list(old_item.get("chunk_ids") or []) if old_item else []
            old_raw_asset_path = str(old_item.get("raw_asset_path") or "").strip() if old_item else ""

            if mode == "append" and old_item:
                if old_item.get("source_hash") == source_hash:
                    result["skipped_assets"] += 1
                    result["warnings"].append(f"{source_name}: 内容未变化，已跳过。")
                    continue
                result["skipped_assets"] += 1
                result["warnings"].append(
                    f"{source_name}: 已存在同名来源且内容变化，append 模式不覆盖。"
                    "请改用 replace_by_source。"
                )
                continue

            synced_at = _utc_now_iso()
            normalized_meta = _normalize_asset_metadata(asset.get("metadata") or {})
            base_meta: Dict[str, Any] = {
                "doc_key": doc_key,
                "source_type": source_type,
                "origin": origin,
                "source_name": source_name,
                "source_hash": source_hash,
                "synced_at": synced_at,
                "extra_meta": normalized_meta,
            }

            docs, chunk_ids = _build_chunks(
                text=text,
                base_metadata=base_meta,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            vectordb.add_documents(documents=docs, ids=chunk_ids)

            raw_asset_path = _persist_raw_asset(asset, source_hash)
            index_data.setdefault("items", {})[doc_key] = {
                "source_type": source_type,
                "origin": origin,
                "source_name": source_name,
                "source_hash": source_hash,
                "chunk_ids": chunk_ids,
                "chunk_count": len(chunk_ids),
                "raw_asset_path": raw_asset_path,
                "synced_at": synced_at,
                "metadata": normalized_meta,
            }

            new_records.append(
                {
                    "doc_key": doc_key,
                    "chunk_ids": list(chunk_ids),
                    "raw_asset_path": raw_asset_path,
                }
            )
            if mode == "replace_by_source" and old_item:
                post_commit_replacements.append(
                    {
                        "source_name": source_name,
                        "old_chunk_ids": old_chunk_ids,
                        "old_raw_asset_path": old_raw_asset_path,
                    }
                )

            result["ingested_assets"] += 1
            result["added_chunks"] += len(chunk_ids)
            result["updated_sources"].append(doc_key)

        except Exception as exc:
            if chunk_ids:
                try:
                    _delete_chunks(vectordb, chunk_ids)
                except Exception as rollback_exc:
                    result["warnings"].append(f"Asset#{idx}: 回滚新增向量失败: {rollback_exc}")
            if raw_asset_path:
                try:
                    _delete_raw_asset_file(raw_asset_path)
                except Exception as rollback_exc:
                    result["warnings"].append(f"Asset#{idx}: 回滚原始文件失败: {rollback_exc}")
            result["errors"].append(f"Asset#{idx}: {exc}")

    index_data["updated_at"] = _utc_now_iso()
    try:
        _save_index(index_data)
    except Exception as exc:
        result["errors"].append(f"索引保存失败，已回滚本次新增向量: {exc}")
        rollback_chunk_count = 0
        rollback_file_count = 0
        for record in reversed(new_records):
            record_chunk_ids = list(record.get("chunk_ids") or [])
            if record_chunk_ids:
                try:
                    rollback_chunk_count += _delete_chunks(vectordb, record_chunk_ids)
                except Exception as rollback_exc:
                    result["warnings"].append(f"回滚向量失败({record.get('doc_key')}): {rollback_exc}")
            record_raw_path = str(record.get("raw_asset_path") or "").strip()
            if record_raw_path:
                try:
                    removed, warning = _delete_raw_asset_file(record_raw_path)
                    if removed:
                        rollback_file_count += 1
                    if warning:
                        result["warnings"].append(f"{record.get('doc_key')}: {warning}")
                except Exception as rollback_exc:
                    result["warnings"].append(f"回滚原始文件失败({record.get('doc_key')}): {rollback_exc}")
        result["warnings"].append(
            f"本次回滚: 删除新增向量 {rollback_chunk_count} 条，删除临时原始文件 {rollback_file_count} 个。"
        )
        result["ok"] = False
        result["partial_success"] = False
        result["ingested_assets"] = 0
        result["added_chunks"] = 0
        result["updated_sources"] = []
        try:
            result["db_count"] = int(vectordb._collection.count())
        except Exception:
            result["db_count"] = 0
        return result

    for replacement in post_commit_replacements:
        old_chunk_ids = list(replacement.get("old_chunk_ids") or [])
        if old_chunk_ids:
            try:
                result["deleted_chunks"] += _delete_chunks(vectordb, old_chunk_ids)
            except Exception as exc:
                result["warnings"].append(
                    f"{replacement.get('source_name')}: 替换后清理旧向量失败: {exc}"
                )

        old_raw_asset_path = str(replacement.get("old_raw_asset_path") or "").strip()
        if old_raw_asset_path:
            try:
                removed, warning = _delete_raw_asset_file(old_raw_asset_path)
                if warning:
                    result["warnings"].append(f"{replacement.get('source_name')}: {warning}")
                if not removed and not warning:
                    result["warnings"].append(
                        f"{replacement.get('source_name')}: 旧原始文件未删除（路径不可用）。"
                    )
            except Exception as exc:
                result["warnings"].append(
                    f"{replacement.get('source_name')}: 替换后清理旧原始文件异常: {exc}"
                )

    if (
        mode == "rebuild_all"
        or result["ingested_assets"] > 0
        or result["deleted_chunks"] > 0
        or bool(post_commit_replacements)
    ):
        _invalidate_retrieval_runtime_cache()

    try:
        result["db_count"] = int(vectordb._collection.count())
    except Exception:
        result["db_count"] = 0

    if result["errors"]:
        if result["ingested_assets"] > 0:
            result["partial_success"] = True
            result["ok"] = True
        else:
            result["ok"] = False

    return result


def build_asset(
    *,
    source_type: str,
    origin: str,
    source_name: str,
    suffix: str = "",
    content_bytes: bytes | None = None,
    text: str = "",
    external_ref: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "source_type": source_type,
        "origin": origin,
        "source_name": source_name,
        "suffix": suffix,
        "content_bytes": content_bytes or b"",
        "text": text,
        "external_ref": external_ref,
        "metadata": metadata or {},
    }


if __name__ == "__main__":
    # Minimal smoke run: ingest the default ticketing rules text file.
    sample_file = PROJECT_ROOT / "data" / "input" / "ticketing_rules.txt"
    if not sample_file.exists():
        print(f"[ERROR] sample file not found: {sample_file}")
        raise SystemExit(1)

    asset = build_asset(
        source_type="requirement",
        origin="file_upload",
        source_name=sample_file.name,
        suffix=sample_file.suffix,
        content_bytes=sample_file.read_bytes(),
        metadata={"bootstrapped": True},
    )

    summary = ingest_assets([asset], mode="append")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
