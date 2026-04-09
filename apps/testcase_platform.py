from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Tuple
from uuid import uuid4

# ==========================================
# Offline/mirror defaults for local RAG runtime
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from src.rag.analysis.badcase_loop import load_badcase_rule_templates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = PROJECT_ROOT / "src" / "rag"
ENV_PATH = PROJECT_ROOT / ".env"
KB_INDEX_PATH = PROJECT_ROOT / "data" / "kb_index.json"
KB_OPERATION_LOG_PATH = PROJECT_ROOT / "data" / "kb_operation_log.jsonl"
REVIEW_QUEUE_PATH = PROJECT_ROOT / "data" / "review_queue.json"
REVIEW_QUEUE_SCHEMA_VERSION = 2

load_dotenv(dotenv_path=ENV_PATH)

PipelineComponents = Tuple[Callable[[str], str], Any, str]


def _bootstrap_import_path() -> None:
    if RAG_DIR.exists() and str(RAG_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_DIR))


def _safe_import(module_name: str, errors: list[str]) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        errors.append(f"模块 `{module_name}` 导入失败: {exc}")
        return None


def _resolve_pipeline_components() -> PipelineComponents:
    _bootstrap_import_path()
    errors: list[str] = []

    rag_module = _safe_import("generate_testcase", errors)

    get_augmented_context = None
    llm = None
    universal_template = None

    if rag_module is not None:
        get_augmented_context = getattr(rag_module, "get_augmented_context", None)
        llm = getattr(rag_module, "llm", None)
        universal_template = getattr(rag_module, "UNIVERSAL_TEMPLATE", None) or getattr(
            rag_module, "TEST_GEN_TEMPLATE", None
        )

    if not callable(get_augmented_context):
        errors.append("未找到可调用的 `get_augmented_context(task_query)`。")
    if llm is None:
        errors.append("未找到已初始化的 `llm` 实例。")
    if not universal_template:
        errors.append("未找到 `UNIVERSAL_TEMPLATE`（已尝试回退到 `TEST_GEN_TEMPLATE`）。")

    if errors:
        raise RuntimeError("\n".join(errors))

    return get_augmented_context, llm, universal_template


@st.cache_resource(show_spinner=False)
def load_pipeline_components() -> PipelineComponents:
    return _resolve_pipeline_components()


@st.cache_resource(show_spinner=False)
def load_kb_upsert_module() -> Any:
    _bootstrap_import_path()
    return importlib.import_module("kb_upsert")


def _normalize_chunk_content(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)

    if content is None:
        return ""

    return str(content)


def _stream_llm_tokens(llm: Any, prompt_text: str) -> Generator[str, None, None]:
    for chunk in llm.stream(prompt_text):
        text = _normalize_chunk_content(chunk)
        if text:
            yield text


def _build_prompt(template: str, context: str, task_query: str) -> str:
    prompt = PromptTemplate(template=template, input_variables=["context", "task"])
    return prompt.format(context=context, task=task_query)


def _generation_mode_instruction(mode_key: str) -> str:
    if mode_key == "field_validation":
        return (
            "当前模式: 字段校验用例（Field Validation）。\n"
            "本模式优先级高于模板内“忽略字段细节”的描述。\n"
            "必须重点覆盖必填、类型、长度、枚举、格式、边界值、默认值、兼容性、错误码与错误文案。\n"
            "输出仍保持结构化业务测试用例格式，严禁输出代码或伪代码。"
        )

    return (
        "当前模式: 业务接口用例（Business API）。\n"
        "聚焦业务状态流转、角色权限、接口联动和上下游影响。\n"
        "不展开字段类型/长度/格式的低价值校验。\n"
        "输出保持结构化业务测试用例格式，严禁输出代码或伪代码。"
    )


def _resolve_local_embedding_model_path() -> str:
    env_path = os.getenv("EMBEDDING_MODEL_PATH")
    env_error = None
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if (candidate / "config.json").exists():
            return str(candidate)
        env_error = f"EMBEDDING_MODEL_PATH 无效: {candidate}（缺少 config.json）"

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

    if env_error:
        raise FileNotFoundError(env_error)
    raise FileNotFoundError("未找到 all-MiniLM-L6-v2 的本地模型目录，请先下载到本机缓存。")


def _upsert_env_var(env_path: Path, key: str, value: str) -> bool:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    target_prefix = f"{key}="
    replaced = False
    changed = False
    new_lines: list[str] = []

    for line in lines:
        if line.startswith(target_prefix):
            replaced = True
            new_line = f"{key}={value}"
            if line != new_line:
                changed = True
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.append(f"{key}={value}")
        changed = True

    if changed:
        content = "\n".join(new_lines).rstrip() + "\n"
        env_path.write_text(content, encoding="utf-8")

    return changed


def _ensure_embedding_env_path() -> tuple[bool, str]:
    try:
        resolved_path = _resolve_local_embedding_model_path()
        os.environ["EMBEDDING_MODEL_PATH"] = resolved_path
        changed = _upsert_env_var(ENV_PATH, "EMBEDDING_MODEL_PATH", resolved_path)
        if changed:
            return True, f"已写入 .env: {resolved_path}"
        return True, f"已就绪: {resolved_path}"
    except Exception as exc:
        return False, str(exc)


def _check_chroma_status() -> tuple[bool, str]:
    chroma_dir = PROJECT_ROOT / "data" / "chroma_db"
    sqlite_file = chroma_dir / "chroma.sqlite3"

    if not chroma_dir.exists():
        return False, f"目录不存在: {chroma_dir}"
    if not sqlite_file.exists():
        return False, f"缺少文件: {sqlite_file.name}"
    if sqlite_file.stat().st_size <= 0:
        return False, "chroma.sqlite3 大小异常（0 字节）"

    try:
        conn = sqlite3.connect(f"file:{sqlite_file}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master LIMIT 1;")
        cur.fetchone()
        conn.close()
        size_kb = sqlite_file.stat().st_size / 1024
        return True, f"可读（{size_kb:.1f} KB）"
    except Exception as exc:
        return False, f"可读性检查失败: {exc}"


def _load_kb_index_data() -> Dict[str, Any]:
    if not KB_INDEX_PATH.exists():
        return {"version": "1", "updated_at": "", "items": {}}

    try:
        payload = json.loads(KB_INDEX_PATH.read_text(encoding="utf-8"))
        payload.setdefault("items", {})
        payload.setdefault("updated_at", "")
        payload.setdefault("version", "1")
        return payload
    except Exception:
        return {"version": "1", "updated_at": "", "items": {}}


def _format_time_display(raw_time: Any) -> str:
    text = str(raw_time or "").strip()
    if not text:
        return "-"

    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")

    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo:
                return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    return text


def _summarize_operation_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in (summary or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
            continue
        if isinstance(value, list):
            compact[f"{key}_count"] = len(value)
            continue
        if isinstance(value, dict):
            compact[f"{key}_count"] = len(value)
            continue
        compact[key] = str(value)
    return compact


def _append_kb_operation_log(
    *,
    operation: str,
    summary: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "ok": bool(summary.get("ok", False)),
        "partial_success": bool(summary.get("partial_success", False)),
        "summary": _summarize_operation_summary(summary),
        "extra": extra or {},
    }

    KB_OPERATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KB_OPERATION_LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
    bump_kb_data_version()


def bump_kb_data_version() -> int:
    try:
        current = int(st.session_state.get("kb_data_version", 0) or 0)
        next_value = current + 1
        st.session_state["kb_data_version"] = next_value
        return next_value
    except Exception:
        return 0


def _load_kb_operation_logs(limit: int = 30) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    if not KB_OPERATION_LOG_PATH.exists():
        return []

    def _read_tail_lines(path: Path, max_lines: int, block_size: int = 65536) -> List[str]:
        if max_lines <= 0:
            return []
        try:
            with path.open("rb") as fp:
                fp.seek(0, os.SEEK_END)
                end_pos = fp.tell()
                buffer = bytearray()
                lines: List[bytes] = []
                while end_pos > 0 and len(lines) <= max_lines:
                    read_size = min(block_size, end_pos)
                    end_pos -= read_size
                    fp.seek(end_pos)
                    buffer[:0] = fp.read(read_size)
                    if b"\n" in buffer:
                        parts = buffer.split(b"\n")
                        buffer = parts[0]
                        tail = parts[1:]
                        if tail:
                            lines = tail + lines
                if buffer:
                    lines = [buffer] + lines
                text_lines = [line.decode("utf-8", errors="ignore") for line in lines if line]
                return text_lines[-max_lines:]
        except Exception:
            return []

    logs: List[Dict[str, Any]] = []
    lines = _read_tail_lines(KB_OPERATION_LOG_PATH, limit * 4)
    for line in reversed(lines):
        if len(logs) >= limit:
            break
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                logs.append(payload)
        except Exception:
            continue

    return logs


def _load_review_queue_from_disk() -> List[Dict[str, Any]]:
    if not REVIEW_QUEUE_PATH.exists():
        return []

    try:
        payload = json.loads(REVIEW_QUEUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    schema_version = 1
    rows: List[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        schema_version = int(payload.get("schema_version", 1) or 1)
        rows = payload.get("items") or []
    else:
        return []

    queue: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        if not str(normalized.get("id", "")).strip():
            normalized["id"] = f"migrated_{uuid4().hex[:10]}"
        status = str(normalized.get("status", "pending")).strip().lower()
        if status not in {"pending", "approved", "rejected"}:
            status = "pending"
        normalized["status"] = status
        if not str(normalized.get("created_at", "")).strip():
            normalized["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        normalized["schema_version"] = REVIEW_QUEUE_SCHEMA_VERSION
        queue.append(normalized)

    if schema_version < REVIEW_QUEUE_SCHEMA_VERSION:
        try:
            REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
            migrated_payload = {
                "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(queue),
                "items": queue,
            }
            REVIEW_QUEUE_PATH.write_text(
                json.dumps(migrated_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    return queue


def persist_review_queue(queue: List[Dict[str, Any]] | None = None) -> Tuple[bool, str]:
    try:
        raw_queue = queue if queue is not None else (st.session_state.get("review_queue") or [])
        normalized_queue = [dict(item) for item in raw_queue if isinstance(item, dict)]
        for item in normalized_queue:
            item["schema_version"] = REVIEW_QUEUE_SCHEMA_VERSION

        REVIEW_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = REVIEW_QUEUE_PATH.with_suffix(".tmp")
        payload = {
            "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(normalized_queue),
            "items": normalized_queue,
        }
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(REVIEW_QUEUE_PATH)
        return True, f"已持久化 {len(normalized_queue)} 条审核记录"
    except Exception as exc:
        return False, str(exc)


def _persist_review_queue_or_warn() -> bool:
    ok, detail = persist_review_queue()
    if not ok:
        st.warning(f"审核队列持久化失败: {detail}")
        return False
    return True


def _is_append_effective_success(summary: Dict[str, Any]) -> bool:
    if not isinstance(summary, dict):
        return False
    if not summary.get("ok"):
        return False
    if int(summary.get("ingested_assets", 0) or 0) > 0:
        return True
    if str(summary.get("duplicate_of", "")).strip():
        return True
    return False


def _extract_blocking_p0_risks(risk_report: Any) -> List[Dict[str, str]]:
    rule_templates = load_badcase_rule_templates(project_root=PROJECT_ROOT)
    p0_rule = (rule_templates.get("templates", {}) or {}).get("p0_blocking", {})
    if not isinstance(p0_rule, dict):
        p0_rule = {}
    if not bool(p0_rule.get("enabled", True)):
        return []
    raw_severities = p0_rule.get("severities", ["P0"])
    if not isinstance(raw_severities, list):
        raw_severities = [raw_severities]
    enabled_severities = {
        str(item).strip().upper() for item in raw_severities if str(item).strip()
    } or {"P0"}

    if not isinstance(risk_report, dict):
        return []
    items = risk_report.get("items", [])
    if not isinstance(items, list):
        items = []
    p0_risks: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "")).strip().upper()
        if severity not in enabled_severities:
            continue
        p0_risks.append(
            {
                "id": str(item.get("id", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "category": str(item.get("category", "")).strip(),
            }
        )
    return p0_risks


def _op_status_label(item: Dict[str, Any]) -> str:
    if item.get("ok"):
        if item.get("partial_success"):
            return "部分成功"
        return "成功"
    return "失败"


def _op_type_label(op: str) -> str:
    mapping = {
        "sync": "知识库同步",
        "delete": "知识库删除",
        "append_generated": "生成结果入库",
    }
    return mapping.get(op, op)


def _render_sidebar_kb_operation_logs_panel(limit: int = 20) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("知识库更新日志")
    st.sidebar.caption("展示最近一次到最近 20 次操作（本机时间）。")

    logs = _load_kb_operation_logs(limit=limit)
    if not logs:
        st.sidebar.info("暂无操作日志。执行同步/删除/入库后会自动记录。")
        return

    def _pick_summary_value(summary: Dict[str, Any], key: str) -> Any:
        if key in summary:
            value = summary.get(key)
        elif f"{key}_count" in summary:
            value = summary.get(f"{key}_count")
        else:
            return None
        if isinstance(value, (list, dict)):
            return len(value)
        return value

    def _format_summary_value(value: Any) -> str:
        if value is None or value == "":
            return "-"
        if isinstance(value, bool):
            return "是" if value else "否"
        return str(value)

    def _format_extra_value(value: Any) -> str:
        if value is None or value == "":
            return "-"
        if isinstance(value, bool):
            return "是" if value else "否"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            cleaned = [str(v).strip() for v in value if str(v).strip()]
            if not cleaned:
                return "-"
            head = cleaned[:6]
            return " / ".join(head) + (" ..." if len(cleaned) > 6 else "")
        if isinstance(value, dict):
            return f"{len(value)} 项"
        return str(value)

    def _render_extra_info(extra: Dict[str, Any]) -> None:
        common_fields = [
            ("同步模式", "sync_mode"),
            ("入库状态", "kb_ingest_status"),
            ("版本标识", "kb_ingest_release"),
            ("模块标签", "kb_ingest_modules"),
            ("追踪ID", "kb_ingest_trace_refs"),
            ("资产数量", "assets_count"),
            ("选择数量", "selected_count"),
            ("删除原始文件", "delete_raw_asset"),
            ("生成模式", "generation_mode"),
            ("审核ID", "review_id"),
            ("来源名称", "source_name"),
            ("幂等跳过", "idempotent_skip"),
            ("内容哈希", "content_hash"),
        ]
        lines = []
        for label, key in common_fields:
            if key not in extra:
                continue
            value = extra.get(key)
            if value is None or value == "":
                continue
            lines.append(f"- {label}: {_format_extra_value(value)}")

        if lines:
            st.markdown("\n".join(lines))
        else:
            st.write("-")

        with st.expander("查看原始补充信息", expanded=False):
            st.json(extra)

    def _render_summary_group(title: str, items: List[tuple[str, Any]]) -> None:
        visible_lines = []
        for label, value in items:
            if value is None or value == "":
                continue
            visible_lines.append(f"- {label}: {_format_summary_value(value)}")
        st.markdown(f"**{title}**")
        if visible_lines:
            st.markdown("\n".join(visible_lines))
        else:
            st.write("-")

    for idx, item in enumerate(logs, start=1):
        time_text = _format_time_display(item.get("timestamp", ""))
        op_label = _op_type_label(str(item.get("operation", "")))
        status_text = _op_status_label(item)
        title = f"{idx}. {op_label} | {status_text} | {time_text}"
        with st.sidebar.expander(title, expanded=False):
            summary = item.get("summary", {}) or {}
            extra = item.get("extra", {}) or {}

            st.caption(f"操作类型: {op_label}")
            st.caption(f"执行状态: {status_text}")
            st.caption(f"执行时间: {time_text}")

            if summary:
                core_parts = []
                for label, key, unit in (
                    ("入库", "ingested_assets", "份"),
                    ("新增切片", "added_chunks", "个"),
                    ("删除资产", "deleted_assets", "份"),
                    ("删除切片", "deleted_chunks", "个"),
                    ("跳过", "skipped_assets", "份"),
                ):
                    value = _pick_summary_value(summary, key)
                    if value is None:
                        continue
                    core_parts.append(f"{label}{value}{unit}")

                error_count = _pick_summary_value(summary, "errors")
                warn_count = _pick_summary_value(summary, "warnings")
                exception_parts = []
                if isinstance(error_count, (int, float)) and error_count > 0:
                    exception_parts.append(f"失败{int(error_count)}条")
                if isinstance(warn_count, (int, float)) and warn_count > 0:
                    exception_parts.append(f"告警{int(warn_count)}条")

                summary_text = f"摘要：{op_label}，结果{status_text}"
                if core_parts:
                    summary_text += "，" + "，".join(core_parts)
                if exception_parts:
                    summary_text += "，" + "，".join(exception_parts)
                summary_text += "。"
                st.write(summary_text)

                _render_summary_group(
                    "执行结果",
                    [
                        ("整体结果", status_text),
                        ("是否成功", _pick_summary_value(summary, "ok")),
                        ("是否部分成功", _pick_summary_value(summary, "partial_success")),
                    ],
                )
                _render_summary_group(
                    "同步统计",
                    [
                        ("接收资产", _pick_summary_value(summary, "received_assets")),
                        ("接收记录", _pick_summary_value(summary, "received_doc_keys")),
                        ("入库资产", _pick_summary_value(summary, "ingested_assets")),
                        ("跳过资产", _pick_summary_value(summary, "skipped_assets")),
                        ("新增切片", _pick_summary_value(summary, "added_chunks")),
                        ("删除切片", _pick_summary_value(summary, "deleted_chunks")),
                        ("删除资产", _pick_summary_value(summary, "deleted_assets")),
                        ("删除原始文件", _pick_summary_value(summary, "deleted_raw_files")),
                    ],
                )
                _render_summary_group(
                    "质量与异常",
                    [
                        ("失败数", _pick_summary_value(summary, "errors")),
                        ("告警数", _pick_summary_value(summary, "warnings")),
                        ("缺失记录", _pick_summary_value(summary, "missing_doc_keys")),
                    ],
                )
                _render_summary_group(
                    "技术详情",
                    [
                        ("同步模式", _pick_summary_value(summary, "mode")),
                        ("影响来源数", _pick_summary_value(summary, "updated_sources")),
                        ("向量库条目数", _pick_summary_value(summary, "db_count")),
                    ],
                )

                with st.expander("查看原始结果", expanded=False):
                    st.json(summary)
            if extra:
                with st.expander("高级信息（调试用）", expanded=False):
                    _render_extra_info(extra)


def _kb_summary_caption(summary: Dict[str, Any]) -> str:
    if not summary:
        return ""

    if "deleted_assets" in summary or "deleted_chunks" in summary:
        return (
            "最近一次知识库变更结果: "
            f"删除资产 {summary.get('deleted_assets', 0)}，"
            f"删除切片 {summary.get('deleted_chunks', 0)}，"
            f"删除原始文件 {summary.get('deleted_raw_files', 0)}，"
            f"库内总切片 {summary.get('db_count', 0)}。"
        )

    return (
        "最近一次知识库变更结果: "
        f"接收资产 {summary.get('received_assets', 0)}，"
        f"成功入库 {summary.get('ingested_assets', 0)}，"
        f"新增切片 {summary.get('added_chunks', 0)}，"
        f"库内总切片 {summary.get('db_count', 0)}。"
    )


def _kb_index_to_rows(index_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = index_data.get("items") or {}
    rows: List[Dict[str, Any]] = []

    for doc_key, item in items.items():
        chunk_count = item.get("chunk_count", 0)
        try:
            chunk_count = int(chunk_count)
        except Exception:
            chunk_count = 0

        rows.append(
            {
                "doc_key": doc_key,
                "source_type": str(item.get("source_type", "unknown")),
                "origin": str(item.get("origin", "unknown")),
                "source_name": str(item.get("source_name", "unknown")),
                "chunk_count": chunk_count,
                "synced_at": str(item.get("synced_at", "")),
                "source_hash": str(item.get("source_hash", "")),
                "raw_asset_path": str(item.get("raw_asset_path", "")),
                "metadata": item.get("metadata", {}),
            }
        )

    rows.sort(key=lambda r: r.get("synced_at", ""), reverse=True)
    return rows


def _render_kb_assets_panel() -> None:
    index_data = _load_kb_index_data()
    rows = _kb_index_to_rows(index_data)

    st.subheader("知识库资产总览")

    if not rows:
        st.info("当前知识库暂无已同步资产。请在左侧边栏上传并同步。")
        return

    total_assets = len(rows)
    total_chunks = sum(r.get("chunk_count", 0) for r in rows)
    updated_at = _format_time_display(index_data.get("updated_at", ""))
    source_types = sorted({r["source_type"] for r in rows})
    origins = sorted({r["origin"] for r in rows})

    c1, c2, c3 = st.columns(3)
    c1.metric("资产总数", total_assets)
    c2.metric("向量切片总数", total_chunks)
    c3.metric("最近更新时间", updated_at)
    st.caption("时间显示为本机时间；可按来源类型、接入方式和关键字筛选。")

    with st.expander("查看与筛选资产明细", expanded=False):
        selected_types = st.multiselect(
            "按资产类型（source_type）筛选",
            options=source_types,
            default=source_types,
            key="kb_filter_types",
        )
        selected_origins = st.multiselect(
            "按接入来源（origin）筛选",
            options=origins,
            default=origins,
            key="kb_filter_origins",
        )
        keyword = st.text_input(
            "关键字筛选（source_name / doc_key）",
            key="kb_filter_keyword",
            placeholder="例如: 退票、figma、feishu_doc",
        ).strip().lower()

        filtered = []
        for row in rows:
            if row["source_type"] not in selected_types:
                continue
            if row["origin"] not in selected_origins:
                continue
            if keyword:
                haystack = f"{row['source_name']} {row['doc_key']}".lower()
                if keyword not in haystack:
                    continue
            filtered.append(row)

        st.caption(f"筛选结果: {len(filtered)} / {len(rows)}")

        if not filtered:
            st.warning("未匹配到资产，请调整筛选条件。")
            return

        for idx, row in enumerate(filtered, start=1):
            title = f"{idx}. [{row['source_type']}] {row['source_name']}"
            with st.expander(title, expanded=False):
                st.write(f"文档主键（doc_key）: `{row['doc_key']}`")
                st.write(f"接入来源（origin）: `{row['origin']}`")
                st.write(f"向量切片数（chunk_count）: `{row['chunk_count']}`")
                st.write(f"同步时间: `{_format_time_display(row['synced_at'])}`")
                st.write(f"内容哈希（前缀）: `{row['source_hash'][:16]}...`")
                if row["raw_asset_path"]:
                    st.write(f"原始文件路径: `{row['raw_asset_path']}`")
                if row["metadata"]:
                    st.caption("元数据")
                    st.json(row["metadata"])


def _render_sidebar_kb_delete_panel() -> None:
    st.sidebar.divider()
    st.sidebar.caption("低频操作（谨慎使用）")

    with st.sidebar.expander("知识库删除（放在底部，减少误操作）", expanded=False):
        last_delete_summary = st.session_state.get("kb_delete_summary")
        if last_delete_summary:
            if last_delete_summary.get("ok"):
                if last_delete_summary.get("partial_success"):
                    st.warning(
                        "最近一次删除部分成功: "
                        f"删除资产 {last_delete_summary.get('deleted_assets', 0)}，"
                        f"错误 {len(last_delete_summary.get('errors', []))}。"
                    )
                else:
                    st.success(
                        "最近一次删除成功: "
                        f"删除资产 {last_delete_summary.get('deleted_assets', 0)}，"
                        f"删除切片 {last_delete_summary.get('deleted_chunks', 0)}。"
                    )
            else:
                st.error(
                    "最近一次删除失败: "
                    + " | ".join(last_delete_summary.get("errors", [])[:2])
                )

        index_data = _load_kb_index_data()
        rows = _kb_index_to_rows(index_data)
        if not rows:
            st.info("当前无可删除资产。")
            return

        source_types = sorted({r["source_type"] for r in rows})
        origins = sorted({r["origin"] for r in rows})

        selected_types = st.multiselect(
            "删除筛选: 资产类型（source_type）",
            options=source_types,
            default=source_types,
            key="kb_delete_filter_types",
        )
        selected_origins = st.multiselect(
            "删除筛选: 接入来源（origin）",
            options=origins,
            default=origins,
            key="kb_delete_filter_origins",
        )
        keyword = st.text_input(
            "删除筛选: 关键字（source_name/doc_key）",
            key="kb_delete_filter_keyword",
            placeholder="例如: 退票、manual_testcase",
        ).strip().lower()

        candidates: List[Dict[str, Any]] = []
        for row in rows:
            if row["source_type"] not in selected_types:
                continue
            if row["origin"] not in selected_origins:
                continue
            if keyword:
                haystack = f"{row['source_name']} {row['doc_key']}".lower()
                if keyword not in haystack:
                    continue
            candidates.append(row)

        st.caption(f"可删除候选: {len(candidates)} / {len(rows)}")
        if not candidates:
            st.warning("没有符合条件的资产。")
            return

        label_to_doc_key: Dict[str, str] = {}
        options: List[str] = []
        for row in candidates:
            label = (
                f"[{row['source_type']}] {row['source_name']} "
                f"| 切片={row['chunk_count']} | {row['doc_key']}"
            )
            label_to_doc_key[label] = row["doc_key"]
            options.append(label)

        selected_labels = st.multiselect(
            "选择待删除资产（可多选）",
            options=options,
            default=[],
            key="kb_delete_selected_labels",
        )

        delete_raw_asset = st.checkbox(
            "同步删除原始附件文件（raw_asset_path）",
            value=True,
            key="kb_delete_remove_raw",
        )
        confirm_delete = st.checkbox(
            "我确认执行删除（不可恢复）",
            value=False,
            key="kb_delete_confirm",
        )

        if st.button(
            "执行删除并更新 Chroma",
            key="kb_delete_execute_btn",
            type="primary",
            use_container_width=True,
        ):
            if not selected_labels:
                st.warning("请先选择至少一条资产。")
            elif not confirm_delete:
                st.warning("请先勾选删除确认。")
            else:
                doc_keys = [label_to_doc_key[label] for label in selected_labels]
                try:
                    kb_module = load_kb_upsert_module()
                    with st.spinner("正在删除资产并更新 Chroma..."):
                        summary = kb_module.delete_assets(
                            doc_keys=doc_keys,
                            delete_raw_asset=delete_raw_asset,
                        )
                    st.session_state["kb_delete_summary"] = summary
                    st.session_state["kb_sync_summary"] = summary
                    _append_kb_operation_log(
                        operation="delete",
                        summary=summary,
                        extra={
                            "selected_count": len(doc_keys),
                            "delete_raw_asset": delete_raw_asset,
                        },
                    )
                    st.rerun()
                except Exception as exc:
                    error_summary = {"ok": False, "partial_success": False, "errors": [str(exc)]}
                    _append_kb_operation_log(
                        operation="delete",
                        summary=error_summary,
                        extra={
                            "selected_count": len(doc_keys),
                            "delete_raw_asset": delete_raw_asset,
                        },
                    )
                    st.error(f"删除异常: {exc}")


def _split_lines(raw: str) -> List[str]:
    return [line.strip() for line in (raw or "").splitlines() if line.strip()]


def _split_tags(raw: str) -> List[str]:
    values: List[str] = []
    for line in _split_lines(raw):
        for token in line.replace("，", ",").split(","):
            value = token.strip()
            if value and value not in values:
                values.append(value)
    return values


def _normalize_task_query_max_chars(value: Any, default: int = 60000) -> int:
    try:
        max_chars = int(str(value).strip())
    except Exception:
        max_chars = int(default)
    return min(200000, max(12000, max_chars))


def _build_base_asset_metadata(
    *,
    status: str,
    module_text: str,
    release_text: str,
    trace_refs_text: str,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "status": (status or "approved").strip().lower(),
    }

    modules = _split_tags(module_text)
    if modules:
        metadata["module"] = ",".join(modules)
        metadata["modules"] = modules

    release_value = (release_text or "").strip()
    if release_value:
        metadata["release"] = release_value

    trace_refs = _split_tags(trace_refs_text)
    if trace_refs:
        metadata["trace_refs"] = trace_refs

    return metadata


def _asset_name_from_ref(prefix: str, reference: str, index: int) -> str:
    tail = reference.rstrip("/").split("/")[-1].strip()
    safe_tail = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in tail)
    safe_tail = safe_tail[:48] or f"item_{index}"
    return f"{prefix}_{safe_tail}_{index}"


def _build_assets_from_uploaded_files(
    kb_module: Any,
    files: List[Any],
    *,
    source_type: str,
    base_metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    base_metadata = base_metadata or {}
    for f in files or []:
        payload = f.getvalue()
        suffix = Path(f.name).suffix.lower()
        metadata = {"uploaded_file_name": f.name}
        metadata.update(base_metadata)
        assets.append(
            kb_module.build_asset(
                source_type=source_type,
                origin="file_upload",
                source_name=f.name,
                suffix=suffix,
                content_bytes=payload,
                metadata=metadata,
            )
        )
    return assets


def _build_assets_from_multiline_refs(
    kb_module: Any,
    refs_text: str,
    *,
    source_type: str,
    origin: str,
    name_prefix: str,
    base_metadata: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    base_metadata = base_metadata or {}
    refs = _split_lines(refs_text)
    for idx, ref in enumerate(refs, start=1):
        metadata = {"reference": ref}
        metadata.update(base_metadata)
        assets.append(
            kb_module.build_asset(
                source_type=source_type,
                origin=origin,
                source_name=_asset_name_from_ref(name_prefix, ref, idx),
                external_ref=ref,
                metadata=metadata,
            )
        )
    return assets


def _decode_payload_text(payload: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return payload.decode(encoding)
        except Exception:
            continue
    return payload.decode("utf-8", errors="ignore")


def _csv_payload_to_text(payload: bytes) -> str:
    raw = _decode_payload_text(payload)
    reader = csv.DictReader(raw.splitlines())
    if not reader.fieldnames:
        return raw

    lines: List[str] = []
    for idx, row in enumerate(reader, start=1):
        kv = [f"{k}: {str(v).strip()}" for k, v in row.items() if str(v).strip()]
        if kv:
            lines.append(f"Row {idx}: " + " | ".join(kv))
    return "\n".join(lines) if lines else raw


def _extract_task_text_from_file(uploaded_file: Any) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    payload = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in {".txt", ".md", ".markdown", ".log", ".json", ".yaml", ".yml"}:
        text = _decode_payload_text(payload).strip()
        return text, warnings

    if suffix == ".csv":
        text = _csv_payload_to_text(payload).strip()
        return text, warnings

    if suffix in {".pdf", ".doc", ".docx"}:
        try:
            _bootstrap_import_path()
            from parsers.document_text import extract_text_from_document_bytes  # type: ignore

            text, warn = extract_text_from_document_bytes(payload, suffix)
            if warn:
                warnings.append(f"{uploaded_file.name}: {warn}")
            return text.strip(), warnings
        except Exception as exc:
            warnings.append(f"{uploaded_file.name}: 文档解析异常: {exc}")
            return "", warnings

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        try:
            from asset_loader import process_image_to_text_with_meta

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(payload)
                tmp_path = os.path.abspath(tmp.name)
            try:
                parse_result = process_image_to_text_with_meta(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            text = str(parse_result.get("markdown", "") or "").strip()
            engine = str(parse_result.get("engine", "") or "").strip()
            warn = str(parse_result.get("warning", "") or "").strip()
            if warn:
                warnings.append(f"{uploaded_file.name}: {warn}")
            if engine:
                warnings.append(f"{uploaded_file.name}: 图片解析方式={engine}")
            return text, warnings
        except Exception as exc:
            warnings.append(f"{uploaded_file.name}: 图片解析异常: {exc}")
            return "", warnings

    if suffix == ".xmind":
        try:
            _bootstrap_import_path()
            from parsers.xmind import parse_xmind_bytes  # type: ignore

            text = parse_xmind_bytes(payload).strip()
            return text, warnings
        except Exception as exc:
            warnings.append(f"{uploaded_file.name}: XMind 解析异常: {exc}")
            return "", warnings

    text = _decode_payload_text(payload).strip()
    if not text:
        warnings.append(f"{uploaded_file.name}: 暂不支持该文件类型或内容为空。")
    return text, warnings


def _compose_task_query(
    *,
    core_text: str,
    extra_text: str,
    links_text: str,
    files: List[Any],
    max_chars: int | None = None,
) -> Tuple[str, List[str], Dict[str, int]]:
    warnings: List[str] = []
    blocks: List[str] = []
    effective_max_chars = _normalize_task_query_max_chars(
        max_chars if max_chars is not None else os.getenv("TASK_QUERY_MAX_CHARS", "60000")
    )

    core_text = (core_text or "").strip()
    extra_text = (extra_text or "").strip()
    links = _split_lines(links_text)

    if core_text:
        blocks.append(f"【核心功能点（文本）】\n{core_text}")
    if extra_text:
        blocks.append(f"【补充说明】\n{extra_text}")
    if links:
        blocks.append("【关联链接】\n" + "\n".join(f"- {x}" for x in links))

    parsed_file_count = 0
    for f in files or []:
        text, file_warnings = _extract_task_text_from_file(f)
        warnings.extend(file_warnings)
        if text.strip():
            parsed_file_count += 1
            blocks.append(f"【文件输入: {f.name}】\n{text.strip()}")

    merged = "\n\n".join(blocks).strip()
    if len(merged) > effective_max_chars:
        merged = merged[:effective_max_chars] + "\n\n[提示] 输入过长，已自动截断。"
        warnings.append(f"本次输入超过 {effective_max_chars} 字符，已截断后生成。")

    stats = {
        "input_chars": len(merged),
        "file_count": len(files or []),
        "parsed_file_count": parsed_file_count,
        "link_count": len(links),
    }
    return merged, warnings, stats


def _render_sidebar(
    model_name: str,
    embedding_ok: bool,
    embedding_detail: str,
    chroma_ok: bool,
    chroma_detail: str,
) -> Dict[str, Any]:
    st.sidebar.header("系统配置")
    st.sidebar.write(f"当前模型: `{model_name}`")
    st.sidebar.markdown(
        f"Embedding 模型状态: {':green[已连接]' if embedding_ok else ':red[异常]'}"
    )
    st.sidebar.caption(embedding_detail)
    st.sidebar.markdown(
        f"Chroma 数据库状态: {':green[已连接]' if chroma_ok else ':red[异常]'}"
    )
    st.sidebar.caption(chroma_detail)

    st.sidebar.divider()

    st.sidebar.header("知识库动态更新")
    st.sidebar.caption("支持需求/测试用例/UI/API 资产融合入库")
    st.sidebar.caption(
        "说明: 飞书白板已支持直连（需 FEISHU_APP_ID/FEISHU_APP_SECRET）；"
        "飞书文档/Figma 仍为接入骨架。"
    )
    st.sidebar.caption("时间均按本机时区展示。")

    mode_options = [
        "1. 增量追加（推荐）: 新来源会入库，已存在同名来源将跳过",
        "2. 按来源替换: 若来源已存在，先删旧数据再重建该来源",
        "3. 全量重建: 清空后仅保留本批次导入内容",
    ]
    mode_label = st.sidebar.selectbox(
        "同步模式（请按场景选择）",
        options=mode_options,
        index=0,
        key="kb_sync_mode",
    )
    mode_map = {
        mode_options[0]: "append",
        mode_options[1]: "replace_by_source",
        mode_options[2]: "rebuild_all",
    }

    st.sidebar.caption("本批次统一元数据（用于检索过滤与追溯）")
    status_options = [
        "1. approved（已审核，可参与检索）",
        "2. draft（草稿，默认不参与检索）",
    ]
    kb_ingest_status = st.sidebar.selectbox(
        "本批次入库状态",
        options=status_options,
        index=0,
        key="kb_ingest_status",
        help="检索默认只使用 approved，draft 会被默认过滤。",
    )
    status_map = {
        status_options[0]: "approved",
        status_options[1]: "draft",
    }
    kb_ingest_release = st.sidebar.text_input(
        "本批次版本标识（可选）",
        key="kb_ingest_release",
        placeholder="例如: v2026.03, 2026Q1",
    )
    kb_ingest_modules = st.sidebar.text_input(
        "本批次模块标签（可选，逗号分隔）",
        key="kb_ingest_modules",
        placeholder="例如: 订单, 支付, 风控",
    )
    kb_ingest_trace_refs = st.sidebar.text_area(
        "本批次追踪ID（可选，每行或逗号分隔）",
        key="kb_ingest_trace_refs",
        height=60,
        placeholder="例如: REQ-123, API-45, UI-88",
    )

    req_files = st.sidebar.file_uploader(
        "需求资产文件（md/txt/csv/doc/docx/pdf/图片）",
        type=["md", "txt", "csv", "doc", "docx", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="kb_req_files",
    )
    req_text = st.sidebar.text_area("需求补充文字描述", key="kb_req_text", height=70)
    feishu_docs = st.sidebar.text_area(
        "飞书文档链接/Token（每行一个）",
        key="kb_feishu_docs",
        height=70,
        placeholder="https://xxx.feishu.cn/docx/...",
    )

    tc_files = st.sidebar.file_uploader(
        "测试用例资产（md/txt/csv/doc/docx/pdf/图片/xmind）",
        type=["md", "txt", "csv", "doc", "docx", "pdf", "png", "jpg", "jpeg", "xmind"],
        accept_multiple_files=True,
        key="kb_tc_files",
    )
    tc_text = st.sidebar.text_area("测试用例补充描述", key="kb_tc_text", height=70)
    feishu_boards = st.sidebar.text_area(
        "飞书画板链接/Token（每行一个）",
        key="kb_feishu_boards",
        height=70,
    )

    ui_files = st.sidebar.file_uploader(
        "UI交互资产（txt/md/doc/docx/pdf/图片）",
        type=["txt", "md", "doc", "docx", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="kb_ui_files",
    )
    ui_text = st.sidebar.text_area("UI交互补充描述", key="kb_ui_text", height=70)
    figma_refs = st.sidebar.text_area(
        "Figma 链接/File Key（每行一个）",
        key="kb_figma_refs",
        height=70,
    )

    api_files = st.sidebar.file_uploader(
        "API接口资产（json/yml/yaml/txt/md/doc/docx/pdf）",
        type=["json", "yml", "yaml", "txt", "md", "doc", "docx", "pdf"],
        accept_multiple_files=True,
        key="kb_api_files",
    )
    api_text = st.sidebar.text_area("API接口补充描述", key="kb_api_text", height=70)
    api_doc_links = st.sidebar.text_area(
        "API接口文档链接（每行一个）",
        key="kb_api_links",
        height=70,
        placeholder="https://api.example.com/openapi.yaml",
    )

    sync_clicked = st.sidebar.button(
        "同步至本地向量库",
        use_container_width=True,
        type="primary",
        key="kb_sync_btn",
    )

    return {
        "sync_clicked": sync_clicked,
        "sync_mode": mode_map[mode_label],
        "req_files": req_files or [],
        "req_text": req_text,
        "feishu_docs": feishu_docs,
        "tc_files": tc_files or [],
        "tc_text": tc_text,
        "feishu_boards": feishu_boards,
        "ui_files": ui_files or [],
        "ui_text": ui_text,
        "figma_refs": figma_refs,
        "api_files": api_files or [],
        "api_text": api_text,
        "api_doc_links": api_doc_links,
        "kb_ingest_status": status_map[kb_ingest_status],
        "kb_ingest_release": kb_ingest_release,
        "kb_ingest_modules": kb_ingest_modules,
        "kb_ingest_trace_refs": kb_ingest_trace_refs,
    }


def _init_session_state() -> None:
    st.session_state.setdefault("generated_markdown", "")
    st.session_state.setdefault("context_length", 0)
    st.session_state.setdefault("generated_at", "")
    st.session_state.setdefault("model_name", "待初始化")
    st.session_state.setdefault("kb_sync_summary", None)
    st.session_state.setdefault("kb_delete_summary", None)
    st.session_state.setdefault("kb_data_version", 0)
    if "review_queue" not in st.session_state:
        st.session_state["review_queue"] = _load_review_queue_from_disk()


def _canonicalize_markdown_text(markdown_text: str) -> str:
    lines = str(markdown_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = [line.rstrip() for line in lines]
    return "\n".join(normalized).strip()


def _hash_markdown_text(markdown_text: str) -> str:
    normalized = _canonicalize_markdown_text(markdown_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _find_existing_generated_by_hash(content_hash: str) -> Dict[str, Any] | None:
    if not content_hash:
        return None

    index_data = _load_kb_index_data()
    items = (index_data or {}).get("items", {}) or {}
    for doc_key, item in items.items():
        if not isinstance(item, dict):
            continue
        if str(item.get("source_type", "")).strip().lower() != "testcase":
            continue
        if str(item.get("origin", "")).strip().lower() != "llm_generated":
            continue

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("content_hash", "")).strip() != content_hash:
            continue

        return {
            "doc_key": str(doc_key),
            "source_name": str(item.get("source_name", "")),
            "synced_at": str(item.get("synced_at", "")),
        }
    return None


def _append_generated_markdown_to_kb(
    *,
    markdown_text: str,
    generation_mode: str,
    task_query: str,
    generated_at: str,
    review_id: str = "",
    module_text: str = "",
    release_text: str = "",
    trace_refs_text: str = "",
    risk_report: Dict[str, Any] | None = None,
    block_on_p0: bool = True,
) -> Dict[str, Any]:
    normalized_markdown = _canonicalize_markdown_text(markdown_text)
    if not normalized_markdown:
        return {
            "ok": False,
            "partial_success": False,
            "mode": "append",
            "received_assets": 1,
            "ingested_assets": 0,
            "skipped_assets": 1,
            "added_chunks": 0,
            "deleted_chunks": 0,
            "warnings": [],
            "errors": ["生成内容为空，已拒绝入库。"],
            "updated_sources": [],
            "db_count": 0,
        }

    p0_risks = _extract_blocking_p0_risks(risk_report or {})
    if block_on_p0 and p0_risks:
        risk_titles = [x.get("title", "") for x in p0_risks if x.get("title")]
        blocked_summary = {
            "ok": False,
            "partial_success": False,
            "mode": "append",
            "received_assets": 1,
            "ingested_assets": 0,
            "skipped_assets": 1,
            "added_chunks": 0,
            "deleted_chunks": 0,
            "warnings": [],
            "errors": [
                "命中 P0 风险阻断策略，已禁止入库。",
                ("P0风险: " + " / ".join(risk_titles[:3])) if risk_titles else "",
            ],
            "updated_sources": [],
            "db_count": 0,
            "blocked_by_policy": True,
            "block_reason": "risk_p0",
            "block_risks": p0_risks[:6],
        }
        blocked_summary["errors"] = [x for x in blocked_summary["errors"] if x]
        st.session_state["kb_sync_summary"] = blocked_summary
        _append_kb_operation_log(
            operation="append_generated",
            summary=blocked_summary,
            extra={
                "generation_mode": generation_mode,
                "review_id": review_id,
                "blocked_by_policy": True,
                "block_reason": "risk_p0",
                "p0_count": len(p0_risks),
                "p0_ids": [x.get("id", "") for x in p0_risks if x.get("id")][:6],
            },
        )
        return blocked_summary

    kb_module = load_kb_upsert_module()
    content_hash = _hash_markdown_text(normalized_markdown)
    duplicate = _find_existing_generated_by_hash(content_hash)
    if duplicate:
        duplicate_source = duplicate.get("source_name", "") or duplicate.get("doc_key", "-")
        append_summary = {
            "ok": True,
            "partial_success": False,
            "mode": "append",
            "received_assets": 1,
            "ingested_assets": 0,
            "skipped_assets": 1,
            "added_chunks": 0,
            "deleted_chunks": 0,
            "warnings": [f"检测到重复内容，已跳过。已存在资产: {duplicate_source}"],
            "errors": [],
            "updated_sources": [],
            "db_count": 0,
            "duplicate_of": duplicate_source,
        }
        st.session_state["kb_sync_summary"] = append_summary
        _append_kb_operation_log(
            operation="append_generated",
            summary=append_summary,
            extra={
                "generation_mode": generation_mode,
                "review_id": review_id,
                "source_name": duplicate_source,
                "idempotent_skip": True,
                "content_hash": content_hash,
            },
        )
        return append_summary

    suffix = f"_{review_id}" if review_id else ""
    source_name = f"generated_{generation_mode}_{generated_at}{suffix}.md"
    asset_metadata = _build_base_asset_metadata(
        status="approved",
        module_text=module_text,
        release_text=release_text,
        trace_refs_text=trace_refs_text,
    )
    asset_metadata.update(
        {
            "task_query": task_query[:2000],
            "generation_mode": generation_mode,
            "generated_at": generated_at,
            "review_id": review_id,
            "content_hash": content_hash,
            "content_chars": len(normalized_markdown),
        }
    )
    generated_asset = kb_module.build_asset(
        source_type="testcase",
        origin="llm_generated",
        source_name=source_name,
        suffix=".md",
        text=normalized_markdown,
        metadata=asset_metadata,
    )
    try:
        with st.spinner("正在将结果追加到知识库..."):
            append_summary = kb_module.ingest_assets(assets=[generated_asset], mode="append")
    except Exception as exc:
        _append_kb_operation_log(
            operation="append_generated",
            summary={"ok": False, "partial_success": False, "errors": [str(exc)]},
            extra={
                "generation_mode": generation_mode,
                "review_id": review_id,
                "source_name": source_name,
            },
        )
        raise

    st.session_state["kb_sync_summary"] = append_summary
    _append_kb_operation_log(
        operation="append_generated",
        summary=append_summary,
        extra={
            "generation_mode": generation_mode,
            "review_id": review_id,
            "source_name": source_name,
        },
    )
    return append_summary


def _render_review_queue_panel() -> None:
    queue: List[Dict[str, Any]] = st.session_state.get("review_queue") or []

    st.subheader("审核入库区")

    pending_count = sum(1 for item in queue if item.get("status") == "pending")
    approved_count = sum(1 for item in queue if item.get("status") == "approved")
    rejected_count = sum(1 for item in queue if item.get("status") == "rejected")
    c1, c2, c3 = st.columns(3)
    c1.metric("待审核", pending_count)
    c2.metric("已入库", approved_count)
    c3.metric("已驳回", rejected_count)

    if not queue:
        st.info("暂无待审核内容。可在“生成结果入库策略”中选择“加入待审核队列”。")
        return

    if st.button("清理已处理记录（仅保留待审核）", key="review_cleanup_done"):
        st.session_state["review_queue"] = [
            item for item in queue if item.get("status") == "pending"
        ]
        if _persist_review_queue_or_warn():
            st.success("已清理已入库/已驳回记录。")
            st.rerun()

    status_text_map = {
        "pending": "待审核",
        "approved": "已入库",
        "rejected": "已驳回",
    }

    remove_ids: set[str] = set()

    for item in queue:
        item_id = str(item.get("id", ""))
        status = str(item.get("status", "pending"))
        created_at = str(item.get("created_at", "-"))
        generation_mode = str(item.get("generation_mode", "-"))
        title = f"{status_text_map.get(status, status)} | {created_at} | {generation_mode}"

        with st.expander(title, expanded=(status == "pending")):
            st.caption(f"检索上下文长度: {int(item.get('context_length', 0))} 字符")
            task_preview = str(item.get("task_query", "")).strip()
            if task_preview:
                st.caption("任务快照（截断）")
                st.code(task_preview[:600], language="text")

            module_text = str(item.get("module_text", "")).strip()
            release_text = str(item.get("release_text", "")).strip()
            if module_text or release_text:
                st.caption(
                    "元数据: "
                    f"module={module_text or '-'} | release={release_text or '-'}"
                )

            content_key = f"review_content_{item_id}"
            if content_key not in st.session_state:
                st.session_state[content_key] = str(item.get("content", ""))
            edited_content = st.text_area(
                "审核内容（可删改后再通过入库）",
                key=content_key,
                height=260,
            )

            approve_disabled = status == "approved"
            reject_disabled = status == "approved"
            cols = st.columns(3)

            if cols[0].button(
                "通过并追加到知识库",
                key=f"review_approve_{item_id}",
                type="primary",
                disabled=approve_disabled,
                use_container_width=True,
            ):
                if not edited_content.strip():
                    st.warning("审核内容为空，无法入库。")
                else:
                    try:
                        append_summary = _append_generated_markdown_to_kb(
                            markdown_text=edited_content,
                            generation_mode=generation_mode,
                            task_query=task_preview,
                            generated_at=str(item.get("generated_at", created_at)),
                            review_id=item_id,
                            module_text=str(item.get("module_text", "")),
                            release_text=str(item.get("release_text", "")),
                            trace_refs_text=str(item.get("trace_refs_text", "")),
                            risk_report=item.get("risk_report", {}),
                        )
                        if _is_append_effective_success(append_summary):
                            item["status"] = "approved"
                            item["content"] = edited_content
                            item["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            _persist_review_queue_or_warn()
                            if int(append_summary.get("ingested_assets", 0) or 0) > 0:
                                st.success("审核通过，已追加进知识库。")
                            else:
                                st.success("审核通过：内容已存在知识库，本次未重复写入。")
                        elif bool(append_summary.get("blocked_by_policy", False)):
                            st.warning("阻断入库: " + " | ".join(append_summary.get("errors", [])[:2]))
                        else:
                            st.error(
                                "追加失败: "
                                + " | ".join(append_summary.get("errors", [])[:2])
                            )
                    except Exception as exc:
                        st.error(f"审核入库异常: {exc}")

            if cols[1].button(
                "驳回",
                key=f"review_reject_{item_id}",
                disabled=reject_disabled,
                use_container_width=True,
            ):
                item["status"] = "rejected"
                item["content"] = edited_content
                item["rejected_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _persist_review_queue_or_warn()
                st.info("已驳回，该条不会入库。")

            if cols[2].button(
                "移除记录",
                key=f"review_remove_{item_id}",
                use_container_width=True,
            ):
                remove_ids.add(item_id)

            if status == "approved":
                st.success(f"入库时间: {item.get('approved_at', '-')}")
            elif status == "rejected":
                st.warning(f"驳回时间: {item.get('rejected_at', '-')}")

    if remove_ids:
        st.session_state["review_queue"] = [
            item for item in queue if str(item.get("id", "")) not in remove_ids
        ]
        if _persist_review_queue_or_warn():
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="测试用例生成平台", layout="wide")

    _init_session_state()

    st.title("测试用例生成平台")

    embedding_ok, embedding_detail = _ensure_embedding_env_path()
    chroma_ok, chroma_detail = _check_chroma_status()
    sync_request = _render_sidebar(
        st.session_state["model_name"],
        embedding_ok=embedding_ok,
        embedding_detail=embedding_detail,
        chroma_ok=chroma_ok,
        chroma_detail=chroma_detail,
    )

    if sync_request["sync_clicked"]:
        assets: List[Dict[str, Any]] = []
        sync_log_extra = {
            "sync_mode": sync_request.get("sync_mode", ""),
            "ingest_status": sync_request.get("kb_ingest_status", ""),
            "ingest_release": (sync_request.get("kb_ingest_release", "") or "").strip(),
            "ingest_modules": (sync_request.get("kb_ingest_modules", "") or "").strip(),
        }
        try:
            kb_module = load_kb_upsert_module()
            base_asset_metadata = _build_base_asset_metadata(
                status=sync_request["kb_ingest_status"],
                module_text=sync_request["kb_ingest_modules"],
                release_text=sync_request["kb_ingest_release"],
                trace_refs_text=sync_request["kb_ingest_trace_refs"],
            )

            assets.extend(
                _build_assets_from_uploaded_files(
                    kb_module,
                    sync_request["req_files"],
                    source_type="requirement",
                    base_metadata=base_asset_metadata,
                )
            )
            req_text = (sync_request["req_text"] or "").strip()
            if req_text:
                assets.append(
                    kb_module.build_asset(
                        source_type="requirement",
                        origin="manual_text",
                        source_name=f"manual_requirement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        text=req_text,
                        metadata=dict(base_asset_metadata),
                    )
                )
            assets.extend(
                _build_assets_from_multiline_refs(
                    kb_module,
                    sync_request["feishu_docs"],
                    source_type="requirement",
                    origin="feishu_doc",
                    name_prefix="feishu_doc",
                    base_metadata=base_asset_metadata,
                )
            )

            assets.extend(
                _build_assets_from_uploaded_files(
                    kb_module,
                    sync_request["tc_files"],
                    source_type="testcase",
                    base_metadata=base_asset_metadata,
                )
            )
            tc_text = (sync_request["tc_text"] or "").strip()
            if tc_text:
                assets.append(
                    kb_module.build_asset(
                        source_type="testcase",
                        origin="manual_text",
                        source_name=f"manual_testcase_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        text=tc_text,
                        metadata=dict(base_asset_metadata),
                    )
                )
            assets.extend(
                _build_assets_from_multiline_refs(
                    kb_module,
                    sync_request["feishu_boards"],
                    source_type="testcase",
                    origin="feishu_board",
                    name_prefix="feishu_board",
                    base_metadata=base_asset_metadata,
                )
            )

            assets.extend(
                _build_assets_from_uploaded_files(
                    kb_module,
                    sync_request["ui_files"],
                    source_type="ui",
                    base_metadata=base_asset_metadata,
                )
            )
            ui_text = (sync_request["ui_text"] or "").strip()
            if ui_text:
                assets.append(
                    kb_module.build_asset(
                        source_type="ui",
                        origin="manual_text",
                        source_name=f"manual_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        text=ui_text,
                        metadata=dict(base_asset_metadata),
                    )
                )
            assets.extend(
                _build_assets_from_multiline_refs(
                    kb_module,
                    sync_request["figma_refs"],
                    source_type="ui",
                    origin="figma",
                    name_prefix="figma",
                    base_metadata=base_asset_metadata,
                )
            )

            assets.extend(
                _build_assets_from_uploaded_files(
                    kb_module,
                    sync_request["api_files"],
                    source_type="api_doc",
                    base_metadata=base_asset_metadata,
                )
            )
            api_text = (sync_request["api_text"] or "").strip()
            if api_text:
                assets.append(
                    kb_module.build_asset(
                        source_type="api_doc",
                        origin="manual_text",
                        source_name=f"manual_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        text=api_text,
                        metadata=dict(base_asset_metadata),
                    )
                )
            assets.extend(
                _build_assets_from_multiline_refs(
                    kb_module,
                    sync_request["api_doc_links"],
                    source_type="api_doc",
                    origin="api_doc_link",
                    name_prefix="api_doc_link",
                    base_metadata=base_asset_metadata,
                )
            )

            if not assets:
                st.sidebar.warning("未检测到可同步资产，请上传文件或填写文本/链接。")
                _append_kb_operation_log(
                    operation="sync",
                    summary={
                        "ok": False,
                        "partial_success": False,
                        "received_assets": 0,
                        "errors": ["未检测到可同步资产"],
                    },
                    extra=sync_log_extra,
                )
            else:
                with st.spinner("正在同步知识库资产到 Chroma..."):
                    summary = kb_module.ingest_assets(assets=assets, mode=sync_request["sync_mode"])
                st.session_state["kb_sync_summary"] = summary
                _append_kb_operation_log(
                    operation="sync",
                    summary=summary,
                    extra={**sync_log_extra, "assets_count": len(assets)},
                )
                if summary.get("ok"):
                    if summary.get("partial_success"):
                        st.sidebar.warning(
                            "同步部分成功："
                            f"入库 {summary.get('ingested_assets', 0)} 份，"
                            f"失败 {len(summary.get('errors', []))} 份。"
                        )
                    else:
                        st.sidebar.success(
                            "同步成功："
                            f"入库 {summary.get('ingested_assets', 0)} 份，"
                            f"新增 {summary.get('added_chunks', 0)} 个切片。"
                        )
                else:
                    st.sidebar.error(
                        "同步失败："
                        f"成功 {summary.get('ingested_assets', 0)} 份，"
                        f"失败 {len(summary.get('errors', []))} 份。"
                    )
        except Exception as exc:
            _append_kb_operation_log(
                operation="sync",
                summary={
                    "ok": False,
                    "partial_success": False,
                    "received_assets": len(assets),
                    "errors": [str(exc)],
                },
                extra=sync_log_extra,
            )
            st.sidebar.error(f"知识库同步异常: {exc}")

    kb_summary = st.session_state.get("kb_sync_summary")
    if kb_summary:
        st.sidebar.divider()
        st.sidebar.caption(_kb_summary_caption(kb_summary))
        if kb_summary.get("warnings"):
            st.sidebar.info(
                "同步告警: " + " | ".join(kb_summary["warnings"][:2])
                + (" ..." if len(kb_summary["warnings"]) > 2 else "")
            )
        if kb_summary.get("errors"):
            st.sidebar.error(
                "同步错误: " + " | ".join(kb_summary["errors"][:2])
                + (" ..." if len(kb_summary["errors"]) > 2 else "")
            )
    _render_sidebar_kb_operation_logs_panel(limit=20)

    _render_kb_assets_panel()

    generation_mode_label = st.selectbox(
        "接口用例生成模式",
        options=["1. 业务接口用例（业务流程优先）", "2. 字段校验用例（字段级）"],
        index=0,
        key="generation_mode_label",
    )
    generation_mode_map = {
        "1. 业务接口用例（业务流程优先）": "business_api",
        "2. 字段校验用例（字段级）": "field_validation",
    }
    generation_mode = generation_mode_map[generation_mode_label]

    append_strategy_label = st.radio(
        "生成结果入库策略",
        options=[
            "1. 加入待审核队列（推荐）",
            "2. 仅生成，不入库",
            "3. 直接追加到知识库（跳过审核）",
        ],
        index=0,
        horizontal=True,
        key="append_strategy_label",
    )
    append_strategy_map = {
        "1. 加入待审核队列（推荐）": "review_queue",
        "2. 仅生成，不入库": "none",
        "3. 直接追加到知识库（跳过审核）": "direct_append",
    }
    append_strategy = append_strategy_map[append_strategy_label]

    retrieval_approved_only = st.checkbox(
        "检索仅使用已审核知识（approved）",
        value=True,
        key="retrieval_approved_only",
    )
    retrieval_release_filter = st.text_input(
        "检索版本过滤（可选）",
        key="retrieval_release_filter",
        placeholder="例如: v2026.03",
    )
    retrieval_module_filter = st.text_input(
        "检索模块过滤（可选，逗号分隔）",
        key="retrieval_module_filter",
        placeholder="例如: 订单, 支付",
    )
    generation_trace_refs = st.text_input(
        "本次生成追踪ID（可选，逗号分隔）",
        key="generation_trace_refs",
        placeholder="例如: REQ-123, API-45",
    )
    task_query_max_chars_default = _normalize_task_query_max_chars(
        os.getenv("TASK_QUERY_MAX_CHARS", "60000")
    )
    task_query_max_chars = int(
        st.number_input(
            "输入长度上限（字符）",
            min_value=12000,
            max_value=200000,
            value=task_query_max_chars_default,
            step=2000,
            key="task_query_max_chars",
            help="用于控制本次生成时文本+文件解析内容拼接后的最大长度，超出后会截断。",
        )
    )

    task_core_text = st.text_area(
        "请输入本次待测的核心功能点（文本）",
        height=160,
        placeholder="示例: 增加观演人维度退票次数限制；每个实名观演人最多可发起 X 次退票。",
        key="task_core_text",
    )
    task_extra_text = st.text_area(
        "补充说明（可选）",
        height=100,
        placeholder="可补充边界条件、业务状态、角色差异、时间规则等；可写“模块: 订单, 支付, 风控”增强跨模块检索。",
        key="task_extra_text",
    )
    task_links_text = st.text_area(
        "本次待测关联链接（可选，每行一个）",
        height=100,
        placeholder="需求/接口/原型链接都可填写。",
        key="task_links_text",
    )
    task_input_files = st.file_uploader(
        "本次待测输入资产（可选）",
        type=[
            "txt",
            "md",
            "csv",
            "json",
            "yml",
            "yaml",
            "doc",
            "docx",
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "xmind",
        ],
        accept_multiple_files=True,
        key="task_input_files",
    )

    generate_clicked = st.button("一键生成业务测试用例（支持多源输入）", type="primary", use_container_width=True)

    if generate_clicked:
        composed_task_query, task_input_warnings, task_input_stats = _compose_task_query(
            core_text=task_core_text,
            extra_text=task_extra_text,
            links_text=task_links_text,
            files=task_input_files or [],
            max_chars=task_query_max_chars,
        )

        if not composed_task_query.strip():
            st.warning("请至少提供一种输入：文本、链接或文件。")
        else:
            st.caption(
                "本次输入汇总: "
                f"{task_input_stats.get('input_chars', 0)} 字符 | "
                f"文件 {task_input_stats.get('file_count', 0)} 个（成功解析 {task_input_stats.get('parsed_file_count', 0)}） | "
                f"链接 {task_input_stats.get('link_count', 0)} 条"
            )
            if task_input_warnings:
                st.info(
                    "输入解析提示: " + " | ".join(task_input_warnings[:2])
                    + (" ..." if len(task_input_warnings) > 2 else "")
                )
            st.caption(
                "检索策略: "
                f"approved_only={'是' if retrieval_approved_only else '否'} | "
                f"release={retrieval_release_filter.strip() or '-'} | "
                f"modules={retrieval_module_filter.strip() or '-'}"
            )

            try:
                os.environ["RAG_APPROVED_ONLY"] = "1" if retrieval_approved_only else "0"
                os.environ["RAG_FILTER_RELEASE"] = (retrieval_release_filter or "").strip()
                os.environ["RAG_FILTER_MODULES"] = (retrieval_module_filter or "").strip()
                os.environ.setdefault("RAG_INCLUDE_LEGACY_UNLABELED", "1")
                retrieval_policy = {
                    "approved_only": retrieval_approved_only,
                    "release": (retrieval_release_filter or "").strip(),
                    "modules": _split_tags(retrieval_module_filter or ""),
                    "include_legacy_unlabeled": True,
                }

                with st.spinner("正在从 Chroma 知识库深挖业务逻辑..."):
                    get_augmented_context, llm, universal_template = load_pipeline_components()
                    model_name = (
                        getattr(llm, "model_name", None)
                        or getattr(llm, "model", None)
                        or "unknown"
                    )
                    st.session_state["model_name"] = model_name
                    context = (
                        get_augmented_context(
                            composed_task_query,
                            retrieval_policy=retrieval_policy,
                        )
                        or ""
                    )

                st.session_state["context_length"] = len(context)
                if not context.strip():
                    st.warning(
                        "检索结果为空。请检查过滤条件（approved/release/module），"
                        "或先同步对应模块且状态为 approved 的知识数据。"
                    )
                    st.stop()
                st.success(f"检索完成，召回核心业务上下文: {len(context)} 字符")

                final_prompt = _build_prompt(
                    template=universal_template,
                    context=context,
                    task_query=composed_task_query,
                )
                mode_instruction = _generation_mode_instruction(generation_mode)
                final_prompt = (
                    f"{final_prompt}\n\n"
                    "【本次生成模式追加指令（高优先级）】\n"
                    f"{mode_instruction}\n"
                )
            except Exception as exc:
                exc_text = str(exc)
                if (
                    ("all-MiniLM-L6-v2" in exc_text)
                    or ("huggingface.co" in exc_text)
                    or ("EMBEDDING_MODEL_PATH" in exc_text)
                    or ("config.json" in exc_text and "Embedding" in exc_text)
                ):
                    st.error(
                        "检索阶段异常: 本地 Embedding 模型未就绪（all-MiniLM-L6-v2）。\n\n"
                        "请先确认 HuggingFace 本地缓存完整，或设置 EMBEDDING_MODEL_PATH 指向包含 config.json 的本地模型目录。"
                    )
                else:
                    st.error(f"检索阶段异常: {exc}")
                st.stop()

            generated_markdown = ""
            try:
                st.info("AI 正在根据输入内容推演业务场景，请稍候...")

                can_stream = hasattr(st, "write_stream") and callable(getattr(llm, "stream", None))
                if can_stream:
                    stream_holder = st.empty()
                    with stream_holder.container():
                        stream_result = st.write_stream(_stream_llm_tokens(llm, final_prompt))

                    if isinstance(stream_result, str):
                        generated_markdown = stream_result
                    elif isinstance(stream_result, list):
                        generated_markdown = "".join(str(item) for item in stream_result)
                    else:
                        generated_markdown = str(stream_result or "")

                    stream_holder.empty()
                else:
                    response = llm.invoke(final_prompt)
                    generated_markdown = _normalize_chunk_content(response)

                if not generated_markdown.strip():
                    st.warning("模型返回为空，请调整输入后重试。")

                st.session_state["generated_markdown"] = generated_markdown
                st.session_state["generated_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")

                if generated_markdown.strip():
                    if append_strategy == "direct_append":
                        try:
                            append_summary = _append_generated_markdown_to_kb(
                                markdown_text=generated_markdown,
                                generation_mode=generation_mode,
                                task_query=composed_task_query,
                                generated_at=st.session_state["generated_at"],
                                module_text=retrieval_module_filter,
                                release_text=retrieval_release_filter,
                                trace_refs_text=generation_trace_refs,
                            )
                            if _is_append_effective_success(append_summary):
                                if int(append_summary.get("ingested_assets", 0) or 0) > 0:
                                    st.success("已将本次生成结果直接追加进知识库。")
                                else:
                                    st.success("检测到重复内容，本次未重复入库（幂等命中）。")
                            else:
                                st.warning(
                                    "生成成功，但追加知识库失败："
                                    + " | ".join(append_summary.get("errors", [])[:2])
                                )
                        except Exception as exc:
                            st.warning(f"生成成功，但追加知识库异常: {exc}")

                    if append_strategy == "review_queue":
                        review_id = f"{st.session_state['generated_at']}_{uuid4().hex[:8]}"
                        st.session_state["review_queue"].insert(
                            0,
                            {
                                "id": review_id,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "generated_at": st.session_state["generated_at"],
                                "status": "pending",
                                "generation_mode": generation_mode,
                                "task_query": composed_task_query[:3000],
                                "context_length": st.session_state["context_length"],
                                "content": generated_markdown,
                                "module_text": retrieval_module_filter,
                                "release_text": retrieval_release_filter,
                                "trace_refs_text": generation_trace_refs,
                            },
                        )
                        _persist_review_queue_or_warn()
                        st.success("已加入待审核队列。你可以在下方逐条审核后再入库。")
            except Exception as exc:
                st.error(f"生成阶段异常: {exc}")

    if st.session_state["context_length"]:
        st.caption(f"最近一次检索上下文长度: {st.session_state['context_length']} 字符")

    if st.session_state["generated_markdown"]:
        st.subheader("测试用例输出区")
        st.markdown(st.session_state["generated_markdown"])

        download_name = f"TestCases_{st.session_state['generated_at']}.md"
        st.download_button(
            label="下载为 Markdown 文件",
            data=st.session_state["generated_markdown"],
            file_name=download_name,
            mime="text/markdown",
            use_container_width=True,
        )

    _render_review_queue_panel()
    _render_sidebar_kb_delete_panel()


if __name__ == "__main__":
    main()
