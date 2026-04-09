import os
import sys
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

# ==========================================
# 【核心修正】：必须在所有 LangChain 导入之前设置！
# 1. 强制走国内镜像站（解决 modules.json 下载超时的终极方案）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 2. 告诉底层 Transformers 库尽量使用离线缓存
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
# ==========================================

import httpx
import socket
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ... 后面的代码保持不变 ...

# --- 新增：Hugging Face 网络隔离补丁 ---
# 1. 强制使用本地缓存，严禁向外网发送请求检查更新
os.environ['HF_HUB_OFFLINE'] = '1'
# 2. 兜底方案：如果必须连网，强制走国内镜像加速节点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. 核心配置加载
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# 变量提取逻辑：兼容多种命名并提供兜底
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("DEEPSEEK_API_BASE")

if not API_KEY:
    raise ValueError("[ERROR] 环境变量中缺失 API_KEY，请检查 .env 文件。")

# 2. 路径处理
persist_directory = str(PROJECT_ROOT / "data" / "chroma_db")


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on", "y", "t"}


RETRIEVAL_TOP_K_PER_QUERY = max(3, _get_int_env("RAG_TOP_K_PER_QUERY", 8))
RETRIEVAL_TARGET_DOCS = max(6, _get_int_env("RAG_TARGET_DOCS", 16))
RETRIEVAL_MAX_CONTEXT_CHARS = max(2000, _get_int_env("RAG_MAX_CONTEXT_CHARS", 12000))
SOURCE_TYPE_ORDER = ["requirement", "testcase", "api_doc", "ui", "unknown"]
SOURCE_TYPE_LABELS = {
    "requirement": "需求与业务规则",
    "testcase": "历史测试用例",
    "api_doc": "接口文档与字段定义",
    "ui": "UI交互与页面行为",
    "unknown": "其他来源",
}

_VECTORDB = None
_VECTORDB_LOCK = Lock()

def _resolve_local_embedding_model_path():
    """
    解析本地 Embedding 模型路径，避免触发 HuggingFace 联网下载。
    优先级：
    1) 环境变量 EMBEDDING_MODEL_PATH
    2) HuggingFace 默认缓存目录下的 snapshot
    """
    custom_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    if custom_model_path:
        custom_path = Path(custom_model_path).expanduser().resolve()
        if (custom_path / "config.json").exists():
            return str(custom_path)
        raise FileNotFoundError(
            f"EMBEDDING_MODEL_PATH 无效：{custom_path}（缺少 config.json）"
        )

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
        " 请先下载到 HuggingFace 缓存，或设置 EMBEDDING_MODEL_PATH 指向本地模型目录。"
    )

def _extract_module_hints(task_query: str) -> List[str]:
    hints: List[str] = []
    pattern = re.compile(r"(?:模块|系统|子系统|领域|domain)[:：]\s*([^\n]+)", flags=re.IGNORECASE)

    for match in pattern.finditer(task_query or ""):
        raw = match.group(1)
        for token in re.split(r"[，,、;/；|]+", raw):
            value = token.strip().strip("[]【】()（）")
            if 1 < len(value) <= 20 and value not in hints:
                hints.append(value)

    return hints[:8]


def _build_retrieval_queries(task_query: str) -> List[str]:
    base = (task_query or "").strip()
    if not base:
        return []

    queries = [
        base,
        f"需求 业务规则 边界条件 状态流转 {base}",
        f"历史测试用例 业务场景 异常场景 覆盖点 {base}",
        f"接口文档 API 参数 字段 约束 错误码 {base}",
        f"UI 页面 交互 文案 提示 联动 {base}",
    ]

    for module in _extract_module_hints(base):
        queries.append(f"{module} 跨模块 联动 场景 用例 {base}")

    deduped: List[str] = []
    seen = set()
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def _parse_multi_values(raw: str) -> List[str]:
    values: List[str] = []
    for token in re.split(r"[,\n，、;/；|]+", str(raw or "")):
        value = token.strip().strip("[]【】()（）")
        if value and value not in values:
            values.append(value)
    return values


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _pick_first_non_empty_meta(metadata: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_bool_value(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "on", "是", "通过", "已审核", "approved"}:
        return True
    if text in {"0", "false", "no", "n", "off", "否", "不通过", "未审核", "rejected"}:
        return False
    return None


def _normalize_meta_values(value: Any, max_items: int = 3) -> List[str]:
    items: List[str] = []
    if value is None:
        return items
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = _parse_multi_values(str(value))
    for item in raw_items:
        text = str(item).strip()
        if text and text not in items:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _format_meta_tag(label: str, value: Any, max_items: int = 3, max_len: int = 80) -> str:
    items = _normalize_meta_values(value, max_items=max_items)
    if not items:
        return ""
    joined = ",".join(items)
    if len(joined) > max_len:
        joined = joined[: max_len - 3] + "..."
    return f" | {label}={joined}"


def _get_runtime_retrieval_policy(override: Dict[str, Any] | None = None) -> Dict[str, Any]:
    override = override or {}
    modules_override = override.get("modules", [])
    if isinstance(modules_override, str):
        modules = _parse_multi_values(modules_override)
    elif isinstance(modules_override, list):
        modules = [str(x).strip() for x in modules_override if str(x).strip()]
    else:
        modules = _parse_multi_values(os.getenv("RAG_FILTER_MODULES", ""))

    release_value = override.get("release")
    if release_value is None:
        release = str(os.getenv("RAG_FILTER_RELEASE", "")).strip()
    else:
        release = str(release_value).strip()

    approved_only_override = override.get("approved_only")
    if approved_only_override is None:
        approved_only = _get_bool_env("RAG_APPROVED_ONLY", True)
    else:
        approved_only = bool(approved_only_override)

    include_legacy_override = override.get("include_legacy_unlabeled")
    if include_legacy_override is None:
        include_legacy = _get_bool_env("RAG_INCLUDE_LEGACY_UNLABELED", True)
    else:
        include_legacy = bool(include_legacy_override)

    return {
        "approved_only": approved_only,
        "release": release,
        "modules": modules,
        "include_legacy_unlabeled": include_legacy,
    }


def _get_cached_vectordb() -> Chroma:
    global _VECTORDB
    if _VECTORDB is not None:
        return _VECTORDB
    with _VECTORDB_LOCK:
        if _VECTORDB is None:
            local_model_path = _resolve_local_embedding_model_path()
            embeddings = HuggingFaceEmbeddings(model_name=local_model_path)
            _VECTORDB = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return _VECTORDB


def clear_retrieval_runtime_cache() -> None:
    global _VECTORDB
    with _VECTORDB_LOCK:
        _VECTORDB = None


def _extract_doc_modules(metadata: Dict[str, Any]) -> List[str]:
    keys = [
        "modules",
        "module",
        "module_name",
        "business_module",
        "domain",
        "ext_module",
        "ext_modules",
        "ext_module_name",
        "ext_domain",
        "ext_business_module",
    ]
    modules: List[str] = []
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            items = value
        else:
            items = _parse_multi_values(str(value))
        for item in items:
            normalized = str(item).strip()
            if normalized and normalized not in modules:
                modules.append(normalized)
    return modules[:8]


def _extract_doc_release(metadata: Dict[str, Any]) -> str:
    raw_release = _pick_first_non_empty_meta(metadata, ["release", "ext_release"])
    return str(raw_release or "").strip()


def _is_doc_approved(metadata: Dict[str, Any], include_legacy_unlabeled: bool) -> bool:
    _ = include_legacy_unlabeled

    approved_value = _normalize_bool_value(
        _pick_first_non_empty_meta(metadata, ["approved", "ext_approved"])
    )
    if approved_value is not None:
        return approved_value

    candidates = [
        _normalize_text(metadata.get("status")),
        _normalize_text(metadata.get("ingest_status")),
        _normalize_text(metadata.get("review_status")),
        _normalize_text(metadata.get("ext_status")),
        _normalize_text(metadata.get("ext_ingest_status")),
        _normalize_text(metadata.get("ext_review_status")),
    ]
    status = ""
    for item in candidates:
        if item:
            status = item
            break

    if not status or status == "unknown":
        return False

    if status in {"approved", "published", "pass", "passed", "已审核", "已入库", "通过"}:
        return True
    return False


def _passes_runtime_filters(metadata: Dict[str, Any], policy: Dict[str, Any]) -> bool:
    approved_only = bool(policy.get("approved_only", True))
    release_filter = str(policy.get("release", "")).strip()
    module_filters = [_normalize_text(m) for m in policy.get("modules", []) if str(m).strip()]
    include_legacy = bool(policy.get("include_legacy_unlabeled", True))

    if approved_only and (not _is_doc_approved(metadata, include_legacy)):
        return False

    if release_filter:
        doc_release = _extract_doc_release(metadata)
        if not doc_release or _normalize_text(doc_release) != _normalize_text(release_filter):
            return False

    if module_filters:
        doc_modules = [_normalize_text(m) for m in _extract_doc_modules(metadata)]
        if not doc_modules:
            return False
        if not any(m in doc_modules for m in module_filters):
            return False

    return True


def _normalize_source_type(metadata: Dict[str, Any]) -> str:
    source_type = str((metadata or {}).get("source_type", "unknown")).strip().lower()
    return source_type if source_type in SOURCE_TYPE_LABELS else "unknown"


def _doc_unique_key(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    doc_key = str(metadata.get("doc_key") or "").strip()
    chunk_index = metadata.get("chunk_index")
    if doc_key:
        return f"{doc_key}#{chunk_index}"

    source_name = str(metadata.get("source_name") or "unknown")
    page_content = str(getattr(doc, "page_content", "") or "")
    digest = hashlib.sha1(
        f"{source_name}|{page_content[:200]}".encode("utf-8", errors="ignore")
    ).hexdigest()
    return digest


def _similarity_search_with_optional_score(
    vectordb: Chroma, query: str, k: int
) -> List[Tuple[Any, float | None]]:
    try:
        result_with_score = vectordb.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in result_with_score]
    except Exception:
        docs = vectordb.similarity_search(query, k=k)
        return [(doc, None) for doc in docs]


def _collect_candidates(
    vectordb: Chroma, task_query: str, policy: Dict[str, Any]
) -> List[Dict[str, Any]]:
    queries = _build_retrieval_queries(task_query)
    if not queries:
        return []

    candidate_map: Dict[str, Dict[str, Any]] = {}
    for query_index, query_text in enumerate(queries):
        hits = _similarity_search_with_optional_score(
            vectordb=vectordb,
            query=query_text,
            k=RETRIEVAL_TOP_K_PER_QUERY,
        )
        for hit_rank, (doc, score) in enumerate(hits):
            metadata = getattr(doc, "metadata", {}) or {}
            if not _passes_runtime_filters(metadata, policy):
                continue

            unique_key = _doc_unique_key(doc)
            score_flag = 0 if isinstance(score, (float, int)) else 1
            score_value = float(score) if isinstance(score, (float, int)) else 10**9
            rank_key = (score_flag, score_value, query_index, hit_rank)

            candidate = {
                "unique_key": unique_key,
                "doc": doc,
                "source_type": _normalize_source_type(metadata),
                "rank_key": rank_key,
            }

            old = candidate_map.get(unique_key)
            if old is None or rank_key < old["rank_key"]:
                candidate_map[unique_key] = candidate

    return sorted(candidate_map.values(), key=lambda item: item["rank_key"])


def _allocate_source_quotas(candidates: List[Dict[str, Any]], target_docs: int) -> Dict[str, int]:
    available: Dict[str, int] = defaultdict(int)
    for item in candidates:
        available[item["source_type"]] += 1

    quotas = {source_type: 0 for source_type in SOURCE_TYPE_ORDER}
    remaining = target_docs

    # 先保证每个关键来源都至少命中 1 条，增强跨模块覆盖。
    for source_type in ["requirement", "testcase", "api_doc", "ui"]:
        if available.get(source_type, 0) > 0 and remaining > 0:
            quotas[source_type] += 1
            remaining -= 1

    # 再按业务权重补齐。
    weighted_cycle = [
        "requirement",
        "testcase",
        "api_doc",
        "ui",
        "requirement",
        "testcase",
        "unknown",
    ]
    while remaining > 0:
        progressed = False
        for source_type in weighted_cycle:
            if remaining <= 0:
                break
            if quotas.get(source_type, 0) < available.get(source_type, 0):
                quotas[source_type] = quotas.get(source_type, 0) + 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return quotas


def _select_balanced_candidates(
    candidates: List[Dict[str, Any]], target_docs: int
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[item["source_type"]].append(item)

    for source_type in grouped:
        grouped[source_type].sort(key=lambda item: item["rank_key"])

    quotas = _allocate_source_quotas(candidates, target_docs)
    selected: List[Dict[str, Any]] = []
    selected_keys = set()

    for source_type in SOURCE_TYPE_ORDER:
        limit = quotas.get(source_type, 0)
        if limit <= 0:
            continue
        for item in grouped.get(source_type, [])[:limit]:
            selected.append(item)
            selected_keys.add(item["unique_key"])

    # 若某些类型数据不足，则按全局排序补足。
    if len(selected) < target_docs:
        for item in candidates:
            if item["unique_key"] in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(item["unique_key"])
            if len(selected) >= target_docs:
                break

    return selected


def _build_source_manifest(candidates: List[Dict[str, Any]]) -> str:
    unique_items: List[str] = []
    seen = set()
    for item in candidates:
        metadata = getattr(item["doc"], "metadata", {}) or {}
        source_type = str(metadata.get("source_type", "unknown"))
        source_name = str(metadata.get("source_name", "unknown"))
        origin = str(metadata.get("origin", "unknown"))
        doc_key = str(metadata.get("doc_key", "")).strip()
        feature_tag = _format_meta_tag("feature", metadata.get("ext_feature_key"), max_items=1, max_len=40)
        trace_tag = _format_meta_tag("trace", metadata.get("ext_trace_refs"), max_items=2, max_len=40)
        doc_tag = f" | doc_key={doc_key}" if doc_key else ""
        line = f"- [{source_type}] {source_name} ({origin}){doc_tag}{feature_tag}{trace_tag}"
        if line not in seen:
            seen.add(line)
            unique_items.append(line)
        if len(unique_items) >= 20:
            break
    if not unique_items:
        return ""
    return "## 召回来源清单\n" + "\n".join(unique_items) + "\n\n"


def _build_context_from_candidates(candidates: List[Dict[str, Any]]) -> str:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_type[item["source_type"]].append(item)

    lines: List[str] = []
    current_len = 0

    for source_type in SOURCE_TYPE_ORDER:
        items = by_type.get(source_type, [])
        if not items:
            continue

        section_title = f"### {SOURCE_TYPE_LABELS.get(source_type, source_type)}"
        section_block = [section_title]

        for index, item in enumerate(items, start=1):
            doc = item["doc"]
            metadata = getattr(doc, "metadata", {}) or {}
            source_name = str(metadata.get("source_name", "unknown"))
            source_type = str(metadata.get("source_type", source_type or "unknown")).strip().lower() or "unknown"
            origin = str(metadata.get("origin", "unknown"))
            doc_key = str(metadata.get("doc_key", "")).strip()
            chunk_index_meta = metadata.get("chunk_index")
            if not isinstance(chunk_index_meta, int):
                chunk_index_meta = index
            modules = _extract_doc_modules(metadata)
            feature_tag = _format_meta_tag("feature", metadata.get("ext_feature_key"), max_items=1, max_len=40)
            trace_tag = _format_meta_tag("trace", metadata.get("ext_trace_refs"), max_items=2, max_len=40)
            domain_tag = _format_meta_tag("domain", metadata.get("ext_business_domain"), max_items=3, max_len=60)
            upstream_tag = _format_meta_tag(
                "upstream", metadata.get("ext_upstream_modules"), max_items=3, max_len=60
            )
            downstream_tag = _format_meta_tag(
                "downstream", metadata.get("ext_downstream_modules"), max_items=3, max_len=60
            )
            related_tag = _format_meta_tag(
                "related", metadata.get("ext_related_doc_keys"), max_items=2, max_len=60
            )
            page_content = str(getattr(doc, "page_content", "") or "").strip()
            page_content = re.sub(r"\n{3,}", "\n\n", page_content)
            if len(page_content) > 900:
                page_content = page_content[:900] + "\n...[片段截断]"

            header = (
                f"- 片段{index}"
                f" | doc_key={doc_key}"
                f" | chunk_index={chunk_index_meta}"
                f" | source_type={source_type}"
                f" | source={source_name}"
                f" | origin={origin}"
            )
            if modules:
                header += f" | modules={','.join(modules)}"
            header += (
                f"{domain_tag}{upstream_tag}{downstream_tag}{feature_tag}{trace_tag}{related_tag}"
            )

            chunk_text = f"{header}\n{page_content}\n"
            section_block.append(chunk_text)

        combined_section = "\n".join(section_block).strip() + "\n\n"
        if current_len + len(combined_section) > RETRIEVAL_MAX_CONTEXT_CHARS:
            remaining = RETRIEVAL_MAX_CONTEXT_CHARS - current_len
            if remaining > 100:
                lines.append(combined_section[: remaining - 20] + "\n...[上下文总量截断]")
            break

        lines.append(combined_section)
        current_len += len(combined_section)

    body = "".join(lines).strip()
    manifest = _build_source_manifest(candidates)
    return f"{manifest}{body}".strip()


# 3. 增强型检索逻辑（跨模块关联召回）
def get_augmented_context(task_query, retrieval_policy: Dict[str, Any] | None = None):
    vectordb = _get_cached_vectordb()
    policy = _get_runtime_retrieval_policy(retrieval_policy)
    candidates = _collect_candidates(vectordb=vectordb, task_query=task_query, policy=policy)
    if not candidates:
        docs = vectordb.similarity_search(task_query, k=RETRIEVAL_TOP_K_PER_QUERY)
        filtered_docs = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            if _passes_runtime_filters(metadata, policy):
                filtered_docs.append(doc)
        if not filtered_docs:
            if (
                policy.get("approved_only")
                or policy.get("release")
                or policy.get("modules")
            ):
                return ""
            filtered_docs = docs
        return "\n\n".join([doc.page_content for doc in filtered_docs])

    selected = _select_balanced_candidates(
        candidates=candidates,
        target_docs=RETRIEVAL_TARGET_DOCS,
    )
    context = _build_context_from_candidates(selected)
    if context.strip():
        return context

    return "\n\n".join(
        [str(getattr(item["doc"], "page_content", "") or "") for item in selected]
    )

# 4. 资深测试开发级 Prompt 模板 (已强化 Emoji 禁用指令)
TEST_GEN_TEMPLATE = """
你是一名资深测试开发工程师，擅长通过【业务逻辑接口测试】和【功能测试】保障票务系统稳定性。
请根据提供的【业务需求上下文】，针对【待测核心逻辑】产出两部分内容。

【输出要求 1：业务逻辑接口测试场景 (API Business Level)】
- 目标：辅助测试人员使用 Postman/Apifox 等工具验证后端业务逻辑。
- 展现形式：严禁输出任何代码或伪代码，必须使用结构化业务文本。
- 结构包含：【接口场景名称】、【前置数据状态】、【关键入参特征】、【预期业务结果】。
- 深度要求：预期业务结果必须明确联动影响（订单状态、库存状态、可退票状态、下游模块影响）。

【输出要求 2：全链路功能测试用例 (Functional Level)】
- 展现形式：**严禁输出任何形式的代码（如 Python、Playwright 等）**。必须以标准的业务测试用例格式输出。
- 结构要求：每个场景必须包含【用例名称】、【前置条件】、【测试步骤】（如：点击XX按钮，切换到XX语言）、【预期结果】。
- 业务细节：预期结果中必须明确写出校验的 UI 文案（如中、英、日文的具体展示内容）和业务状态。

【输出要求 3：工程化收尾清单 (Engineering Close-out)（可选）】
- 仅当需求明确要求工程化收尾或上线交付时输出。
- 若无明确要求，请直接省略该部分（不输出标题）。

---
【业务需求上下文】：
{context}

【待测核心逻辑】：
{task}

【输出格式约束】：
1. 保持业务视角，逻辑清晰，拒绝冗长客套话。
2. **严禁输出任何代码、伪代码、脚本片段或代码块标记（```）**。
3. **严禁使用任何 Emoji 图标**。
4. 必须结合上下文中的业务实体、规则、角色、状态与文案；若关键信息不足，先输出“缺失信息清单”再给出合理假设。
5. 直接输出 Markdown 格式。
"""

# 兼容 Web 包装层：提供统一模板命名
UNIVERSAL_TEMPLATE = TEST_GEN_TEMPLATE

# 自动修补 URL 路径，避免调用方重复处理
if BASE_URL and not BASE_URL.endswith('/v1'):
    BASE_URL = BASE_URL.rstrip('/') + '/v1'

# 模块级 LLM 实例，供 Streamlit 直接复用
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
http_client = httpx.Client(
    proxy=None,
    trust_env=False,
    verify=False,
    timeout=60.0
)
llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0.1,
    http_client=http_client
)

def generate_test_case():
    global BASE_URL
    task_desc = """
    请针对【支付时间限制功能】生成全链路功能测试用例，必须严格覆盖以下核心维度：
    1. 商户端配置：支付时间限制开关的默认值、历史项目处理方式及下游修改权限。
    2. 联动逻辑：开关开启后，额外出现的配置项及其默认值逻辑。
    3. 多语言支持：C端对于英文和日文环境下的支付时间限制文案映射规则。
    4. C端展示范围：支付时间限制规则在C端包含的具体页面或模块（如列表、详情、下单页等）。
    5. 核心算法判定：支付时间逻辑的判定标准，之前或之后如何严格定义，以什么时间戳为准。
    """
    
    print("正在从 Chroma 数据库提取业务上下文...")
    try:
        context = get_augmented_context(task_desc)
        print(f"[INFO] 检索成功，当前召回上下文长度: {len(context)} 字符")
        if len(context) < 300:
            print("[WARN] 检索内容过少，建议重新运行数据同步脚本。")
    except Exception as e:
        print(f"[ERROR] 数据库检索异常: {e}")
        return

    # 网络连通性快速检查
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        print("网络物理连通性: OK")
    except:
        print("网络物理连通性: FAIL")

    # 核心网络补丁
    os.environ['NO_PROXY'] = '*'
    os.environ['no_proxy'] = '*'
    
    # 5. 初始化 LLM 驱动（复用模块级 llm）
    prompt = PromptTemplate(
        template=TEST_GEN_TEMPLATE, 
        input_variables=["context", "task"]
    )
    
    final_prompt = prompt.format(context=context, task=task_desc)
    
    print("正在调用 LLM 进行全链路测试用例建模...")
    try:
        response = llm.invoke(final_prompt)
        print("\n" + "="*20 + " 工业级测试用例产出 " + "="*20 + "\n")
        print(response.content)
        
        # 将产出结果保存到本地文件
        output_file = PROJECT_ROOT / "data" / "output" / "generated_testcase.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            f.write(response.content)
        print(f"\n结果已同步保存至: {output_file}")
        
    except Exception as e:
        print(f"[ERROR] LLM 调度异常: {e}")
        print(f"DEBUG: BASE_URL={BASE_URL}")

if __name__ == "__main__":
    generate_test_case()
