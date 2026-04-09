from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .evidence_anchor import anchor_has_valid_locator, build_evidence_anchors

_HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")

_FIELD_HINTS = [
    "sku",
    "spu",
    "product_id",
    "item_id",
    "stock",
    "inventory",
    "qty",
    "quantity",
    "order_id",
    "status",
    "price",
    "trace_id",
    "tenant_id",
    "timestamp",
    "version",
]

_CODE_HINTS = ["200", "400", "401", "403", "404", "409", "429", "500", "502", "503", "504"]

_EXTERNAL_HINTS = [
    "三方",
    "third",
    "erp",
    "external",
    "供应商",
    "外部",
    "partner",
]

_IDEMPOTENCY_HINTS = ["幂等", "重复请求", "去重", "idempotent", "重放"]
_TIMING_HINTS = ["超时", "重试", "回调", "异步", "最终一致", "补偿", "rollback", "timeout"]


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _norm_lower(value: Any) -> str:
    return _normalize(value).lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = _norm_lower(text)
    for keyword in keywords:
        key = _norm_lower(keyword)
        if key and key in lowered:
            return True
    return False


def _extract_interface_signals(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out

    # HTTP method + path
    for method in _HTTP_METHODS:
        pattern = rf"\b{method}\s+(/[A-Za-z0-9_\-/{{}}]+)"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = f"{method.upper()} {match.group(1)}"
            if value not in out:
                out.append(value)

    # path only
    for match in re.finditer(r"(/[A-Za-z0-9_\-/]{4,})", text):
        path = match.group(1)
        if path.count("/") >= 2 and all(x not in path.lower() for x in ("http://", "https://")):
            value = f"PATH {path}"
            if value not in out:
                out.append(value)
    return out[:20]


def _extract_field_signals(text: str) -> List[str]:
    lowered = _norm_lower(text)
    out: List[str] = []
    for hint in _FIELD_HINTS:
        if hint in lowered and hint not in out:
            out.append(hint)
    return out[:20]


def _extract_status_codes(text: str) -> List[str]:
    out: List[str] = []
    for code in _CODE_HINTS:
        if re.search(rf"\b{re.escape(code)}\b", text):
            out.append(code)
    return out


def _anchor_source_bucket(anchor: Dict[str, Any]) -> str:
    source_name = _norm_lower(anchor.get("source_name"))
    origin = _norm_lower(anchor.get("origin"))
    doc_key = _norm_lower(anchor.get("doc_key"))
    snippet = _norm_lower(anchor.get("snippet"))
    merged = " ".join([source_name, origin, doc_key, snippet])
    if _contains_any(merged, _EXTERNAL_HINTS):
        return "external"
    return "internal"


def _build_contract_from_anchors(anchors: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not anchors:
        return {
            "interfaces": [],
            "key_fields": [],
            "status_codes": [],
            "idempotency_signals": [],
            "timing_signals": [],
            "evidence_anchor_ids": [],
        }

    text_blob = "\n".join(str(anchor.get("snippet", "") or "") for anchor in anchors)
    interfaces = _extract_interface_signals(text_blob)
    key_fields = _extract_field_signals(text_blob)
    status_codes = _extract_status_codes(text_blob)

    idempotency_signals = [
        hint for hint in _IDEMPOTENCY_HINTS if _contains_any(text_blob, [hint])
    ][:6]
    timing_signals = [hint for hint in _TIMING_HINTS if _contains_any(text_blob, [hint])][:8]

    evidence_anchor_ids: List[str] = []
    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id") or "").strip()
        if anchor_id and anchor_id not in evidence_anchor_ids:
            evidence_anchor_ids.append(anchor_id)
    return {
        "interfaces": interfaces,
        "key_fields": key_fields,
        "status_codes": status_codes,
        "idempotency_signals": idempotency_signals,
        "timing_signals": timing_signals,
        "evidence_anchor_ids": evidence_anchor_ids[:8],
    }


def build_dual_contracts(
    *,
    task_query: str,
    retrieval_context: str,
    anchors: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Build internal/external contract summaries used by generation prompt and UI review.
    """
    anchors = anchors if isinstance(anchors, list) else build_evidence_anchors(retrieval_context)
    valid_anchors = [a for a in anchors if isinstance(a, dict) and anchor_has_valid_locator(a)]
    internal_anchors: List[Dict[str, Any]] = []
    external_anchors: List[Dict[str, Any]] = []
    for anchor in valid_anchors:
        bucket = _anchor_source_bucket(anchor)
        if bucket == "external":
            external_anchors.append(anchor)
        else:
            internal_anchors.append(anchor)

    internal_contract = _build_contract_from_anchors(internal_anchors)
    external_contract = _build_contract_from_anchors(external_anchors)

    # Augment with task-level hints so weak retrieval can still keep contract intent.
    task_text = _normalize(task_query)
    task_interfaces = _extract_interface_signals(task_text)
    task_fields = _extract_field_signals(task_text)
    task_codes = _extract_status_codes(task_text)

    for value in task_interfaces:
        if value not in internal_contract["interfaces"]:
            internal_contract["interfaces"].append(value)
    for value in task_fields:
        if value not in internal_contract["key_fields"]:
            internal_contract["key_fields"].append(value)
    for value in task_codes:
        if value not in internal_contract["status_codes"]:
            internal_contract["status_codes"].append(value)

    summary = (
        f"internal={len(internal_contract['interfaces'])}接口/"
        f"{len(internal_contract['key_fields'])}字段, "
        f"external={len(external_contract['interfaces'])}接口/"
        f"{len(external_contract['key_fields'])}字段"
    )
    return {
        "internal_contract": internal_contract,
        "external_contract": external_contract,
        "contract_summary": summary,
    }

