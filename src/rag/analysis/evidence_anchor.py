from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

SOURCE_BASE_CONFIDENCE: Dict[str, float] = {
    "requirement": 0.72,
    "api_doc": 0.68,
    "testcase": 0.62,
    "ui": 0.58,
    "unknown": 0.45,
}

SOURCE_PRIORITY: Dict[str, int] = {
    "requirement": 0,
    "api_doc": 1,
    "testcase": 2,
    "ui": 3,
    "unknown": 9,
}


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _split_values(raw: Any) -> List[str]:
    if raw is None:
        return []
    values: List[str] = []
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        # 不拆分 '/'，避免把“库存/权益”“通知/消息”拆成两个模块别名。
        for token in re.split(r"[,\n，、;；|]+", str(item)):
            text = token.strip()
            if text and text not in values:
                values.append(text)
    return values


def _split_doc_keys(raw: Any) -> List[str]:
    if raw is None:
        return []
    values: List[str] = []
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        for token in re.split(r"[,\n，、;；]+", str(item)):
            text = token.strip()
            if text and text not in values:
                values.append(text)
    return values


def _safe_int(raw: Any) -> Optional[int]:
    try:
        return int(raw)
    except Exception:
        return None


def _parse_key_values(line: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for token in str(line or "").split(" | "):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = str(key).strip()
        value = str(value).strip()
        if key and value:
            parsed[key] = value
    return parsed


def _extract_snippet(lines: List[str], start: int) -> str:
    chunks: List[str] = []
    for line in lines[start + 1 :]:
        text = str(line).strip()
        if text.startswith("- 片段") or text.startswith("### ") or text.startswith("## "):
            break
        if not text:
            continue
        chunks.append(text)
        if len(" ".join(chunks)) >= 160:
            break
    snippet = " ".join(chunks).strip()
    return snippet[:160]


def _normalize_source_type(raw: Any) -> str:
    text = str(raw or "unknown").strip().lower()
    return text if text in SOURCE_BASE_CONFIDENCE else "unknown"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _build_anchor_from_line(
    line: str,
    snippet: str,
    section_source_type: str,
) -> Optional[Dict[str, Any]]:
    values = _parse_key_values(line)
    doc_key = str(values.get("doc_key", "")).strip()
    chunk_index = _safe_int(values.get("chunk_index"))

    # 兼容旧格式：从片段标题中提取序号作为降级值。
    if chunk_index is None:
        m = re.search(r"片段(\d+)", line or "")
        if m:
            chunk_index = _safe_int(m.group(1))

    source_type = _normalize_source_type(values.get("source_type") or section_source_type)
    source_name = str(values.get("source", "")).strip() or "unknown"
    origin = str(values.get("origin", "")).strip() or "unknown"

    if not doc_key or chunk_index is None or not source_name:
        return None

    feature_key = str(values.get("feature", "")).strip()
    trace_refs = _split_values(values.get("trace"))

    anchor = {
        "anchor_id": f"{doc_key}#{chunk_index}",
        "doc_key": doc_key,
        "chunk_index": chunk_index,
        "source_type": source_type,
        "source_name": source_name,
        "origin": origin,
        "feature_key": feature_key,
        "trace_refs": trace_refs,
        "module_tags": _split_values(values.get("modules")),
        "domain_tags": _split_values(values.get("domain")),
        "upstream_modules": _split_values(values.get("upstream")),
        "downstream_modules": _split_values(values.get("downstream")),
        "related_doc_keys": _split_doc_keys(values.get("related")),
        "snippet": snippet,
        "source_confidence": 0.0,
    }
    return anchor


def _apply_source_confidence(anchors: List[Dict[str, Any]]) -> None:
    feature_counts: Dict[str, int] = {}
    trace_counts: Dict[str, int] = {}

    for anchor in anchors:
        feature_key = str(anchor.get("feature_key") or "").strip()
        if feature_key:
            feature_counts[feature_key] = feature_counts.get(feature_key, 0) + 1
        for trace in anchor.get("trace_refs", []) or []:
            trace_text = str(trace).strip()
            if trace_text:
                trace_counts[trace_text] = trace_counts.get(trace_text, 0) + 1

    for anchor in anchors:
        source_type = _normalize_source_type(anchor.get("source_type"))
        base = SOURCE_BASE_CONFIDENCE.get(source_type, SOURCE_BASE_CONFIDENCE["unknown"])

        chunk_index = _safe_int(anchor.get("chunk_index"))
        if chunk_index is None:
            chunk_bonus = 0.0
        elif chunk_index <= 2:
            chunk_bonus = 0.06
        elif chunk_index <= 5:
            chunk_bonus = 0.03
        else:
            chunk_bonus = 0.0

        feature_key = str(anchor.get("feature_key") or "").strip()
        trace_refs = [str(x).strip() for x in anchor.get("trace_refs", []) or [] if str(x).strip()]

        consistency_bonus = 0.0
        if feature_key:
            consistency_bonus += 0.04
        if trace_refs:
            consistency_bonus += 0.04

        has_cluster_support = False
        if feature_key and feature_counts.get(feature_key, 0) >= 2:
            has_cluster_support = True
        if not has_cluster_support:
            for trace in trace_refs:
                if trace_counts.get(trace, 0) >= 2:
                    has_cluster_support = True
                    break
        if has_cluster_support:
            consistency_bonus += 0.06

        score = _clamp(base + chunk_bonus + consistency_bonus)
        if source_type == "unknown" and not feature_key and not trace_refs:
            score = min(score, 0.50)

        anchor["source_confidence"] = round(score, 4)


def build_evidence_anchors(retrieval_context: str) -> List[Dict[str, Any]]:
    lines = str(retrieval_context or "").splitlines()
    anchors: List[Dict[str, Any]] = []
    seen = set()
    section_source_type = "unknown"

    for idx, line in enumerate(lines):
        text = str(line).strip()
        if not text:
            continue

        if text.startswith("### "):
            title = _normalize_text(text[4:])
            if "需求" in title:
                section_source_type = "requirement"
            elif "接口" in title:
                section_source_type = "api_doc"
            elif "用例" in title:
                section_source_type = "testcase"
            elif "页面" in title or "ui" in title:
                section_source_type = "ui"
            else:
                section_source_type = "unknown"
            continue

        if not text.startswith("- 片段"):
            continue

        snippet = _extract_snippet(lines, idx)
        anchor = _build_anchor_from_line(text, snippet, section_source_type)
        if not anchor:
            continue

        anchor_id = str(anchor.get("anchor_id") or "").strip()
        if not anchor_id or anchor_id in seen:
            continue
        seen.add(anchor_id)
        anchors.append(anchor)

    _apply_source_confidence(anchors)
    return anchors


def anchor_has_valid_locator(anchor: Dict[str, Any]) -> bool:
    anchor_id = str(anchor.get("anchor_id") or "").strip()
    doc_key = str(anchor.get("doc_key") or "").strip()
    chunk_index = _safe_int(anchor.get("chunk_index"))
    return bool(anchor_id and doc_key and chunk_index is not None)


def _anchor_matches_alias(anchor: Dict[str, Any], aliases: List[str]) -> bool:
    alias_norm = [_normalize_text(x) for x in aliases if str(x).strip()]
    if not alias_norm:
        return False

    fields: List[str] = []
    for key in (
        "module_tags",
        "domain_tags",
        "upstream_modules",
        "downstream_modules",
    ):
        values = anchor.get(key, [])
        if isinstance(values, list):
            fields.extend([str(x) for x in values])

    fields.append(str(anchor.get("snippet") or ""))

    haystack = "\n".join(fields).lower()
    for alias in alias_norm:
        if alias and alias in haystack:
            return True
    return False


def _anchor_structured_match(anchor: Dict[str, Any], aliases: List[str]) -> bool:
    alias_norm = [_normalize_text(x) for x in aliases if str(x).strip()]
    if not alias_norm:
        return False

    values: List[str] = []
    for key in ("module_tags", "domain_tags", "upstream_modules", "downstream_modules"):
        raw = anchor.get(key, [])
        if isinstance(raw, list):
            values.extend([str(x) for x in raw])
    haystack = "\n".join(values).lower()

    for alias in alias_norm:
        if alias and alias in haystack:
            return True
    return False


def select_best_anchor(
    anchors: List[Dict[str, Any]],
    module_aliases: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    candidates = [a for a in anchors if anchor_has_valid_locator(a)]
    if module_aliases:
        candidates = [a for a in candidates if _anchor_matches_alias(a, module_aliases)]
    if not candidates:
        return None

    def sort_key(anchor: Dict[str, Any]) -> Any:
        source_type = _normalize_source_type(anchor.get("source_type"))
        priority = SOURCE_PRIORITY.get(source_type, SOURCE_PRIORITY["unknown"])
        source_conf = float(anchor.get("source_confidence") or 0.0)
        chunk_index = _safe_int(anchor.get("chunk_index"))
        if chunk_index is None:
            chunk_index = 9999
        structured = _anchor_structured_match(anchor, module_aliases or [])
        return (0 if structured else 1, priority, -source_conf, chunk_index)

    candidates.sort(key=sort_key)
    return candidates[0]
