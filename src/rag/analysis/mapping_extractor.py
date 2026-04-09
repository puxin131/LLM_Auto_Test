from __future__ import annotations

import re
from typing import Any, Dict, List

from .evidence_anchor import anchor_has_valid_locator, build_evidence_anchors

_MAPPING_SEPARATORS = ["->", "=>", "映射到", "对应", "转换为", "translate to"]
_DEFAULT_HINTS = ["默认", "default", "兜底", "fallback"]
_ENUM_HINTS = ["枚举", "状态", "status", "type", "code", "标记"]


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


def _extract_pairs_from_text(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for raw_line in str(text or "").splitlines():
        line = _normalize(raw_line)
        if not line:
            continue
        if len(line) > 220:
            line = line[:220]

        # Prefer strict token-to-token capture first.
        regex_pairs = re.findall(
            r"([A-Za-z_][A-Za-z0-9_.-]{0,40})\s*(?:->|=>)\s*([A-Za-z_][A-Za-z0-9_.-]{0,40})",
            line,
        )
        if regex_pairs:
            for src, dst in regex_pairs:
                key = (_norm_lower(src), _norm_lower(dst))
                if key in seen:
                    continue
                seen.add(key)
                transform = "field_mapping"
                if _contains_any(line, _ENUM_HINTS):
                    transform = "enum_or_status_translate"
                elif _contains_any(line, ["时间", "timestamp", "时区"]):
                    transform = "time_normalize"
                elif _contains_any(line, ["金额", "price", "amount"]):
                    transform = "amount_normalize"
                default_rule = "has_default_fallback" if _contains_any(line, _DEFAULT_HINTS) else ""
                rows.append(
                    {
                        "source_field": src[:60],
                        "target_field": dst[:60],
                        "transform_rule": transform,
                        "default_rule": default_rule,
                    }
                )
            continue

        pair: Dict[str, str] | None = None
        for sep in _MAPPING_SEPARATORS:
            if sep in line:
                left, right = line.split(sep, 1)
                src = left.strip(" -:：，,。")
                dst = right.strip(" -:：，,。")
                if src and dst and src != dst:
                    pair = {"source_field": src[:60], "target_field": dst[:60]}
                    break
        if pair:
            key = (_norm_lower(pair["source_field"]), _norm_lower(pair["target_field"]))
            if key in seen:
                continue
            seen.add(key)
            transform = ""
            if _contains_any(line, _ENUM_HINTS):
                transform = "enum_or_status_translate"
            elif _contains_any(line, ["时间", "timestamp", "时区"]):
                transform = "time_normalize"
            elif _contains_any(line, ["金额", "price", "amount"]):
                transform = "amount_normalize"
            else:
                transform = "field_mapping"

            default_rule = ""
            if _contains_any(line, _DEFAULT_HINTS):
                default_rule = "has_default_fallback"

            rows.append(
                {
                    "source_field": pair["source_field"],
                    "target_field": pair["target_field"],
                    "transform_rule": transform,
                    "default_rule": default_rule,
                }
            )
    return rows


def build_mapping_rules(
    *,
    task_query: str,
    retrieval_context: str,
    anchors: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Extract lightweight mapping rules for internal<->third-party integration tests.
    """
    anchors = anchors if isinstance(anchors, list) else build_evidence_anchors(retrieval_context)
    valid_anchors = [a for a in anchors if isinstance(a, dict) and anchor_has_valid_locator(a)]

    rules: List[Dict[str, Any]] = []
    seen = set()
    for anchor in valid_anchors:
        snippet = str(anchor.get("snippet", "") or "")
        for rule in _extract_pairs_from_text(snippet):
            key = (
                _norm_lower(rule.get("source_field")),
                _norm_lower(rule.get("target_field")),
                _norm_lower(rule.get("transform_rule")),
            )
            if key in seen:
                continue
            seen.add(key)
            rules.append(
                {
                    **rule,
                    "rule_key": f"{rule['source_field']}->{rule['target_field']}",
                    "evidence_anchor": anchor,
                    "evidence_anchor_id": str(anchor.get("anchor_id", "")).strip(),
                }
            )

    # Task-level fallback: ensure minimum mapping hints for weak contexts.
    if not rules:
        task_rules = _extract_pairs_from_text(task_query)
        for rule in task_rules[:4]:
            key = (
                _norm_lower(rule.get("source_field")),
                _norm_lower(rule.get("target_field")),
                _norm_lower(rule.get("transform_rule")),
            )
            if key in seen:
                continue
            seen.add(key)
            rules.append(
                {
                    **rule,
                    "rule_key": f"{rule['source_field']}->{rule['target_field']}",
                    "evidence_anchor": {},
                    "evidence_anchor_id": "",
                }
            )

    summary = (
        f"mapping_rules={len(rules)}"
        + (", with_anchor" if any(r.get("evidence_anchor_id") for r in rules) else ", no_anchor")
    )
    return {"mapping_rules": rules[:20], "mapping_summary": summary}
