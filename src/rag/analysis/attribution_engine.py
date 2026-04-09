from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from .evidence_anchor import anchor_has_valid_locator, select_best_anchor

MODULE_CATALOG: List[Dict[str, Any]] = [
    {"module": "订单", "aliases": ["订单", "下单", "手动下单", "订单状态", "订单流转", "出票"]},
    {"module": "支付", "aliases": ["支付", "付款", "扣款", "交易", "收款", "支付确认"]},
    {"module": "退款", "aliases": ["退款", "退费", "退货", "回退资金", "退款完成"]},
    {"module": "库存/权益", "aliases": ["库存", "权益", "配额", "扣减", "回补", "占用", "库存锁定"]},
    {"module": "用户/权限", "aliases": ["用户", "权限", "角色", "鉴权", "rbac", "登录", "租户"]},
    {"module": "通知/消息", "aliases": ["通知", "消息", "短信", "邮件", "推送", "消息发送"]},
    {"module": "风控/安全", "aliases": ["风控", "风险", "安全", "黑名单", "拦截"]},
    {"module": "优惠券/营销", "aliases": ["优惠券", "营销", "活动", "折扣", "满减"]},
    {"module": "结算/账单", "aliases": ["结算", "账单", "对账", "发票", "入账"]},
]


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _dedupe(items: Iterable[str]) -> List[str]:
    values: List[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def _iter_aliases(module_item: Dict[str, Any]) -> List[str]:
    module = str(module_item.get("module") or "").strip()
    aliases = module_item.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = [str(aliases)]
    return _dedupe([module, *[str(x).strip() for x in aliases if str(x).strip()]])


def _catalog_alias_entries() -> List[Tuple[str, str, str]]:
    # (canonical_module, alias, normalized_alias)
    entries: List[Tuple[str, str, str]] = []
    for item in MODULE_CATALOG:
        canonical = str(item.get("module") or "").strip()
        for alias in _iter_aliases(item):
            norm = _normalize_text(alias)
            if canonical and alias and norm:
                entries.append((canonical, alias, norm))
    entries.sort(key=lambda x: len(x[2]), reverse=True)
    return entries


_ALIAS_ENTRIES = _catalog_alias_entries()


def canonicalize_module(raw: Any, extra_candidates: Iterable[str] | None = None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    best_match: Tuple[str, int] | None = None
    for canonical, _, alias_norm in _ALIAS_ENTRIES:
        if normalized == alias_norm or alias_norm in normalized or normalized in alias_norm:
            score = len(alias_norm)
            if best_match is None or score > best_match[1]:
                best_match = (canonical, score)

    if best_match:
        return best_match[0]

    for candidate in extra_candidates or []:
        candidate_text = str(candidate).strip()
        candidate_norm = _normalize_text(candidate_text)
        if not candidate_norm:
            continue
        if normalized == candidate_norm or candidate_norm in normalized or normalized in candidate_norm:
            return candidate_text

    return text


def module_aliases_for(module_name: str) -> List[str]:
    canonical = canonicalize_module(module_name)
    target = _normalize_text(canonical or module_name)
    for item in MODULE_CATALOG:
        if _normalize_text(item.get("module")) == target:
            return _iter_aliases(item)
    return _dedupe([str(module_name).strip()])


def resolve_module_name(raw: Any, extra_candidates: Iterable[str] | None = None) -> str:
    return canonicalize_module(raw, extra_candidates=extra_candidates)


def _contains_any(text: str, aliases: List[str]) -> List[str]:
    normalized_text = _normalize_text(text)
    hits: List[str] = []
    for alias in aliases:
        alias_text = str(alias).strip()
        alias_norm = _normalize_text(alias_text)
        if alias_norm and alias_norm in normalized_text and alias_text not in hits:
            hits.append(alias_text)
    return hits


def _first_hit_position(text: str, aliases: List[str]) -> int:
    normalized_text = _normalize_text(text)
    pos = 10**9
    for alias in aliases:
        alias_norm = _normalize_text(alias)
        if not alias_norm:
            continue
        idx = normalized_text.find(alias_norm)
        if idx >= 0 and idx < pos:
            pos = idx
    return pos


def _anchor_text(anchor: Dict[str, Any]) -> str:
    fields: List[str] = []
    for key in (
        "module_tags",
        "domain_tags",
        "upstream_modules",
        "downstream_modules",
        "snippet",
        "source_name",
    ):
        raw = anchor.get(key)
        if isinstance(raw, list):
            fields.extend([str(x) for x in raw])
        elif raw is not None:
            fields.append(str(raw))
    return _normalize_text(" ".join(fields))


def _canonicalize_anchor_modules(anchor: Dict[str, Any]) -> Dict[str, List[str]]:
    canonical: Dict[str, List[str]] = {}
    for key in ("module_tags", "domain_tags", "upstream_modules", "downstream_modules"):
        values = anchor.get(key, [])
        if not isinstance(values, list):
            values = [values]
        canonical[key] = _dedupe(
            [
                canonicalize_module(value)
                for value in values
                if str(value).strip() and canonicalize_module(value)
            ]
        )
    return canonical


def _anchor_structured_hit(anchor: Dict[str, Any], module_name: str, aliases: List[str]) -> bool:
    canonical_fields = _canonicalize_anchor_modules(anchor)
    canonical_module = canonicalize_module(module_name)
    alias_norm = {_normalize_text(a) for a in aliases if str(a).strip()}
    if canonical_module:
        alias_norm.add(_normalize_text(canonical_module))

    for key in ("module_tags", "domain_tags", "upstream_modules", "downstream_modules"):
        for value in canonical_fields.get(key, []):
            norm = _normalize_text(value)
            if norm in alias_norm:
                return True
    return False


def _anchor_any_hit(anchor: Dict[str, Any], aliases: List[str]) -> bool:
    haystack = _anchor_text(anchor)
    for alias in aliases:
        key = _normalize_text(alias)
        if key and key in haystack:
            return True
    return False


def _build_candidate_modules() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in MODULE_CATALOG:
        module = str(item.get("module") or "").strip()
        if not module:
            continue
        candidates.append({"module": module, "aliases": _iter_aliases(item)})
    return candidates


def build_current_involved_modules(
    *,
    task_query: str,
    anchors: List[Dict[str, Any]],
    max_modules: int = 2,
) -> List[Dict[str, Any]]:
    task_text = str(task_query or "")
    normalized_task = _normalize_text(task_text)
    if not normalized_task or not anchors:
        return []

    feature_counts: Dict[str, int] = {}
    trace_counts: Dict[str, int] = {}
    for anchor in anchors:
        feature = str(anchor.get("feature_key") or "").strip()
        if feature:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
        for trace in anchor.get("trace_refs", []) or []:
            trace_text = str(trace).strip()
            if trace_text:
                trace_counts[trace_text] = trace_counts.get(trace_text, 0) + 1

    candidates = _build_candidate_modules()
    total_anchors = max(1, len(anchors))
    results: List[Dict[str, Any]] = []

    for candidate in candidates:
        module = str(candidate["module"])
        aliases = [str(x).strip() for x in candidate.get("aliases", []) if str(x).strip()]
        if not aliases:
            continue

        # 硬门槛：必须有需求直证，辅助信号不可把 0 拉起来。
        direct_hits = _contains_any(task_text, aliases)
        if not direct_hits:
            continue

        first_pos = _first_hit_position(task_text, aliases)

        module_anchors = [a for a in anchors if _anchor_any_hit(a, aliases)]
        if not module_anchors:
            continue

        structured_hits = [a for a in module_anchors if _anchor_structured_hit(a, module, aliases)]
        non_unknown_hits = [a for a in module_anchors if str(a.get("source_type") or "unknown") != "unknown"]

        has_relation_support = any(
            bool(a.get("feature_key"))
            or bool(a.get("trace_refs"))
            or bool(a.get("related_doc_keys"))
            or bool(a.get("upstream_modules"))
            or bool(a.get("downstream_modules"))
            for a in module_anchors
        )

        # 仅 unknown 来源且无结构化关系支撑，淘汰。
        if not non_unknown_hits and not has_relation_support:
            continue

        # 主依据: 需求直证 + 命中位置
        direct_score = min(1.0, 0.72 + 0.12 * max(0, len(direct_hits) - 1))
        if first_pos <= 6:
            position_score = 1.0
        elif first_pos <= 16:
            position_score = 0.85
        else:
            position_score = 0.70

        # 辅助信号: 仅用于微调，避免喧宾夺主。
        distribution_score = min(1.0, (len(module_anchors) / total_anchors) * 2.0)
        metadata_score = min(1.0, len(structured_hits) / max(1, len(module_anchors)))

        cluster_support = False
        for anchor in module_anchors:
            feature_key = str(anchor.get("feature_key") or "").strip()
            if feature_key and feature_counts.get(feature_key, 0) >= 2:
                cluster_support = True
                break
            for trace in anchor.get("trace_refs", []) or []:
                trace_text = str(trace).strip()
                if trace_text and trace_counts.get(trace_text, 0) >= 2:
                    cluster_support = True
                    break
            if cluster_support:
                break

        cluster_score = 1.0 if cluster_support else 0.0
        related_doc_score = 1.0 if any(a.get("related_doc_keys") for a in module_anchors) else 0.0

        final_score = (
            0.60 * direct_score
            + 0.20 * position_score
            + 0.08 * distribution_score
            + 0.05 * metadata_score
            + 0.04 * cluster_score
            + 0.03 * related_doc_score
        )

        if final_score < 0.62:
            continue

        top_anchor = select_best_anchor(module_anchors, module_aliases=aliases)
        if not top_anchor or not anchor_has_valid_locator(top_anchor):
            continue

        direct_term = direct_hits[0]
        source_type = str(top_anchor.get("source_type") or "unknown")
        source_name = str(top_anchor.get("source_name") or "unknown")
        chunk_index = top_anchor.get("chunk_index")
        top_evidence = f'需求直证:"{direct_term}" + 锚点[{source_type}] {source_name}#片段{chunk_index}'

        results.append(
            {
                "module": module,
                "confidence": round(final_score, 4),
                "top_evidence": top_evidence,
                "evidence_anchor": top_anchor,
                "signal_breakdown": {
                    "direct_evidence": round(direct_score, 4),
                    "position_priority": round(position_score, 4),
                    "retrieval_distribution": round(distribution_score, 4),
                    "metadata_alignment": round(metadata_score, 4),
                    "relation_cluster": round(cluster_score, 4),
                    "related_doc_support": round(related_doc_score, 4),
                    "final_score": round(final_score, 4),
                },
            }
        )

    results.sort(
        key=lambda x: (
            -float(x.get("confidence") or 0.0),
            _first_hit_position(task_text, module_aliases_for(str(x.get("module") or ""))),
            str(x.get("module") or ""),
        )
    )

    output: List[Dict[str, Any]] = []
    seen_modules = set()
    for item in results:
        module = canonicalize_module(item.get("module"))
        if not module or module in seen_modules:
            continue
        item["module"] = module
        seen_modules.add(module)
        output.append(item)
        if len(output) >= max(1, int(max_modules)):
            break

    return output
