from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .attribution_engine import canonicalize_module, module_aliases_for, resolve_module_name
from .evidence_anchor import SOURCE_PRIORITY, anchor_has_valid_locator

LINKAGE_TERMS = [
    "联动",
    "影响",
    "依赖",
    "一致性",
    "同步",
    "上下游",
    "触发",
    "回滚",
    "级联",
    "关联",
]

_EDGE_TYPE_PRIORITY = {
    "downstream": 0,
    "related_doc": 1,
    "shared_trace": 2,
    "shared_feature": 3,
    "upstream": 4,
    "domain_cooccur": 9,
}

_EDGE_BASE_STRENGTH = {
    "downstream": 1.0,
    "related_doc": 0.76,
    "shared_trace": 0.70,
    "shared_feature": 0.66,
    "upstream": 0.52,
    "domain_cooccur": 0.30,
}


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _contains_any(text: str, terms: List[str]) -> bool:
    target = _normalize_text(text)
    for term in terms:
        key = _normalize_text(term)
        if key and key in target:
            return True
    return False


def _canonical_list(raw_values: Any, extra_candidates: Set[str]) -> List[str]:
    if isinstance(raw_values, list):
        values = raw_values
    else:
        values = [raw_values]

    modules: List[str] = []
    for value in values:
        resolved = resolve_module_name(value, extra_candidates)
        canonical = canonicalize_module(resolved, extra_candidates)
        if canonical and canonical not in modules:
            modules.append(canonical)
    return modules


def _anchor_module_hit(anchor: Dict[str, Any], module_name: str) -> bool:
    aliases = module_aliases_for(module_name)
    canonical = canonicalize_module(module_name)

    haystack_parts: List[str] = []
    for key in (
        "module_tags",
        "domain_tags",
        "upstream_modules",
        "downstream_modules",
        "snippet",
    ):
        raw = anchor.get(key)
        if isinstance(raw, list):
            haystack_parts.extend([str(x) for x in raw])
        elif raw is not None:
            haystack_parts.append(str(raw))

    haystack = _normalize_text(" ".join(haystack_parts))
    for alias in aliases + ([canonical] if canonical else []):
        key = _normalize_text(alias)
        if key and key in haystack:
            return True
    return False


def _anchor_quality(anchor: Dict[str, Any]) -> float:
    score = 0.0
    if anchor.get("module_tags") or anchor.get("domain_tags"):
        score += 0.40
    if anchor.get("feature_key") or anchor.get("trace_refs"):
        score += 0.25
    if anchor.get("related_doc_keys") or anchor.get("upstream_modules") or anchor.get("downstream_modules"):
        score += 0.20
    source_type = str(anchor.get("source_type") or "unknown")
    if source_type in {"requirement", "api_doc"}:
        score += 0.15
    return min(1.0, score)


def _best_edge_type(edge_types: Set[str]) -> str:
    if not edge_types:
        return "mixed"
    ordered = sorted(edge_types, key=lambda x: _EDGE_TYPE_PRIORITY.get(x, 99))
    return ordered[0] if len(ordered) == 1 else "mixed"


def _build_relation_consumed(anchors: List[Dict[str, Any]]) -> Dict[str, bool]:
    return {
        "feature": any(bool(str(a.get("feature_key") or "").strip()) for a in anchors),
        "trace": any(bool(a.get("trace_refs")) for a in anchors),
        "domain": any(bool(a.get("domain_tags")) for a in anchors),
        "upstream": any(bool(a.get("upstream_modules")) for a in anchors),
        "downstream": any(bool(a.get("downstream_modules")) for a in anchors),
        "related": any(bool(a.get("related_doc_keys")) for a in anchors),
    }


def build_potential_linked_modules(
    *,
    task_query: str,
    anchors: List[Dict[str, Any]],
    current_involved_modules: List[Dict[str, Any]],
    max_modules: int = 2,
) -> List[Dict[str, Any]]:
    if not anchors or not current_involved_modules:
        return []

    task_has_linkage = _contains_any(task_query, LINKAGE_TERMS)

    current_conf: Dict[str, float] = {}
    for item in current_involved_modules:
        if not isinstance(item, dict):
            continue
        module = canonicalize_module(item.get("module"))
        if not module:
            continue
        confidence = float(item.get("confidence") or 0.0)
        current_conf[module] = max(current_conf.get(module, 0.0), confidence)

    trigger_modules = [module for module, conf in current_conf.items() if conf >= 0.68]
    if not trigger_modules:
        return []

    # current 置信度不稳时，宁可不输出 linked。
    if max([current_conf.get(m, 0.0) for m in trigger_modules] + [0.0]) < 0.70:
        return []

    current_set = set(trigger_modules)

    anchors_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    anchors_by_feature: Dict[str, List[Dict[str, Any]]] = {}
    anchors_by_trace: Dict[str, List[Dict[str, Any]]] = {}
    anchors_by_domain: Dict[str, List[Dict[str, Any]]] = {}

    extra_candidates: Set[str] = set(current_set)
    for anchor in anchors:
        for module in anchor.get("module_tags", []) or []:
            canonical = canonicalize_module(module)
            if canonical:
                extra_candidates.add(canonical)

        doc_key = str(anchor.get("doc_key") or "").strip()
        if doc_key:
            anchors_by_doc.setdefault(doc_key, []).append(anchor)

        feature_key = str(anchor.get("feature_key") or "").strip()
        if feature_key:
            anchors_by_feature.setdefault(feature_key, []).append(anchor)

        for trace in anchor.get("trace_refs", []) or []:
            trace_text = str(trace).strip()
            if trace_text:
                anchors_by_trace.setdefault(trace_text, []).append(anchor)

        for domain in anchor.get("domain_tags", []) or []:
            domain_text = str(domain).strip()
            if domain_text:
                anchors_by_domain.setdefault(domain_text, []).append(anchor)

    candidates: Dict[str, Dict[str, Any]] = {}

    def add_candidate(
        module: str,
        *,
        trigger_module: str,
        edge_type: str,
        anchor: Dict[str, Any],
        edge_value: str,
    ) -> None:
        canonical = canonicalize_module(module, extra_candidates)
        if not canonical or canonical in current_set:
            return
        if not anchor_has_valid_locator(anchor):
            return

        item = candidates.setdefault(
            canonical,
            {
                "module": canonical,
                "edge_types": set(),
                "edge_counts": {},
                "edge_score_sum": 0.0,
                "trigger_modules": set(),
                "best_anchor": None,
                "best_anchor_rank": None,
                "top_edge_type": "",
                "top_edge_value": "",
            },
        )

        item["edge_types"].add(edge_type)
        edge_counts = item["edge_counts"]
        edge_counts[edge_type] = int(edge_counts.get(edge_type, 0)) + 1
        item["edge_score_sum"] += float(_EDGE_BASE_STRENGTH.get(edge_type, 0.0))
        item["trigger_modules"].add(trigger_module)

        source_type = str(anchor.get("source_type") or "unknown")
        rank = (
            SOURCE_PRIORITY.get(source_type, SOURCE_PRIORITY["unknown"]),
            -float(anchor.get("source_confidence") or 0.0),
            int(anchor.get("chunk_index") or 9999),
        )
        old_rank = item.get("best_anchor_rank")
        if old_rank is None or rank < old_rank:
            item["best_anchor"] = anchor
            item["best_anchor_rank"] = rank
            item["top_edge_type"] = edge_type
            item["top_edge_value"] = edge_value

    for trigger_module in trigger_modules:
        trigger_anchors = [a for a in anchors if _anchor_module_hit(a, trigger_module)]
        for anchor in trigger_anchors:
            downstream = _canonical_list(anchor.get("downstream_modules", []), extra_candidates)
            for module in downstream:
                add_candidate(
                    module,
                    trigger_module=trigger_module,
                    edge_type="downstream",
                    anchor=anchor,
                    edge_value="downstream_modules",
                )

            upstream = _canonical_list(anchor.get("upstream_modules", []), extra_candidates)
            for module in upstream:
                add_candidate(
                    module,
                    trigger_module=trigger_module,
                    edge_type="upstream",
                    anchor=anchor,
                    edge_value="upstream_modules",
                )

            for doc_key in anchor.get("related_doc_keys", []) or []:
                related_anchors = anchors_by_doc.get(str(doc_key).strip(), [])
                for related_anchor in related_anchors:
                    related_modules = _canonical_list(related_anchor.get("module_tags", []), extra_candidates)
                    for module in related_modules:
                        add_candidate(
                            module,
                            trigger_module=trigger_module,
                            edge_type="related_doc",
                            anchor=related_anchor,
                            edge_value=f"related_doc_keys:{doc_key}",
                        )

            for trace in anchor.get("trace_refs", []) or []:
                peers = anchors_by_trace.get(str(trace).strip(), [])
                for peer in peers:
                    if peer is anchor:
                        continue
                    peer_modules = _canonical_list(peer.get("module_tags", []), extra_candidates)
                    for module in peer_modules:
                        add_candidate(
                            module,
                            trigger_module=trigger_module,
                            edge_type="shared_trace",
                            anchor=peer,
                            edge_value=f"shared_trace_refs:{trace}",
                        )

            feature_key = str(anchor.get("feature_key") or "").strip()
            if feature_key:
                peers = anchors_by_feature.get(feature_key, [])
                for peer in peers:
                    if peer is anchor:
                        continue
                    peer_modules = _canonical_list(peer.get("module_tags", []), extra_candidates)
                    for module in peer_modules:
                        add_candidate(
                            module,
                            trigger_module=trigger_module,
                            edge_type="shared_feature",
                            anchor=peer,
                            edge_value=f"shared_feature_key:{feature_key}",
                        )

            for domain in anchor.get("domain_tags", []) or []:
                peers = anchors_by_domain.get(str(domain).strip(), [])
                for peer in peers:
                    if peer is anchor:
                        continue
                    peer_modules = _canonical_list(peer.get("module_tags", []), extra_candidates)
                    for module in peer_modules:
                        add_candidate(
                            module,
                            trigger_module=trigger_module,
                            edge_type="domain_cooccur",
                            anchor=peer,
                            edge_value=f"domain_cooccur:{domain}",
                        )

    linked: List[Dict[str, Any]] = []
    for module, item in candidates.items():
        edges: Set[str] = set(item.get("edge_types") or set())
        if not edges or edges == {"domain_cooccur"}:
            continue

        edge_counts = item.get("edge_counts", {}) or {}
        non_domain_edge_count = sum(int(v) for k, v in edge_counts.items() if k != "domain_cooccur")
        has_downstream = "downstream" in edges
        has_strong_relation = bool(edges & {"related_doc", "shared_trace", "shared_feature"})

        # 方向约束：current <- upstream 需要更强证据，不允许单边反向误报。
        if not has_downstream:
            if not has_strong_relation:
                continue
            if non_domain_edge_count < 2:
                continue

        # 无联动语义时，进一步收紧。
        if (not task_has_linkage) and (not has_downstream) and ("related_doc" not in edges):
            continue

        best_anchor = item.get("best_anchor")
        if not isinstance(best_anchor, dict):
            continue

        trigger_list = sorted(
            [canonicalize_module(x, extra_candidates) for x in item.get("trigger_modules", set()) if canonicalize_module(x, extra_candidates)]
        )
        if not trigger_list:
            continue

        trigger_conf = max([current_conf.get(trigger, 0.0) for trigger in trigger_list] + [0.0])
        anchor_quality = _anchor_quality(best_anchor)
        source_trust = float(best_anchor.get("source_confidence") or 0.0)

        support_bonus = min(0.20, 0.05 * max(0, non_domain_edge_count - 1))
        max_edge = max([float(_EDGE_BASE_STRENGTH.get(edge, 0.0)) for edge in edges] + [0.0])
        edge_strength = min(1.0, max_edge + support_bonus)
        if ("upstream" in edges) and (not has_downstream):
            edge_strength = max(0.0, edge_strength - 0.12)

        linked_score = 0.50 * edge_strength + 0.20 * anchor_quality + 0.20 * trigger_conf + 0.10 * source_trust

        if linked_score < 0.68:
            continue

        confidence = round(linked_score, 4)
        if confidence >= 0.78:
            confidence_level = "high"
        elif confidence >= 0.68:
            confidence_level = "medium"
        else:
            continue

        impact_type = _best_edge_type(edges)
        top_edge_value = str(item.get("top_edge_value") or impact_type)
        source_type = str(best_anchor.get("source_type") or "unknown")
        source_name = str(best_anchor.get("source_name") or "unknown")
        chunk_index = best_anchor.get("chunk_index")
        top_evidence = (
            f"触发模块:{'、'.join(trigger_list)} -> {top_edge_value}；"
            f"证据[{source_type}] {source_name}#片段{chunk_index}"
        )

        linked.append(
            {
                "module": module,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "impact_type": impact_type,
                "trigger_modules": trigger_list,
                "top_evidence": top_evidence,
                "evidence_anchor": best_anchor,
            }
        )

    linked.sort(key=lambda x: (-float(x.get("confidence") or 0.0), str(x.get("module") or "")))
    return linked[: max(1, int(max_modules))]


def build_impact_analysis_v2(
    *,
    task_query: str,
    anchors: List[Dict[str, Any]],
    current_involved_modules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    potential = build_potential_linked_modules(
        task_query=task_query,
        anchors=anchors,
        current_involved_modules=current_involved_modules,
    )

    current_names = [
        canonicalize_module(item.get("module"))
        for item in current_involved_modules
        if isinstance(item, dict) and canonicalize_module(item.get("module"))
    ]
    linked_names = [
        canonicalize_module(item.get("module"))
        for item in potential
        if isinstance(item, dict) and canonicalize_module(item.get("module"))
    ]

    summary = ""
    if current_names:
        summary = "当前需求涉及" + "、".join(current_names)
    if linked_names:
        suffix = "可能联动" + "、".join(linked_names)
        summary = f"{summary}，{suffix}" if summary else suffix
    summary = summary[:120]

    return {
        "version": "v2",
        "impact_summary": summary,
        "current_involved_modules": current_involved_modules,
        "potential_linked_modules": potential,
        "evidence_anchors": anchors,
        "relation_consumed": _build_relation_consumed(anchors),
        "task_has_linkage_terms": _contains_any(task_query, LINKAGE_TERMS),
    }
