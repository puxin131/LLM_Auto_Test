from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .attribution_engine import canonicalize_module
from .evidence_anchor import build_evidence_anchors

RELATION_BASE_SCORE: Dict[str, float] = {
    "related_doc": 0.78,
    "shared_trace": 0.72,
    "shared_feature": 0.68,
    "module_downstream": 0.74,
    "module_upstream": 0.66,
    "doc_module": 0.60,
}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _node_doc_id(doc_key: str) -> str:
    return f"doc:{doc_key}"


def _node_module_id(module_name: str) -> str:
    return f"module:{module_name}"


def _build_doc_nodes(anchors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    docs: Dict[str, Dict[str, Any]] = {}
    for anchor in anchors:
        doc_key = _normalize_text(anchor.get("doc_key"))
        if not doc_key:
            continue
        source_type = _normalize_text(anchor.get("source_type")) or "unknown"
        source_name = _normalize_text(anchor.get("source_name")) or "unknown"
        score = float(anchor.get("source_confidence") or 0.0)

        item = docs.get(doc_key)
        if item is None:
            docs[doc_key] = {
                "doc_key": doc_key,
                "source_type": source_type,
                "source_name": source_name,
                "source_confidence": score,
            }
        else:
            item["source_confidence"] = max(float(item.get("source_confidence", 0.0)), score)
    return docs


def _trace_refs_from_docs(docs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    req_ids: List[str] = []
    api_ids: List[str] = []
    testcase_ids: List[str] = []
    ui_ids: List[str] = []
    for doc_key, meta in docs.items():
        source_type = str(meta.get("source_type", "unknown"))
        if source_type == "requirement" and doc_key not in req_ids:
            req_ids.append(doc_key)
        elif source_type == "api_doc" and doc_key not in api_ids:
            api_ids.append(doc_key)
        elif source_type == "testcase" and doc_key not in testcase_ids:
            testcase_ids.append(doc_key)
        elif source_type == "ui" and doc_key not in ui_ids:
            ui_ids.append(doc_key)
    return {
        "req_ids": req_ids[:20],
        "api_ids": api_ids[:20],
        "testcase_ids": testcase_ids[:20],
        "ui_ids": ui_ids[:20],
    }


def _edge_confidence(
    *,
    relation: str,
    src_conf: float,
    dst_conf: float,
) -> float:
    base = float(RELATION_BASE_SCORE.get(relation, 0.55))
    conf = base + (0.12 * min(src_conf, dst_conf))
    if conf > 0.95:
        conf = 0.95
    return round(conf, 4)


def _add_edge(
    edges: Dict[Tuple[str, str, str], Dict[str, Any]],
    *,
    src_id: str,
    src_type: str,
    dst_id: str,
    dst_type: str,
    relation: str,
    confidence: float,
    evidence_anchor_id: str,
) -> None:
    if not src_id or not dst_id or src_id == dst_id:
        return
    key = (src_id, dst_id, relation)
    old = edges.get(key)
    payload = {
        "src_id": src_id,
        "src_type": src_type,
        "dst_id": dst_id,
        "dst_type": dst_type,
        "relation": relation,
        "confidence": float(confidence),
        "evidence_anchor_id": evidence_anchor_id,
    }
    if old is None or float(payload["confidence"]) > float(old.get("confidence", 0.0)):
        edges[key] = payload


def _add_bidirectional_edges(
    edges: Dict[Tuple[str, str, str], Dict[str, Any]],
    *,
    src_id: str,
    src_type: str,
    dst_id: str,
    dst_type: str,
    relation: str,
    reverse_relation: str,
    confidence: float,
    evidence_anchor_id: str,
) -> None:
    _add_edge(
        edges,
        src_id=src_id,
        src_type=src_type,
        dst_id=dst_id,
        dst_type=dst_type,
        relation=relation,
        confidence=confidence,
        evidence_anchor_id=evidence_anchor_id,
    )
    _add_edge(
        edges,
        src_id=dst_id,
        src_type=dst_type,
        dst_id=src_id,
        dst_type=src_type,
        relation=reverse_relation,
        confidence=confidence,
        evidence_anchor_id=evidence_anchor_id,
    )


def build_bidirectional_link_analysis(
    *,
    task_query: str,
    retrieval_context: str,
    max_edges: int = 80,
) -> Dict[str, Any]:
    _ = task_query
    anchors = build_evidence_anchors(retrieval_context)
    docs = _build_doc_nodes(anchors)
    doc_keys = set(docs.keys())
    edges_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    trace_to_docs: Dict[str, List[str]] = {}
    feature_to_docs: Dict[str, List[str]] = {}

    for anchor in anchors:
        doc_key = _normalize_text(anchor.get("doc_key"))
        if not doc_key or doc_key not in doc_keys:
            continue
        anchor_id = _normalize_text(anchor.get("anchor_id"))
        source_conf = float(anchor.get("source_confidence") or 0.0)
        source_type = str(docs.get(doc_key, {}).get("source_type", "unknown"))
        source_id = _node_doc_id(doc_key)

        related_doc_keys = anchor.get("related_doc_keys", [])
        if not isinstance(related_doc_keys, list):
            related_doc_keys = [related_doc_keys]
        for related_doc in related_doc_keys:
            dst_doc = _normalize_text(related_doc)
            if not dst_doc or dst_doc not in doc_keys:
                continue
            dst_conf = float(docs.get(dst_doc, {}).get("source_confidence", 0.0))
            confidence = _edge_confidence(
                relation="related_doc",
                src_conf=source_conf,
                dst_conf=dst_conf,
            )
            _add_bidirectional_edges(
                edges_map,
                src_id=source_id,
                src_type=source_type,
                dst_id=_node_doc_id(dst_doc),
                dst_type=str(docs.get(dst_doc, {}).get("source_type", "unknown")),
                relation="related_doc",
                reverse_relation="related_doc",
                confidence=confidence,
                evidence_anchor_id=anchor_id,
            )

        trace_refs = anchor.get("trace_refs", [])
        if not isinstance(trace_refs, list):
            trace_refs = [trace_refs]
        for trace in trace_refs:
            trace_id = _normalize_text(trace)
            if not trace_id:
                continue
            trace_to_docs.setdefault(trace_id, [])
            if doc_key not in trace_to_docs[trace_id]:
                trace_to_docs[trace_id].append(doc_key)

        feature_key = _normalize_text(anchor.get("feature_key"))
        if feature_key:
            feature_to_docs.setdefault(feature_key, [])
            if doc_key not in feature_to_docs[feature_key]:
                feature_to_docs[feature_key].append(doc_key)

        modules = anchor.get("module_tags", [])
        if not isinstance(modules, list):
            modules = [modules]
        canonical_modules: List[str] = []
        for mod in modules:
            canonical = canonicalize_module(mod)
            if canonical and canonical not in canonical_modules:
                canonical_modules.append(canonical)

        for module_name in canonical_modules:
            mod_id = _node_module_id(module_name)
            confidence = _edge_confidence(
                relation="doc_module",
                src_conf=source_conf,
                dst_conf=source_conf,
            )
            _add_bidirectional_edges(
                edges_map,
                src_id=source_id,
                src_type=source_type,
                dst_id=mod_id,
                dst_type="module",
                relation="doc_module",
                reverse_relation="module_doc",
                confidence=confidence,
                evidence_anchor_id=anchor_id,
            )

        upstream_modules = anchor.get("upstream_modules", [])
        downstream_modules = anchor.get("downstream_modules", [])
        if not isinstance(upstream_modules, list):
            upstream_modules = [upstream_modules]
        if not isinstance(downstream_modules, list):
            downstream_modules = [downstream_modules]
        canonical_upstream = [canonicalize_module(x) for x in upstream_modules if _normalize_text(x)]
        canonical_downstream = [canonicalize_module(x) for x in downstream_modules if _normalize_text(x)]
        canonical_upstream = [x for x in canonical_upstream if x]
        canonical_downstream = [x for x in canonical_downstream if x]

        for src_module in canonical_modules:
            for dst_module in canonical_downstream:
                confidence = _edge_confidence(
                    relation="module_downstream",
                    src_conf=source_conf,
                    dst_conf=source_conf,
                )
                _add_bidirectional_edges(
                    edges_map,
                    src_id=_node_module_id(src_module),
                    src_type="module",
                    dst_id=_node_module_id(dst_module),
                    dst_type="module",
                    relation="module_downstream",
                    reverse_relation="module_upstream",
                    confidence=confidence,
                    evidence_anchor_id=anchor_id,
                )
            for up_module in canonical_upstream:
                confidence = _edge_confidence(
                    relation="module_upstream",
                    src_conf=source_conf,
                    dst_conf=source_conf,
                )
                _add_bidirectional_edges(
                    edges_map,
                    src_id=_node_module_id(src_module),
                    src_type="module",
                    dst_id=_node_module_id(up_module),
                    dst_type="module",
                    relation="module_upstream",
                    reverse_relation="module_downstream",
                    confidence=confidence,
                    evidence_anchor_id=anchor_id,
                )

    for trace_id, trace_docs in trace_to_docs.items():
        if len(trace_docs) < 2:
            continue
        for i, src_doc in enumerate(trace_docs):
            for dst_doc in trace_docs[i + 1 :]:
                src_conf = float(docs.get(src_doc, {}).get("source_confidence", 0.0))
                dst_conf = float(docs.get(dst_doc, {}).get("source_confidence", 0.0))
                confidence = _edge_confidence(
                    relation="shared_trace",
                    src_conf=src_conf,
                    dst_conf=dst_conf,
                )
                _add_bidirectional_edges(
                    edges_map,
                    src_id=_node_doc_id(src_doc),
                    src_type=str(docs.get(src_doc, {}).get("source_type", "unknown")),
                    dst_id=_node_doc_id(dst_doc),
                    dst_type=str(docs.get(dst_doc, {}).get("source_type", "unknown")),
                    relation="shared_trace",
                    reverse_relation="shared_trace",
                    confidence=confidence,
                    evidence_anchor_id=f"trace:{trace_id}",
                )

    for feature_key, feature_docs in feature_to_docs.items():
        if len(feature_docs) < 2:
            continue
        for i, src_doc in enumerate(feature_docs):
            for dst_doc in feature_docs[i + 1 :]:
                src_conf = float(docs.get(src_doc, {}).get("source_confidence", 0.0))
                dst_conf = float(docs.get(dst_doc, {}).get("source_confidence", 0.0))
                confidence = _edge_confidence(
                    relation="shared_feature",
                    src_conf=src_conf,
                    dst_conf=dst_conf,
                )
                _add_bidirectional_edges(
                    edges_map,
                    src_id=_node_doc_id(src_doc),
                    src_type=str(docs.get(src_doc, {}).get("source_type", "unknown")),
                    dst_id=_node_doc_id(dst_doc),
                    dst_type=str(docs.get(dst_doc, {}).get("source_type", "unknown")),
                    relation="shared_feature",
                    reverse_relation="shared_feature",
                    confidence=confidence,
                    evidence_anchor_id=f"feature:{feature_key}",
                )

    edges = sorted(
        edges_map.values(),
        key=lambda item: (
            -float(item.get("confidence", 0.0)),
            str(item.get("relation", "")),
            str(item.get("src_id", "")),
            str(item.get("dst_id", "")),
        ),
    )[:max_edges]
    trace_refs = _trace_refs_from_docs(docs)

    doc_doc_edges = [e for e in edges if str(e.get("src_type")) != "module" and str(e.get("dst_type")) != "module"]
    module_edges = [e for e in edges if str(e.get("src_type")) == "module" or str(e.get("dst_type")) == "module"]

    summary = {
        "total_edges": len(edges),
        "doc_doc_edges": len(doc_doc_edges),
        "module_edges": len(module_edges),
        "relations": sorted(
            list({str(edge.get("relation", "")) for edge in edges if str(edge.get("relation", "")).strip()})
        ),
    }

    return {
        "link_edges": edges,
        "trace_refs": trace_refs,
        "link_summary": summary,
    }

