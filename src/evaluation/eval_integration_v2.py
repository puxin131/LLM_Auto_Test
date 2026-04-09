from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.rag.analysis import (
    build_dual_contracts,
    build_evidence_anchors,
    build_integration_coverage_matrix,
    build_mapping_rules,
)


@dataclass
class EvalThresholds:
    internal_assertion_coverage: float = 0.85
    mapping_rule_hit_rate: float = 0.75
    integration_exception_coverage: float = 0.70
    evidence_traceability_rate: float = 0.95


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        values = [values]
    out: List[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _set_norm(values: Any) -> set[str]:
    return {_norm(v) for v in _norm_list(values) if _norm(v)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate integration-oriented generation supports (v2)")
    parser.add_argument("--samples", default="data/eval/impact_v2_samples.jsonl")
    parser.add_argument("--report", default="data/eval/reports/integration_v2_eval_report.json")
    parser.add_argument("--strict", default=True, action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def load_samples(path: str) -> List[Dict[str, Any]]:
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(f"samples not found: {sample_path}")
    rows: List[Dict[str, Any]] = []
    for idx, raw_line in enumerate(sample_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"line {idx}: sample should be object")
        for key in ("sample_id", "task_query", "retrieval_context"):
            if key not in item:
                raise ValueError(f"line {idx}: missing required field {key}")
        item.setdefault("category", "default")
        item.setdefault("gold_internal_interfaces", [])
        item.setdefault("gold_external_interfaces", [])
        item.setdefault("gold_mapping_rule_keys", [])
        item.setdefault("gold_exception_points", [])
        rows.append(item)
    return rows


def run_inference(task_query: str, retrieval_context: str) -> Dict[str, Any]:
    anchors = build_evidence_anchors(retrieval_context)
    contracts = build_dual_contracts(
        task_query=task_query,
        retrieval_context=retrieval_context,
        anchors=anchors,
    )
    mappings = build_mapping_rules(
        task_query=task_query,
        retrieval_context=retrieval_context,
        anchors=anchors,
    )
    coverage = build_integration_coverage_matrix(
        task_query=task_query,
        retrieval_context=retrieval_context,
        current_modules=[],
    )
    return {
        "contracts": contracts,
        "mapping_rules": mappings,
        "coverage_matrix": coverage,
    }


def evaluate_sample(sample: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    contracts = pred.get("contracts", {}) if isinstance(pred.get("contracts"), dict) else {}
    mappings = pred.get("mapping_rules", {}) if isinstance(pred.get("mapping_rules"), dict) else {}
    coverage = pred.get("coverage_matrix", {}) if isinstance(pred.get("coverage_matrix"), dict) else {}

    internal = contracts.get("internal_contract", {}) if isinstance(contracts.get("internal_contract"), dict) else {}
    external = contracts.get("external_contract", {}) if isinstance(contracts.get("external_contract"), dict) else {}
    mapping_items = mappings.get("mapping_rules", []) if isinstance(mappings.get("mapping_rules"), list) else []
    matrix_items = (
        coverage.get("coverage_matrix", []) if isinstance(coverage.get("coverage_matrix"), list) else []
    )

    pred_internal = _set_norm(internal.get("interfaces", []))
    pred_external = _set_norm(external.get("interfaces", []))
    pred_mapping = _set_norm([item.get("rule_key", "") for item in mapping_items if isinstance(item, dict)])
    pred_exceptions = _set_norm(
        [
            f"{item.get('domain', '')}-{item.get('action', '')}-{item.get('outcome', '')}"
            for item in matrix_items
            if isinstance(item, dict)
            and bool(item.get("selected"))
            and str(item.get("action", "")) in {"重试", "补偿", "回滚"}
            and str(item.get("outcome", "")) != "成功"
        ]
    )

    gold_internal = _set_norm(sample.get("gold_internal_interfaces", []))
    gold_external = _set_norm(sample.get("gold_external_interfaces", []))
    gold_mapping = _set_norm(sample.get("gold_mapping_rule_keys", []))
    gold_exceptions = _set_norm(sample.get("gold_exception_points", []))

    internal_hits = len((gold_internal | gold_external) & (pred_internal | pred_external))
    internal_total = len(gold_internal | gold_external)
    internal_coverage = internal_hits / max(1, internal_total) if internal_total else 0.0

    mapping_hits = len(gold_mapping & pred_mapping)
    mapping_total = len(gold_mapping)
    mapping_coverage = mapping_hits / max(1, mapping_total) if mapping_total else 0.0

    exception_hits = len(gold_exceptions & pred_exceptions)
    exception_total = len(gold_exceptions)
    exception_coverage = exception_hits / max(1, exception_total) if exception_total else 0.0

    anchor_ids = []
    for anchor_id in _norm_list(internal.get("evidence_anchor_ids", [])) + _norm_list(
        external.get("evidence_anchor_ids", [])
    ):
        if anchor_id and anchor_id not in anchor_ids:
            anchor_ids.append(anchor_id)
    for item in mapping_items:
        if not isinstance(item, dict):
            continue
        anchor_id = str(item.get("evidence_anchor_id", "")).strip()
        if anchor_id and anchor_id not in anchor_ids:
            anchor_ids.append(anchor_id)
    traceable = [aid for aid in anchor_ids if "#" in aid and len(aid.split("#", 1)[0]) > 0]
    traceability = len(traceable) / max(1, len(anchor_ids)) if anchor_ids else 1.0

    return {
        "sample_id": str(sample.get("sample_id")),
        "category": str(sample.get("category", "default")),
        "internal_assertion_coverage": round(internal_coverage, 6),
        "mapping_rule_hit_rate": round(mapping_coverage, 6),
        "integration_exception_coverage": round(exception_coverage, 6),
        "evidence_traceability_rate": round(traceability, 6),
        "pred_summary": {
            "internal_interfaces": sorted(pred_internal)[:8],
            "external_interfaces": sorted(pred_external)[:8],
            "mapping_rule_keys": sorted(pred_mapping)[:8],
            "exception_points": sorted(pred_exceptions)[:8],
            "anchor_count": len(anchor_ids),
        },
    }


def aggregate(
    rows: List[Dict[str, Any]], thresholds: EvalThresholds
) -> Dict[str, Any]:
    def mean(key: str) -> float:
        values = [float(item.get(key, 0.0) or 0.0) for item in rows]
        return round(sum(values) / max(1, len(values)), 6)

    metrics = {
        "internal_assertion_coverage": mean("internal_assertion_coverage"),
        "mapping_rule_hit_rate": mean("mapping_rule_hit_rate"),
        "integration_exception_coverage": mean("integration_exception_coverage"),
        "evidence_traceability_rate": mean("evidence_traceability_rate"),
        "sample_count": len(rows),
    }
    gates = {
        "internal_assertion_coverage": metrics["internal_assertion_coverage"]
        >= thresholds.internal_assertion_coverage,
        "mapping_rule_hit_rate": metrics["mapping_rule_hit_rate"] >= thresholds.mapping_rule_hit_rate,
        "integration_exception_coverage": metrics["integration_exception_coverage"]
        >= thresholds.integration_exception_coverage,
        "evidence_traceability_rate": metrics["evidence_traceability_rate"]
        >= thresholds.evidence_traceability_rate,
    }
    gates["overall_pass"] = all(gates.values())

    by_category: Dict[str, Dict[str, Any]] = {}
    for item in rows:
        category = str(item.get("category", "default"))
        by_category.setdefault(category, {"rows": []})
        by_category[category]["rows"].append(item)
    for category, payload in by_category.items():
        bucket_rows = payload["rows"]
        payload["count"] = len(bucket_rows)
        payload["internal_assertion_coverage"] = round(
            sum(float(r.get("internal_assertion_coverage", 0.0) or 0.0) for r in bucket_rows)
            / max(1, len(bucket_rows)),
            6,
        )
        payload["mapping_rule_hit_rate"] = round(
            sum(float(r.get("mapping_rule_hit_rate", 0.0) or 0.0) for r in bucket_rows)
            / max(1, len(bucket_rows)),
            6,
        )
        payload["integration_exception_coverage"] = round(
            sum(float(r.get("integration_exception_coverage", 0.0) or 0.0) for r in bucket_rows)
            / max(1, len(bucket_rows)),
            6,
        )
        payload["evidence_traceability_rate"] = round(
            sum(float(r.get("evidence_traceability_rate", 0.0) or 0.0) for r in bucket_rows)
            / max(1, len(bucket_rows)),
            6,
        )
        payload.pop("rows", None)

    return {"metrics": metrics, "gates": gates, "by_category": by_category}


def main() -> int:
    args = parse_args()
    thresholds = EvalThresholds()
    samples = load_samples(args.samples)
    rows = [evaluate_sample(sample, run_inference(sample["task_query"], sample["retrieval_context"])) for sample in samples]
    aggregate_payload = aggregate(rows, thresholds)
    report = {
        "run_meta": {
            "ts": _utc_now_iso(),
            "samples_path": args.samples,
            "total_samples": len(samples),
        },
        "thresholds": asdict(thresholds),
        **aggregate_payload,
        "errors": [item for item in rows if min(
            float(item.get("internal_assertion_coverage", 0.0)),
            float(item.get("mapping_rule_hit_rate", 0.0)),
            float(item.get("integration_exception_coverage", 0.0)),
        ) < 1.0][:30],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    gates = report["gates"]
    print("Integration V2 Eval")
    print(
        " | ".join(
            [
                f"internal={metrics['internal_assertion_coverage']:.3f}",
                f"mapping={metrics['mapping_rule_hit_rate']:.3f}",
                f"exception={metrics['integration_exception_coverage']:.3f}",
                f"trace={metrics['evidence_traceability_rate']:.3f}",
                f"pass={gates['overall_pass']}",
            ]
        )
    )
    print(f"report: {report_path}")
    if args.strict and (not bool(gates.get("overall_pass"))):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
