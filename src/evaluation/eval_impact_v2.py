from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.rag.analysis import (
    build_current_involved_modules,
    build_evidence_anchors,
    build_impact_analysis_v2,
)


@dataclass
class EvalThresholds:
    top1_precision: float = 0.70
    top2_recall: float = 0.82
    linked_precision: float = 0.55
    triplet_constraint_rate: float = 1.0
    evidence_traceability_rate: float = 0.90
    avg_linked_modules_max: float = 1.5


@dataclass
class EvalArgs:
    samples: str = "data/eval/impact_v2_samples.jsonl"
    report: str = "data/eval/reports/impact_v2_eval_report.json"
    strict: bool = True
    print_errors: int = 20


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_name_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        values = [values]
    output: List[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in output:
            output.append(text)
    return output


def _normalize_set(values: Any) -> set[str]:
    return {str(x).strip().lower() for x in _normalize_name_list(values) if str(x).strip()}


def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser(description="Evaluate impact_analysis v2 on labeled samples")
    parser.add_argument("--samples", default="data/eval/impact_v2_samples.jsonl", help="path to JSONL samples")
    parser.add_argument(
        "--report",
        default="data/eval/reports/impact_v2_eval_report.json",
        help="path to output report json",
    )
    parser.add_argument(
        "--strict",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="return non-zero when overall gate fails",
    )
    parser.add_argument(
        "--print-errors",
        type=int,
        default=20,
        help="print top N error samples in console",
    )
    ns = parser.parse_args()
    return EvalArgs(
        samples=str(ns.samples),
        report=str(ns.report),
        strict=bool(ns.strict),
        print_errors=max(0, int(ns.print_errors)),
    )


def load_samples_jsonl(path: str) -> List[Dict[str, Any]]:
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Samples file not found: {sample_path}")

    required = {
        "sample_id",
        "category",
        "task_query",
        "retrieval_context",
        "gold_current_modules",
        "gold_linked_modules",
        "gold_anchor_ids_primary",
    }

    rows: List[Dict[str, Any]] = []
    for line_no, raw_line in enumerate(sample_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_no}: sample must be JSON object")

        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Line {line_no}: missing fields {missing}")

        item.setdefault("gold_current_top1", "")
        item.setdefault("gold_anchor_ids_supporting", [])
        item.setdefault("expect_no_linked", False)
        item.setdefault("notes", "")

        item["gold_current_modules"] = _normalize_name_list(item.get("gold_current_modules", []))
        item["gold_linked_modules"] = _normalize_name_list(item.get("gold_linked_modules", []))
        item["gold_anchor_ids_primary"] = _normalize_name_list(item.get("gold_anchor_ids_primary", []))
        item["gold_anchor_ids_supporting"] = _normalize_name_list(item.get("gold_anchor_ids_supporting", []))
        rows.append(item)

    return rows


def run_impact_v2_inference(task_query: str, retrieval_context: str) -> Dict[str, Any]:
    anchors = build_evidence_anchors(retrieval_context)
    current = build_current_involved_modules(
        task_query=task_query,
        anchors=anchors,
        max_modules=2,
    )
    impact = build_impact_analysis_v2(
        task_query=task_query,
        anchors=anchors,
        current_involved_modules=current,
    )
    return impact if isinstance(impact, dict) else {}


def extract_pred_current_modules(pred_impact: Dict[str, Any]) -> List[str]:
    rows = pred_impact.get("current_involved_modules", [])
    if not isinstance(rows, list):
        return []

    ranked: List[Tuple[float, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        module = str(item.get("module", "")).strip()
        if not module:
            continue
        confidence = float(item.get("confidence", 0.0) or 0.0)
        ranked.append((confidence, module))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    ordered: List[str] = []
    for _, module in ranked:
        if module not in ordered:
            ordered.append(module)
    return ordered[:2]


def extract_top1_top2(pred_current_modules: List[str]) -> Tuple[str, List[str]]:
    top2 = pred_current_modules[:2]
    top1 = top2[0] if top2 else ""
    return top1, top2


def extract_pred_linked_modules(pred_impact: Dict[str, Any]) -> List[str]:
    rows = pred_impact.get("potential_linked_modules", [])
    if not isinstance(rows, list):
        return []
    modules: List[str] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        module = str(item.get("module", "")).strip()
        if module and module not in modules:
            modules.append(module)
    return modules[:2]


def _anchor_traceable(anchor: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(anchor, dict):
        return False, ""
    anchor_id = str(anchor.get("anchor_id", "")).strip()
    doc_key = str(anchor.get("doc_key", "")).strip()
    chunk_index = anchor.get("chunk_index")
    source_type = str(anchor.get("source_type", "")).strip()
    source_name = str(anchor.get("source_name", "")).strip()
    if not (anchor_id and doc_key and isinstance(chunk_index, int) and source_type and source_name):
        return False, anchor_id
    expected = f"{doc_key}#{chunk_index}"
    return anchor_id == expected, anchor_id


def validate_triplet_constraints(pred_impact: Dict[str, Any]) -> Dict[str, Any]:
    linked = pred_impact.get("potential_linked_modules", [])
    if not isinstance(linked, list):
        linked = []

    total = 0
    valid = 0
    violations: List[str] = []

    for item in linked:
        if not isinstance(item, dict):
            continue
        total += 1
        module = str(item.get("module", "")).strip() or "<unknown>"
        triggers = item.get("trigger_modules", [])
        impact_type = str(item.get("impact_type", "")).strip()
        anchor = item.get("evidence_anchor")
        ok_anchor, _ = _anchor_traceable(anchor if isinstance(anchor, dict) else {})
        ok = bool(isinstance(triggers, list) and len([x for x in triggers if str(x).strip()]) > 0)
        ok = ok and bool(impact_type) and ok_anchor
        if ok:
            valid += 1
        else:
            violations.append(module)

    if total == 0:
        return {"ok": True, "valid": 0, "total": 0, "violations": []}
    return {"ok": valid == total, "valid": valid, "total": total, "violations": violations}


def validate_evidence_traceability(pred_impact: Dict[str, Any]) -> Dict[str, Any]:
    current = pred_impact.get("current_involved_modules", [])
    linked = pred_impact.get("potential_linked_modules", [])
    if not isinstance(current, list):
        current = []
    if not isinstance(linked, list):
        linked = []

    total = 0
    valid = 0
    anchor_ids: List[str] = []
    bad_modules: List[str] = []

    for item in current + linked:
        if not isinstance(item, dict):
            continue
        module = str(item.get("module", "")).strip() or "<unknown>"
        anchor = item.get("evidence_anchor")
        if not isinstance(anchor, dict):
            total += 1
            bad_modules.append(module)
            continue

        total += 1
        ok, anchor_id = _anchor_traceable(anchor)
        if anchor_id and anchor_id not in anchor_ids:
            anchor_ids.append(anchor_id)
        if ok:
            valid += 1
        else:
            bad_modules.append(module)

    if total == 0:
        return {"ok": True, "valid": 0, "total": 0, "anchor_ids": anchor_ids, "bad_modules": []}
    return {
        "ok": valid == total,
        "valid": valid,
        "total": total,
        "anchor_ids": anchor_ids,
        "bad_modules": bad_modules,
    }


def _top2_recall(gold_modules: List[str], pred_top2: List[str]) -> float:
    gold = _normalize_set(gold_modules)
    if not gold:
        return 0.0
    pred = _normalize_set(pred_top2)
    hit = len(gold & pred)
    return hit / max(1, len(gold))


def evaluate_one_sample(sample: Dict[str, Any], pred_impact: Dict[str, Any]) -> Dict[str, Any]:
    category = str(sample.get("category", "")).strip()
    weak_input = category == "weak_input"

    pred_current_modules = extract_pred_current_modules(pred_impact)
    pred_current_top1, pred_current_top2 = extract_top1_top2(pred_current_modules)
    pred_linked_modules = extract_pred_linked_modules(pred_impact)

    gold_current_modules = _normalize_name_list(sample.get("gold_current_modules", []))
    gold_current_top1 = str(sample.get("gold_current_top1", "")).strip()
    if not gold_current_top1 and gold_current_modules:
        gold_current_top1 = gold_current_modules[0]
    gold_linked_modules = _normalize_name_list(sample.get("gold_linked_modules", []))

    top1_eligible = (not weak_input) and bool(gold_current_top1)
    top2_eligible = (not weak_input) and bool(gold_current_modules)

    top1_hit = False
    if top1_eligible:
        top1_hit = _normalize_name(pred_current_top1) == _normalize_name(gold_current_top1)

    top2_recall_value = _top2_recall(gold_current_modules, pred_current_top2) if top2_eligible else 0.0

    gold_linked_set = _normalize_set(gold_linked_modules)
    pred_linked_set = _normalize_set(pred_linked_modules)
    linked_tp = len(gold_linked_set & pred_linked_set)
    linked_fp = len(pred_linked_set - gold_linked_set)
    linked_fn = len(gold_linked_set - pred_linked_set)

    triplet = validate_triplet_constraints(pred_impact)
    traceability = validate_evidence_traceability(pred_impact)

    failed_checks: List[str] = []
    if top1_eligible and not top1_hit:
        failed_checks.append("current_top1_mismatch")
    if top2_eligible and top2_recall_value < 1.0:
        failed_checks.append("current_top2_partial")
    if linked_fp > 0:
        failed_checks.append("linked_fp")
    if linked_fn > 0:
        failed_checks.append("linked_fn")
    if sample.get("expect_no_linked") and pred_linked_modules:
        failed_checks.append("unexpected_linked")
    if not triplet.get("ok", False):
        failed_checks.append("triplet_invalid")
    if not traceability.get("ok", False):
        failed_checks.append("anchor_untraceable")

    impact_summary = str(pred_impact.get("impact_summary", "")).strip()
    pred_summary = impact_summary or (
        f"current={','.join(pred_current_modules) or '-'};linked={','.join(pred_linked_modules) or '-'}"
    )

    return {
        "sample_id": sample.get("sample_id"),
        "category": category,
        "weak_input": weak_input,
        "task_query": sample.get("task_query", ""),
        "gold_current_modules": gold_current_modules,
        "gold_current_top1": gold_current_top1,
        "pred_current_modules": pred_current_modules,
        "pred_current_top1": pred_current_top1,
        "pred_current_top2": pred_current_top2,
        "gold_linked_modules": gold_linked_modules,
        "pred_linked_modules": pred_linked_modules,
        "gold_anchor_ids_primary": _normalize_name_list(sample.get("gold_anchor_ids_primary", [])),
        "pred_anchor_ids": traceability.get("anchor_ids", []),
        "triplet_ok": bool(triplet.get("ok", False)),
        "traceable_ok": bool(traceability.get("ok", False)),
        "failed_checks": failed_checks,
        "impact_summary": impact_summary,
        "pred_summary": pred_summary,
        # aggregations
        "top1_eligible": top1_eligible,
        "top1_hit": top1_hit,
        "top2_eligible": top2_eligible,
        "top2_recall": round(top2_recall_value, 6),
        "linked_pred_count": len(pred_linked_modules),
        "linked_tp": linked_tp,
        "linked_fp": linked_fp,
        "linked_fn": linked_fn,
        "triplet_valid_count": int(triplet.get("valid", 0)),
        "triplet_total_count": int(triplet.get("total", 0)),
        "traceable_valid_count": int(traceability.get("valid", 0)),
        "traceable_total_count": int(traceability.get("total", 0)),
    }


def evaluate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for sample in samples:
        pred = run_impact_v2_inference(
            task_query=str(sample.get("task_query", "")),
            retrieval_context=str(sample.get("retrieval_context", "")),
        )
        results.append(evaluate_one_sample(sample, pred))
    return results


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator <= 0:
        return default
    return numerator / denominator


def compute_overall_metrics(per_sample: List[Dict[str, Any]]) -> Dict[str, float]:
    total_samples = max(1, len(per_sample))

    top1_eligible_count = sum(1 for item in per_sample if item.get("top1_eligible"))
    top1_hit_count = sum(1 for item in per_sample if item.get("top1_eligible") and item.get("top1_hit"))
    top1_precision = _safe_div(top1_hit_count, top1_eligible_count, default=0.0)

    top2_eligible_count = sum(1 for item in per_sample if item.get("top2_eligible"))
    top2_recall_sum = sum(float(item.get("top2_recall", 0.0) or 0.0) for item in per_sample if item.get("top2_eligible"))
    top2_recall = _safe_div(top2_recall_sum, top2_eligible_count, default=0.0)

    linked_tp_total = sum(int(item.get("linked_tp", 0) or 0) for item in per_sample)
    linked_pred_total = sum(int(item.get("linked_pred_count", 0) or 0) for item in per_sample)
    linked_precision = _safe_div(linked_tp_total, linked_pred_total, default=0.0)

    triplet_valid_total = sum(int(item.get("triplet_valid_count", 0) or 0) for item in per_sample)
    triplet_total = sum(int(item.get("triplet_total_count", 0) or 0) for item in per_sample)
    triplet_constraint_rate = _safe_div(triplet_valid_total, triplet_total, default=1.0)

    traceable_valid_total = sum(int(item.get("traceable_valid_count", 0) or 0) for item in per_sample)
    traceable_total = sum(int(item.get("traceable_total_count", 0) or 0) for item in per_sample)
    evidence_traceability_rate = _safe_div(traceable_valid_total, traceable_total, default=1.0)

    avg_linked_modules = _safe_div(linked_pred_total, total_samples, default=0.0)

    return {
        "top1_precision": round(top1_precision, 6),
        "top2_recall": round(top2_recall, 6),
        "linked_precision": round(linked_precision, 6),
        "triplet_constraint_rate": round(triplet_constraint_rate, 6),
        "evidence_traceability_rate": round(evidence_traceability_rate, 6),
        "avg_linked_modules": round(avg_linked_modules, 6),
        "top1_eligible_count": int(top1_eligible_count),
        "top2_eligible_count": int(top2_eligible_count),
        "linked_pred_total": int(linked_pred_total),
    }


def compute_metrics_by_category(per_sample: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for item in per_sample:
        category = str(item.get("category", "unknown")).strip() or "unknown"
        by.setdefault(category, []).append(item)

    summary: Dict[str, Dict[str, Any]] = {}
    for category, rows in by.items():
        metrics = compute_overall_metrics(rows)
        summary[category] = {
            "count": len(rows),
            **metrics,
        }
    return summary


def evaluate_gates(metrics: Dict[str, float], thresholds: EvalThresholds) -> Dict[str, Any]:
    gates = {
        "top1_precision": float(metrics.get("top1_precision", 0.0)) >= thresholds.top1_precision,
        "top2_recall": float(metrics.get("top2_recall", 0.0)) >= thresholds.top2_recall,
        "linked_precision": float(metrics.get("linked_precision", 0.0)) >= thresholds.linked_precision,
        "triplet_constraint_rate": float(metrics.get("triplet_constraint_rate", 0.0)) >= thresholds.triplet_constraint_rate,
        "evidence_traceability_rate": float(metrics.get("evidence_traceability_rate", 0.0)) >= thresholds.evidence_traceability_rate,
        "avg_linked_modules_max": float(metrics.get("avg_linked_modules", 0.0)) <= thresholds.avg_linked_modules_max,
    }
    gates["overall_pass"] = all(gates.values())
    return gates


def build_report(
    *,
    samples_path: str,
    total_samples: int,
    thresholds: EvalThresholds,
    metrics: Dict[str, float],
    gates: Dict[str, Any],
    by_category: Dict[str, Dict[str, Any]],
    per_sample: List[Dict[str, Any]],
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    for item in per_sample:
        failed = item.get("failed_checks", [])
        if not isinstance(failed, list) or not failed:
            continue
        errors.append(
            {
                "sample_id": item.get("sample_id"),
                "category": item.get("category"),
                "task_query": item.get("task_query"),
                "gold_current_modules": item.get("gold_current_modules", []),
                "pred_current_modules": item.get("pred_current_modules", []),
                "gold_linked_modules": item.get("gold_linked_modules", []),
                "pred_linked_modules": item.get("pred_linked_modules", []),
                "gold_anchor_ids_primary": item.get("gold_anchor_ids_primary", []),
                "pred_anchor_ids": item.get("pred_anchor_ids", []),
                "triplet_ok": item.get("triplet_ok"),
                "traceable_ok": item.get("traceable_ok"),
                "failed_checks": failed,
                "impact_summary": item.get("impact_summary", ""),
                "pred_summary": item.get("pred_summary", ""),
            }
        )

    return {
        "run_meta": {
            "ts": _utc_now_iso(),
            "samples_path": samples_path,
            "total_samples": int(total_samples),
        },
        "thresholds": asdict(thresholds),
        "metrics": metrics,
        "gates": gates,
        "by_category": by_category,
        "errors": errors,
    }


def write_report_json(report: Dict[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_console_summary(report: Dict[str, Any], print_errors: int = 20) -> None:
    metrics = report.get("metrics", {}) or {}
    thresholds = report.get("thresholds", {}) or {}
    gates = report.get("gates", {}) or {}
    total = int(((report.get("run_meta", {}) or {}).get("total_samples", 0)) or 0)

    print(f"[Impact V2 Eval] samples={total}")
    print(
        f"Top1 Precision: {float(metrics.get('top1_precision', 0.0)):.2f} "
        f"(threshold {float(thresholds.get('top1_precision', 0.0)):.2f}) "
        f"{'PASS' if gates.get('top1_precision') else 'FAIL'}"
    )
    print(
        f"Top2 Recall: {float(metrics.get('top2_recall', 0.0)):.2f} "
        f"(threshold {float(thresholds.get('top2_recall', 0.0)):.2f}) "
        f"{'PASS' if gates.get('top2_recall') else 'FAIL'}"
    )
    print(
        f"Linked Precision: {float(metrics.get('linked_precision', 0.0)):.2f} "
        f"(threshold {float(thresholds.get('linked_precision', 0.0)):.2f}) "
        f"{'PASS' if gates.get('linked_precision') else 'FAIL'}"
    )
    print(
        f"Triplet Constraint Rate: {float(metrics.get('triplet_constraint_rate', 0.0)):.2f} "
        f"(threshold {float(thresholds.get('triplet_constraint_rate', 0.0)):.2f}) "
        f"{'PASS' if gates.get('triplet_constraint_rate') else 'FAIL'}"
    )
    print(
        f"Evidence Traceability Rate: {float(metrics.get('evidence_traceability_rate', 0.0)):.2f} "
        f"(threshold {float(thresholds.get('evidence_traceability_rate', 0.0)):.2f}) "
        f"{'PASS' if gates.get('evidence_traceability_rate') else 'FAIL'}"
    )
    print(
        f"Avg Linked Modules: {float(metrics.get('avg_linked_modules', 0.0)):.2f} "
        f"(max {float(thresholds.get('avg_linked_modules_max', 0.0)):.2f}) "
        f"{'PASS' if gates.get('avg_linked_modules_max') else 'FAIL'}"
    )
    print(
        f"Eligible Counts: top1={int(metrics.get('top1_eligible_count', 0))}, "
        f"top2={int(metrics.get('top2_eligible_count', 0))}, "
        f"linked_pred_total={int(metrics.get('linked_pred_total', 0))}"
    )

    overall = bool(gates.get("overall_pass", False))
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    if not overall:
        failed = [key for key, value in gates.items() if key != "overall_pass" and not value]
        if failed:
            print("Failed Gates: " + ", ".join(failed))

    errors = report.get("errors", []) or []
    if errors and print_errors > 0:
        top = errors[:print_errors]
        ids = [str(item.get("sample_id", "")).strip() for item in top if str(item.get("sample_id", "")).strip()]
        if ids:
            print("Top error samples: " + ", ".join(ids))


def decide_exit_code(gates: Dict[str, Any], strict: bool) -> int:
    if not strict:
        return 0
    return 0 if bool(gates.get("overall_pass", False)) else 1


def main() -> int:
    args = parse_args()
    thresholds = EvalThresholds()

    samples = load_samples_jsonl(args.samples)
    per_sample = evaluate_samples(samples)

    metrics = compute_overall_metrics(per_sample)
    by_category = compute_metrics_by_category(per_sample)
    gates = evaluate_gates(metrics, thresholds)

    report = build_report(
        samples_path=args.samples,
        total_samples=len(samples),
        thresholds=thresholds,
        metrics=metrics,
        gates=gates,
        by_category=by_category,
        per_sample=per_sample,
    )
    write_report_json(report, args.report)
    print_console_summary(report, print_errors=args.print_errors)
    return decide_exit_code(gates, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
