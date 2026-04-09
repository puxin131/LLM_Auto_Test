from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_RULE_TEMPLATES: Dict[str, Any] = {
    "version": "v1",
    "updated_at": "",
    "templates": {
        "history_badcase": {
            "enabled": True,
            "min_sample": 12,
            "min_bad_rate": 0.35,
            "min_delta": 0.12,
            "delta_scale": 24.0,
            "max_boost": 24,
        },
        "p0_blocking": {
            "enabled": True,
            "severities": ["P0"],
        },
    },
}


def _utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalized_root(project_root: Path | str | None) -> Path:
    if isinstance(project_root, Path):
        return project_root
    if isinstance(project_root, str) and project_root.strip():
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[3]


def _events_path(project_root: Path | str | None = None) -> Path:
    root = _normalized_root(project_root)
    return root / "data" / "badcase_events.jsonl"


def _templates_path(project_root: Path | str | None = None) -> Path:
    root = _normalized_root(project_root)
    return root / "data" / "risk_rule_templates.json"


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _deepcopy_template(value: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _normalize_history_rows(values: Any, *, max_items: int = 80) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        templates = item.get("templates", {})
        if not isinstance(templates, dict):
            continue
        out.append(
            {
                "ts": str(item.get("ts", "")).strip(),
                "version": str(item.get("version", "")).strip() or "v1",
                "updated_at": str(item.get("updated_at", "")).strip(),
                "templates": _deepcopy_template(templates),
            }
        )
    if len(out) > max_items:
        out = out[-max_items:]
    return out


def load_badcase_rule_templates(project_root: Path | str | None = None) -> Dict[str, Any]:
    path = _templates_path(project_root)
    defaults = _deepcopy_template(DEFAULT_RULE_TEMPLATES)
    defaults["history_count"] = 0
    try:
        if not path.exists():
            return defaults
        payload = _safe_json_loads(path.read_text(encoding="utf-8"))
        if not payload:
            return defaults
        templates = payload.get("templates", {})
        if not isinstance(templates, dict):
            templates = {}
        history = _normalize_history_rows(payload.get("history", []), max_items=80)
        merged = _deepcopy_template(DEFAULT_RULE_TEMPLATES)
        merged["version"] = str(payload.get("version", merged.get("version", "v1"))).strip() or "v1"
        merged["updated_at"] = str(payload.get("updated_at", "")).strip()
        merged["history_count"] = len(history)
        for key in ("history_badcase", "p0_blocking"):
            if isinstance(templates.get(key), dict):
                merged["templates"][key].update(templates[key])
        return merged
    except Exception:
        return defaults


def save_badcase_rule_templates(
    payload: Dict[str, Any],
    project_root: Path | str | None = None,
) -> Dict[str, Any]:
    templates = load_badcase_rule_templates(project_root)
    old_templates = _deepcopy_template(templates.get("templates", {}))
    old_version = str(templates.get("version", "v1")).strip() or "v1"
    old_updated_at = str(templates.get("updated_at", "")).strip()
    incoming = payload if isinstance(payload, dict) else {}
    incoming_templates = incoming.get("templates", {})
    if not isinstance(incoming_templates, dict):
        incoming_templates = {}
    for key in ("history_badcase", "p0_blocking"):
        if isinstance(incoming_templates.get(key), dict):
            templates["templates"][key].update(incoming_templates[key])
    if str(incoming.get("version", "")).strip():
        templates["version"] = str(incoming.get("version")).strip()
    new_templates = _deepcopy_template(templates.get("templates", {}))
    new_version = str(templates.get("version", old_version)).strip() or old_version
    changed = (new_templates != old_templates) or (new_version != old_version)
    if not changed:
        templates["history_count"] = int(templates.get("history_count", 0) or 0)
        return templates

    path = _templates_path(project_root)
    raw_payload = _safe_json_loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    history = _normalize_history_rows(raw_payload.get("history", []), max_items=80)
    history.append(
        {
            "ts": _utc_now_iso(),
            "version": old_version,
            "updated_at": old_updated_at,
            "templates": old_templates,
        }
    )
    history = _normalize_history_rows(history, max_items=80)
    templates["updated_at"] = _utc_now_iso()
    templates["history_count"] = len(history)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": str(templates.get("version", "v1")),
                "updated_at": str(templates.get("updated_at", "")),
                "templates": templates.get("templates", {}),
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return templates


def list_badcase_rule_template_history(
    project_root: Path | str | None = None,
    *,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    path = _templates_path(project_root)
    if not path.exists():
        return []
    payload = _safe_json_loads(path.read_text(encoding="utf-8"))
    history = _normalize_history_rows(payload.get("history", []), max_items=200)
    if limit <= 0:
        return list(reversed(history))
    return list(reversed(history[-limit:]))


def rollback_badcase_rule_templates(
    project_root: Path | str | None = None,
) -> Dict[str, Any]:
    path = _templates_path(project_root)
    if not path.exists():
        return {"applied": False, "reason": "template_file_not_found", "history_count": 0}
    payload = _safe_json_loads(path.read_text(encoding="utf-8"))
    if not payload:
        return {"applied": False, "reason": "template_payload_invalid", "history_count": 0}

    history = _normalize_history_rows(payload.get("history", []), max_items=200)
    if not history:
        return {"applied": False, "reason": "history_empty", "history_count": 0}

    target = history[-1]
    remain = history[:-1]
    restored_templates = target.get("templates", {})
    if not isinstance(restored_templates, dict):
        return {"applied": False, "reason": "history_target_invalid", "history_count": len(remain)}

    next_payload = {
        "version": str(target.get("version", "v1")).strip() or "v1",
        "updated_at": _utc_now_iso(),
        "templates": restored_templates,
        "history": remain,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(next_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "applied": True,
        "to_version": str(next_payload.get("version", "v1")),
        "history_count": len(remain),
    }


def _read_recent_rows(path: Path, *, max_lines: int = 600, max_bytes: int = 512 * 1024) -> List[Dict[str, Any]]:
    try:
        if not path.exists():
            return []
        with path.open("rb") as fp:
            fp.seek(0, 2)
            file_size = fp.tell()
            if file_size <= 0:
                return []
            read_size = min(max_bytes, file_size)
            fp.seek(-read_size, 2)
            chunk = fp.read().decode("utf-8", errors="ignore")
        lines = chunk.splitlines()
        if read_size < file_size and lines:
            lines = lines[1:]
        rows: List[Dict[str, Any]] = []
        for line in lines[-max_lines:]:
            row = _safe_json_loads(line)
            if row:
                rows.append(row)
        return rows
    except Exception:
        return []


def _normalize_signature(task_query: str, generation_mode: str, intent_label: str) -> str:
    text = re.sub(r"\s+", " ", str(task_query or "")).strip().lower()
    if not text:
        head = "-"
    else:
        tokens = [tok for tok in re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", text) if tok]
        head = "_".join(tokens[:12]) if tokens else text[:36]
    return f"{str(generation_mode or '-').strip()}|{str(intent_label or '-').strip()}|{head[:72]}"


def record_badcase_event(
    *,
    request_id: str,
    task_query: str,
    generation_mode: str,
    intent_label: str,
    final_status: str,
    risk_report: Dict[str, Any] | None = None,
    route_history: List[str] | None = None,
    recommended_mode: str = "",
    retrieval_context_len: int = 0,
    project_root: Path | str | None = None,
) -> Dict[str, Any]:
    report = risk_report if isinstance(risk_report, dict) else {}
    severity_counts = report.get("severity_counts", {})
    if not isinstance(severity_counts, dict):
        severity_counts = {}
    p0_count = int(severity_counts.get("P0", 0) or 0)

    status = str(final_status or "").strip().lower()
    is_bad = status in {"failed", "success_with_warning"} or p0_count > 0

    tags: List[str] = []
    if int(retrieval_context_len or 0) <= 0:
        tags.append("context_empty")
    if p0_count > 0:
        tags.append("risk_p0")
    if status == "failed":
        tags.append("failed")
    if status == "success_with_warning":
        tags.append("warning")

    signature = _normalize_signature(task_query, generation_mode, intent_label)
    event = {
        "ts": _utc_now_iso(),
        "request_id": str(request_id or "").strip(),
        "signature": signature,
        "generation_mode": str(generation_mode or "").strip(),
        "recommended_mode": str(recommended_mode or "").strip(),
        "intent_label": str(intent_label or "").strip(),
        "final_status": status or "unknown",
        "is_badcase": bool(is_bad),
        "p0_count": p0_count,
        "risk_level": str(report.get("overall_level", "")).strip(),
        "risk_score": float(report.get("overall_score", 0.0) or 0.0),
        "retrieval_context_len": int(retrieval_context_len or 0),
        "tags": tags,
        "route_history": [str(x) for x in (route_history or []) if str(x).strip()][:12],
    }

    path = _events_path(project_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        return event
    return event


def build_badcase_replay_report(
    *,
    project_root: Path | str | None = None,
    window_days: int = 30,
    max_events: int = 600,
    min_sample: int = 6,
    alert_bad_rate: float = 0.45,
) -> Dict[str, Any]:
    rows = _read_recent_rows(_events_path(project_root), max_lines=max_events)
    now = datetime.now().astimezone()
    if window_days > 0:
        cutoff = now - timedelta(days=int(window_days))
    else:
        cutoff = None

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        ts = str(row.get("ts", "")).strip()
        if cutoff and ts:
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                dt = None
            if dt and dt < cutoff:
                continue
        filtered.append(row)

    total = len(filtered)
    bad = sum(1 for row in filtered if bool(row.get("is_badcase", False)))
    overall_bad_rate = round((bad / float(total)) if total > 0 else 0.0, 4)

    by_signature: Dict[str, Dict[str, Any]] = {}
    tag_stats: Dict[str, Dict[str, Any]] = {}
    for row in filtered:
        signature = str(row.get("signature", "")).strip() or "-"
        bucket = by_signature.setdefault(
            signature,
            {
                "signature": signature,
                "sample": 0,
                "bad": 0,
                "bad_rate": 0.0,
                "latest_ts": "",
                "generation_mode": str(row.get("generation_mode", "")),
            },
        )
        bucket["sample"] += 1
        if bool(row.get("is_badcase", False)):
            bucket["bad"] += 1
        ts = str(row.get("ts", "")).strip()
        if ts and ts > str(bucket.get("latest_ts", "")):
            bucket["latest_ts"] = ts

        tags = row.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags]
        for raw_tag in tags:
            tag = str(raw_tag or "").strip()
            if not tag:
                continue
            stats = tag_stats.setdefault(tag, {"tag": tag, "sample": 0, "bad": 0, "bad_rate": 0.0})
            stats["sample"] += 1
            if bool(row.get("is_badcase", False)):
                stats["bad"] += 1

    signature_rows = []
    for row in by_signature.values():
        sample = int(row.get("sample", 0) or 0)
        bad_count = int(row.get("bad", 0) or 0)
        row["bad_rate"] = round((bad_count / float(sample)) if sample > 0 else 0.0, 4)
        signature_rows.append(row)
    signature_rows.sort(
        key=lambda item: (
            -float(item.get("bad_rate", 0.0) or 0.0),
            -int(item.get("sample", 0) or 0),
            str(item.get("signature", "")),
        )
    )

    tag_rows = []
    for row in tag_stats.values():
        sample = int(row.get("sample", 0) or 0)
        bad_count = int(row.get("bad", 0) or 0)
        row["bad_rate"] = round((bad_count / float(sample)) if sample > 0 else 0.0, 4)
        tag_rows.append(row)
    tag_rows.sort(
        key=lambda item: (
            -float(item.get("bad_rate", 0.0) or 0.0),
            -int(item.get("sample", 0) or 0),
            str(item.get("tag", "")),
        )
    )

    alerts = [
        row
        for row in signature_rows
        if int(row.get("sample", 0) or 0) >= int(min_sample)
        and float(row.get("bad_rate", 0.0) or 0.0) >= float(alert_bad_rate)
    ][:10]

    rule_update_hints: List[Dict[str, Any]] = []
    for row in tag_rows[:8]:
        sample = int(row.get("sample", 0) or 0)
        bad_rate = float(row.get("bad_rate", 0.0) or 0.0)
        if sample < max(8, int(min_sample)) or bad_rate < max(0.5, float(alert_bad_rate)):
            continue
        tag = str(row.get("tag", ""))
        if tag in {"context_empty", "warning", "failed"}:
            rule_update_hints.append(
                {
                    "template": "history_badcase",
                    "reason_tag": tag,
                    "suggestion": "提高该标签权重或降低触发阈值，优先暴露高风险场景。",
                    "sample": sample,
                    "bad_rate": bad_rate,
                }
            )

    rule_templates = load_badcase_rule_templates(project_root=project_root)
    return {
        "version": "v1",
        "window_days": int(window_days),
        "event_count": total,
        "badcase_count": bad,
        "overall_bad_rate": overall_bad_rate,
        "rule_template": {
            "version": str(rule_templates.get("version", "v1")).strip() or "v1",
            "updated_at": str(rule_templates.get("updated_at", "")).strip(),
            "history_count": int(rule_templates.get("history_count", 0) or 0),
        },
        "signature_stats": signature_rows[:20],
        "tag_stats": tag_rows[:20],
        "alerts": alerts,
        "rule_update_hints": rule_update_hints[:10],
    }


def auto_tune_rule_templates_from_replay(
    replay_report: Dict[str, Any],
    *,
    project_root: Path | str | None = None,
) -> Dict[str, Any]:
    report = replay_report if isinstance(replay_report, dict) else {}
    event_count = int(report.get("event_count", 0) or 0)
    overall_bad_rate = float(report.get("overall_bad_rate", 0.0) or 0.0)
    alerts = report.get("alerts", [])
    if not isinstance(alerts, list):
        alerts = []

    templates = load_badcase_rule_templates(project_root=project_root)
    history_rule = (templates.get("templates", {}) or {}).get("history_badcase", {})
    if not isinstance(history_rule, dict):
        return {"applied": False, "changes": [], "reason": "history_rule_missing"}

    changes: List[Dict[str, Any]] = []
    if event_count >= 40 and overall_bad_rate >= 0.45 and alerts:
        old_bad_rate = float(history_rule.get("min_bad_rate", 0.35) or 0.35)
        old_delta = float(history_rule.get("min_delta", 0.12) or 0.12)
        new_bad_rate = max(0.28, round(old_bad_rate - 0.02, 4))
        new_delta = max(0.08, round(old_delta - 0.01, 4))
        if new_bad_rate != old_bad_rate:
            history_rule["min_bad_rate"] = new_bad_rate
            changes.append({"field": "min_bad_rate", "old": old_bad_rate, "new": new_bad_rate})
        if new_delta != old_delta:
            history_rule["min_delta"] = new_delta
            changes.append({"field": "min_delta", "old": old_delta, "new": new_delta})
    elif event_count >= 80 and overall_bad_rate <= 0.18 and not alerts:
        old_bad_rate = float(history_rule.get("min_bad_rate", 0.35) or 0.35)
        old_delta = float(history_rule.get("min_delta", 0.12) or 0.12)
        new_bad_rate = min(0.5, round(old_bad_rate + 0.01, 4))
        new_delta = min(0.2, round(old_delta + 0.005, 4))
        if new_bad_rate != old_bad_rate:
            history_rule["min_bad_rate"] = new_bad_rate
            changes.append({"field": "min_bad_rate", "old": old_bad_rate, "new": new_bad_rate})
        if new_delta != old_delta:
            history_rule["min_delta"] = new_delta
            changes.append({"field": "min_delta", "old": old_delta, "new": new_delta})

    if not changes:
        return {
            "applied": False,
            "changes": [],
            "reason": "no_adjustment_needed",
            "event_count": event_count,
            "overall_bad_rate": overall_bad_rate,
        }

    saved = save_badcase_rule_templates(
        {
            "templates": {
                "history_badcase": history_rule,
            }
        },
        project_root=project_root,
    )
    return {
        "applied": True,
        "changes": changes,
        "event_count": event_count,
        "overall_bad_rate": overall_bad_rate,
        "version": str(saved.get("version", "v1")),
        "updated_at": str(saved.get("updated_at", "")),
    }


def prune_badcase_events(
    *,
    project_root: Path | str | None = None,
    keep_days: int = 90,
    max_keep_lines: int = 4000,
) -> Dict[str, Any]:
    path = _events_path(project_root)
    rows = _read_recent_rows(path, max_lines=max(max_keep_lines * 2, 2000), max_bytes=2 * 1024 * 1024)
    before = len(rows)
    if before == 0:
        return {"before": 0, "after": 0, "removed": 0}

    now = datetime.now().astimezone()
    cutoff = now - timedelta(days=max(1, int(keep_days)))
    kept: List[Dict[str, Any]] = []
    for row in rows:
        ts = str(row.get("ts", "")).strip()
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        if dt >= cutoff:
            kept.append(row)

    if len(kept) > int(max_keep_lines):
        kept = kept[-int(max_keep_lines) :]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for row in kept:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    after = len(kept)
    return {"before": before, "after": after, "removed": max(0, before - after)}
