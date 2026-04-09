from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.rag.analysis.badcase_loop import (
    auto_tune_rule_templates_from_replay,
    build_badcase_replay_report,
    list_badcase_rule_template_history,
    load_badcase_rule_templates,
    prune_badcase_events,
    record_badcase_event,
    rollback_badcase_rule_templates,
    save_badcase_rule_templates,
)


class TestBadcaseLoop(unittest.TestCase):
    def test_load_templates_returns_defaults_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            templates = load_badcase_rule_templates(project_root=root)
            self.assertIsInstance(templates, dict)
            self.assertEqual(str(templates.get("version", "")), "v1")
            p0_rule = (templates.get("templates", {}) or {}).get("p0_blocking", {})
            self.assertTrue(bool(p0_rule.get("enabled", False)))

    def test_replay_report_has_alert_and_rule_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for i in range(10):
                record_badcase_event(
                    request_id=f"req_{i}",
                    task_query="订单退票接口",
                    generation_mode="business_api",
                    intent_label="api",
                    final_status="failed" if i < 7 else "success",
                    risk_report={"severity_counts": {"P0": 1 if i < 7 else 0}, "overall_level": "high"},
                    route_history=["classifier", "api_generator"],
                    recommended_mode="business_api",
                    retrieval_context_len=0,
                    project_root=root,
                )
            report = build_badcase_replay_report(
                project_root=root,
                window_days=30,
                min_sample=6,
                alert_bad_rate=0.5,
            )
            self.assertIsInstance(report, dict)
            self.assertGreaterEqual(int(report.get("event_count", 0) or 0), 10)
            self.assertTrue(report.get("alerts"))
            self.assertTrue(report.get("rule_update_hints"))

    def test_prune_badcase_events_removes_outdated_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            path = data_dir / "badcase_events.jsonl"
            old_ts = (datetime.now().astimezone() - timedelta(days=120)).isoformat(timespec="seconds")
            new_ts = datetime.now().astimezone().isoformat(timespec="seconds")
            rows = [
                {"ts": old_ts, "request_id": "old", "is_badcase": True},
                {"ts": new_ts, "request_id": "new", "is_badcase": False},
            ]
            path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
                encoding="utf-8",
            )
            summary = prune_badcase_events(project_root=root, keep_days=30, max_keep_lines=100)
            self.assertEqual(int(summary.get("before", 0) or 0), 2)
            self.assertEqual(int(summary.get("after", 0) or 0), 1)

    def test_save_templates_can_override_p0_severities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            saved = save_badcase_rule_templates(
                {
                    "version": "v2",
                    "templates": {
                        "p0_blocking": {
                            "enabled": True,
                            "severities": ["P0", "P1"],
                        }
                    },
                },
                project_root=root,
            )
            self.assertEqual(str(saved.get("version", "")), "v2")
            loaded = load_badcase_rule_templates(project_root=root)
            severities = (loaded.get("templates", {}) or {}).get("p0_blocking", {}).get("severities", [])
            self.assertIn("P1", severities)
            self.assertGreaterEqual(int(loaded.get("history_count", 0) or 0), 1)

    def test_auto_tune_should_adjust_history_template_when_bad_rate_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            save_badcase_rule_templates(
                {
                    "templates": {
                        "history_badcase": {"min_bad_rate": 0.35, "min_delta": 0.12},
                    }
                },
                project_root=root,
            )
            replay_report = {
                "event_count": 80,
                "overall_bad_rate": 0.62,
                "alerts": [{"signature": "sig1", "sample": 10, "bad_rate": 0.8}],
            }
            summary = auto_tune_rule_templates_from_replay(replay_report, project_root=root)
            self.assertTrue(bool(summary.get("applied", False)))
            loaded = load_badcase_rule_templates(project_root=root)
            history = (loaded.get("templates", {}) or {}).get("history_badcase", {})
            self.assertLess(float(history.get("min_bad_rate", 1.0)), 0.35)

    def test_rule_template_history_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            save_badcase_rule_templates(
                {
                    "version": "v2",
                    "templates": {"history_badcase": {"min_bad_rate": 0.33}},
                },
                project_root=root,
            )
            save_badcase_rule_templates(
                {
                    "version": "v3",
                    "templates": {"history_badcase": {"min_bad_rate": 0.30}},
                },
                project_root=root,
            )
            history_rows = list_badcase_rule_template_history(project_root=root, limit=5)
            self.assertGreaterEqual(len(history_rows), 1)

            rollback = rollback_badcase_rule_templates(project_root=root)
            self.assertTrue(bool(rollback.get("applied", False)))
            loaded = load_badcase_rule_templates(project_root=root)
            self.assertIn(str(loaded.get("version", "")), {"v1", "v2", "v3"})
            current_rule = (loaded.get("templates", {}) or {}).get("history_badcase", {})
            self.assertGreater(float(current_rule.get("min_bad_rate", 0.0) or 0.0), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
