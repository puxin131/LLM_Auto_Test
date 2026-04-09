from __future__ import annotations

import unittest

from src.rag.analysis.constraint_compliance import (
    build_constraint_dsl,
    evaluate_constraint_compliance,
)


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _FailingLlmReview:
    def invoke(self, prompt: str) -> _Resp:
        _ = prompt
        return _Resp(
            '{"pass":false,"score":20,"reasons":["LLM判定覆盖不足"],'
            '"rule_assertions":[{"id":"include_all:1","passed":false,"score":0.3,"evidence_ids":["L1"]}]}'
        )


class TestConstraintCompliance(unittest.TestCase):
    def test_build_constraint_dsl_from_human_inputs(self) -> None:
        dsl = build_constraint_dsl(
            {
                "must_cover": ["订单", "库存"],
                "must_not_cover": ["退款清算"],
                "risk_tags": ["并发", "一致性"],
            }
        )
        self.assertEqual(dsl.get("dsl_version"), "1.0")
        self.assertIn("订单", dsl.get("include_all", []))
        self.assertIn("退款清算", dsl.get("exclude_any", []))
        self.assertIn("一致性", dsl.get("risk_focus", []))

    def test_evaluate_constraint_passes_with_hits(self) -> None:
        report = evaluate_constraint_compliance(
            draft_md="## 用例\n- 验证订单与库存一致性\n- 并发下库存数据一致",
            human_inputs={
                "must_cover": ["订单", "库存"],
                "must_not_cover": ["退款清算"],
                "risk_tags": ["一致性", "并发"],
            },
        )
        self.assertTrue(bool(report.get("pass", False)))
        self.assertGreater(float(report.get("score", 0.0)), 60.0)
        self.assertTrue(report.get("constraints"))

    def test_evaluate_constraint_fails_on_exclusion_and_missing(self) -> None:
        report = evaluate_constraint_compliance(
            draft_md="## 用例\n- 覆盖退款清算链路",
            human_inputs={
                "must_cover": ["订单"],
                "must_not_cover": ["退款清算"],
                "risk_tags": ["一致性"],
            },
        )
        self.assertFalse(bool(report.get("pass", True)))
        missing = report.get("missing_items", {})
        self.assertIn("订单", missing.get("include_all", []))
        self.assertIn("退款清算", missing.get("must_not_cover_hits", []))
        self.assertTrue(report.get("rewrite_instructions"))

    def test_semantic_match_should_hit_similar_phrase(self) -> None:
        report = evaluate_constraint_compliance(
            draft_md="库存数据保持一致，避免串单。",
            human_inputs={
                "must_cover": ["库存一致性"],
                "must_not_cover": [],
                "risk_tags": [],
            },
        )
        constraints = report.get("constraints", [])
        include_rule = [r for r in constraints if str(r.get("category")) == "include_all"]
        self.assertTrue(include_rule)
        self.assertTrue(bool(include_rule[0].get("passed", False)))

    def test_llm_review_channel_can_block_pass(self) -> None:
        report = evaluate_constraint_compliance(
            draft_md="订单流程覆盖。",
            human_inputs={"must_cover": ["订单"], "must_not_cover": [], "risk_tags": []},
            llm=_FailingLlmReview(),
        )
        self.assertTrue(bool(report.get("llm_review", {}).get("used", False)))
        self.assertFalse(bool(report.get("pass", True)))


if __name__ == "__main__":
    unittest.main(verbosity=2)

