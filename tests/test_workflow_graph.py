from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import workflow_graph
from workflow_graph import run_testcase_workflow


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _SmartFakeLLM:
    def invoke(self, prompt: str) -> _Resp:
        text = str(prompt or "")
        if "测试需求路由器" in text or "意图路由器" in text:
            return _Resp('{"intent_label":"api","confidence":0.92,"reason":"api_logic"}')
        if "测试用例评审官" in text or "LLM Judge" in text:
            return _Resp(
                '{"decision":"pass","hard_fail_reasons":[],"scores":{"business_coverage":5,"exception_coverage":4,"assertion_specificity":4,"executability":4,"traceability":4,"redundancy_control":4,"total":25},"comments":[],"missing_points":[],"rewrite_instructions":[]}'
            )
        return _Resp("## 业务测试用例\n- 场景A")


class _AdaptiveModeFakeLLM:
    def __init__(self) -> None:
        self.review_round = 0
        self.generator_prompts: list[str] = []

    def invoke(self, prompt: str) -> _Resp:
        text = str(prompt or "")
        if "测试需求路由器" in text or "意图路由器" in text:
            return _Resp('{"intent_label":"api","confidence":0.95,"reason":"api_first"}')
        if "测试用例评审官" in text or "LLM Judge" in text:
            self.review_round += 1
            if self.review_round == 1:
                return _Resp(
                    '{"decision":"fail","hard_fail_reasons":[],"scores":{"business_coverage":3,"exception_coverage":3,"assertion_specificity":2,"executability":3,"traceability":3,"redundancy_control":3,"total":17},"comments":["字段覆盖不足"],"missing_points":["缺少字段类型与边界校验"],"rewrite_instructions":["补充字段必填/类型/长度/格式/枚举断言"]}'
                )
            return _Resp(
                '{"decision":"pass","hard_fail_reasons":[],"scores":{"business_coverage":4,"exception_coverage":4,"assertion_specificity":4,"executability":4,"traceability":4,"redundancy_control":4,"total":24},"comments":[],"missing_points":[],"rewrite_instructions":[]}'
            )

        self.generator_prompts.append(text)
        return _Resp(f"## 业务测试用例\n- 轮次{len(self.generator_prompts)}")


class _HumanInputCaptureLLM:
    def __init__(self) -> None:
        self.captured_prompts: list[str] = []

    def invoke(self, prompt: str) -> _Resp:
        text = str(prompt or "")
        if "测试需求路由器" in text or "意图路由器" in text:
            return _Resp('{"intent_label":"api","confidence":0.91,"reason":"api_logic"}')
        if "测试用例评审官" in text or "LLM Judge" in text:
            return _Resp(
                '{"decision":"pass","hard_fail_reasons":[],"scores":{"business_coverage":4,"exception_coverage":4,"assertion_specificity":4,"executability":4,"traceability":4,"redundancy_control":4,"total":24},"comments":[],"missing_points":[],"rewrite_instructions":[]}'
            )
        self.captured_prompts.append(text)
        return _Resp("## 业务测试用例\n- 场景A")


class _ConstraintViolationLLM:
    def __init__(self) -> None:
        self.generator_prompts: list[str] = []

    def invoke(self, prompt: str) -> _Resp:
        text = str(prompt or "")
        if "测试需求路由器" in text or "意图路由器" in text:
            return _Resp('{"intent_label":"api","confidence":0.93,"reason":"api_logic"}')
        if "测试用例评审官" in text or "LLM Judge" in text:
            return _Resp(
                '{"decision":"pass","hard_fail_reasons":[],"scores":{"business_coverage":5,"exception_coverage":5,"assertion_specificity":5,"executability":5,"traceability":5,"redundancy_control":5,"total":30},"comments":[],"missing_points":[],"rewrite_instructions":[]}'
            )
        self.generator_prompts.append(text)
        return _Resp("## 业务测试用例\n- 覆盖退款清算链路")


class TestWorkflowGraph(unittest.TestCase):
    def test_retrieval_policy_is_forwarded(self) -> None:
        captured = {"policy": None}

        def fake_context(query: str, retrieval_policy=None) -> str:
            captured["policy"] = retrieval_policy
            return "上下文"

        result = run_testcase_workflow(
            task_query="验证退票接口跨模块业务流程",
            get_augmented_context=fake_context,
            llm=_SmartFakeLLM(),
            universal_template="context:{context}\ntask:{task}",
            generation_mode="business_api",
            retrieval_policy={"approved_only": True, "release": "v1", "modules": ["订单"]},
            max_iterations=1,
        )

        self.assertIsInstance(result, dict)
        self.assertTrue(bool(captured["policy"]))
        self.assertEqual(captured["policy"].get("release"), "v1")
        self.assertIn("final_status", result)

    def test_workflow_does_not_crash_when_context_fails(self) -> None:
        def bad_context(_: str, retrieval_policy=None) -> str:
            raise RuntimeError("retrieval down")

        result = run_testcase_workflow(
            task_query="验证支付接口",
            get_augmented_context=bad_context,
            llm=_SmartFakeLLM(),
            universal_template="context:{context}\ntask:{task}",
            generation_mode="business_api",
            max_iterations=1,
        )

        self.assertIsInstance(result, dict)
        self.assertIn(result.get("final_status"), {"failed", "success_with_warning", "success"})
        self.assertIn("error", result)

    def test_observation_log_failure_does_not_break_workflow(self) -> None:
        def fake_context(_: str, retrieval_policy=None) -> str:
            return "上下文"

        with patch.object(workflow_graph.Path, "open", side_effect=OSError("write fail")):
            result = run_testcase_workflow(
                task_query="验证支付接口",
                get_augmented_context=fake_context,
                llm=_SmartFakeLLM(),
                universal_template="context:{context}\ntask:{task}",
                generation_mode="business_api",
                max_iterations=1,
            )

        self.assertIsInstance(result, dict)
        self.assertIn("final_status", result)

    def test_workflow_outputs_impact_analysis_v2(self) -> None:
        context = "\n".join(
            [
                "### 需求文档",
                "- 片段1 | doc_key=requirement|file_upload|req.md | chunk_index=1 | source_type=requirement | source=req.md | origin=file_upload | modules=订单 | downstream=库存/权益 | feature=f1 | trace=t1",
                "订单状态变化需要关注联动影响",
                "### API接口文档",
                "- 片段1 | doc_key=api_doc|file_upload|api.md | chunk_index=2 | source_type=api_doc | source=api.md | origin=file_upload | modules=库存/权益 | upstream=订单 | trace=t1",
                "库存扣减接口",
            ]
        )

        def fake_context(_: str, retrieval_policy=None) -> str:
            return context

        result = run_testcase_workflow(
            task_query="订单权限调整，需要评估联动影响",
            get_augmented_context=fake_context,
            llm=_SmartFakeLLM(),
            universal_template="context:{context}\ntask:{task}",
            generation_mode="business_api",
            max_iterations=1,
        )
        impact = result.get("impact_analysis", {})
        self.assertIsInstance(impact, dict)
        self.assertEqual(impact.get("version"), "v2")
        self.assertIn("current_involved_modules", impact)
        self.assertIn("potential_linked_modules", impact)
        self.assertIsInstance(result.get("link_edges"), list)
        self.assertIsInstance(result.get("trace_refs"), dict)
        self.assertIsInstance(result.get("link_summary"), dict)
        self.assertGreaterEqual(len(result.get("link_edges", [])), 1)
        risk_report = result.get("risk_report", {})
        self.assertIsInstance(risk_report, dict)
        self.assertIn("overall_level", risk_report)
        self.assertIsInstance(risk_report.get("items", []), list)
        self.assertIsInstance(result.get("badcase_replay", {}), dict)
        self.assertIsInstance(result.get("contracts"), dict)
        self.assertIsInstance(result.get("mapping_rules"), dict)
        self.assertIsInstance(result.get("coverage_matrix"), dict)

    def test_risk_report_history_adjustment_boosts_score_on_high_badcase_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_dir = root / "data"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "observation_log.jsonl"

            rows = []
            for idx in range(60):
                rows.append(
                    {
                        "request_id": f"r_bad_{idx}",
                        "final_status": "failed" if idx < 48 else "success",
                        "context_len": 0,
                        "weak_input": False,
                        "gap_hints_hit": True,
                        "impact_high_count": 0,
                        "impact_medium_count": 1,
                    }
                )
            for idx in range(40):
                rows.append(
                    {
                        "request_id": f"r_good_{idx}",
                        "final_status": "success",
                        "context_len": 1200,
                        "weak_input": False,
                        "gap_hints_hit": False,
                        "impact_high_count": 0,
                        "impact_medium_count": 0,
                    }
                )
            log_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8",
            )

            with patch.object(workflow_graph, "PROJECT_ROOT", root):
                report = workflow_graph._build_risk_report(
                    retrieval_context_len=0,
                    compliance_report={"pass": True},
                    gap_hints={"coverage_risks": ["异常分支覆盖不足"]},
                    impact_analysis={},
                    link_summary={"total_edges": 0},
                    trace_refs={"req_ids": [], "api_ids": [], "testcase_ids": []},
                )

        self.assertIsInstance(report, dict)
        history_adj = report.get("history_adjustment", {})
        self.assertIsInstance(history_adj, dict)
        self.assertTrue(bool(history_adj.get("enabled", False)))
        self.assertGreater(float(history_adj.get("score_boost", 0.0) or 0.0), 0.0)
        self.assertGreaterEqual(
            float(report.get("overall_score", 0.0) or 0.0),
            float(history_adj.get("base_score", 0.0) or 0.0),
        )

    def test_recommended_mode_affects_retry_generation_prompt(self) -> None:
        if not workflow_graph.LANGGRAPH_AVAILABLE:
            self.skipTest("LangGraph unavailable")

        llm = _AdaptiveModeFakeLLM()

        def fake_context(_: str, retrieval_policy=None) -> str:
            return "订单退票场景，需要覆盖字段约束与状态流转。"

        result = run_testcase_workflow(
            task_query="请补齐退票接口测试覆盖",
            get_augmented_context=fake_context,
            llm=llm,
            universal_template="context:{context}\ntask:{task}",
            generation_mode="business_api",
            max_iterations=1,
        )

        mode_prompts = [
            text
            for text in llm.generator_prompts
            if "【本次生成模式追加指令（高优先级）】" in text
        ]
        self.assertGreaterEqual(len(mode_prompts), 2)
        self.assertIn("当前模式: 业务接口用例", mode_prompts[0])
        self.assertIn("当前模式: 字段校验用例", mode_prompts[1])
        self.assertEqual(result.get("recommended_mode"), "field_validation")

    def test_recommended_mode_keeps_current_when_signals_weak(self) -> None:
        review_result = {
            "scores": {
                "business_coverage": 4,
                "exception_coverage": 4,
                "assertion_specificity": 4,
                "executability": 4,
                "traceability": 4,
                "redundancy_control": 4,
            },
            "comments": ["建议进一步检查字段一致性"],
            "missing_points": [],
            "rewrite_instructions": [],
            "hard_fail_reasons": [],
        }
        recommended = workflow_graph._recommend_generation_mode("business_api", review_result)
        self.assertEqual(recommended, "business_api")

    def test_recommended_mode_lock_prevents_switching(self) -> None:
        if not workflow_graph.LANGGRAPH_AVAILABLE:
            self.skipTest("LangGraph unavailable")

        llm = _AdaptiveModeFakeLLM()

        def fake_context(_: str, retrieval_policy=None) -> str:
            return "订单退票场景，需要覆盖字段约束与状态流转。"

        result = run_testcase_workflow(
            task_query="请补齐退票接口测试覆盖",
            get_augmented_context=fake_context,
            llm=llm,
            universal_template="context:{context}\ntask:{task}",
            generation_mode="business_api",
            recommended_mode_lock="business_api",
            max_iterations=1,
        )

        mode_prompts = [
            text
            for text in llm.generator_prompts
            if "【本次生成模式追加指令（高优先级）】" in text
        ]
        self.assertGreaterEqual(len(mode_prompts), 2)
        self.assertIn("当前模式: 业务接口用例", mode_prompts[0])
        self.assertIn("当前模式: 业务接口用例", mode_prompts[1])
        self.assertEqual(result.get("recommended_mode"), "business_api")
        self.assertEqual(result.get("recommended_mode_lock"), "business_api")

    def test_human_inputs_are_forwarded_and_affect_prompt(self) -> None:
        llm = _HumanInputCaptureLLM()

        def fake_context(_: str, retrieval_policy=None) -> str:
            return "上下文"

        result = run_testcase_workflow(
            task_query="验证退票接口",
            get_augmented_context=fake_context,
            llm=llm,
            universal_template="context:{context}\ntask:{task}",
            generation_mode="business_api",
            human_inputs={
                "scope": "仅退票主流程",
                "exclusions": "不含退款清算",
                "risk_focus": "状态幂等与库存一致性",
                "priority_modules": ["订单", "库存"],
                "release": "v2026.04",
                "approved_only": True,
                "run_context": {
                    "confirmed_by_user": True,
                    "confirmation_ts": "2026-04-08T10:00:00+08:00",
                },
            },
            max_iterations=1,
        )

        self.assertTrue(result.get("run_context", {}).get("confirmed_by_user"))
        self.assertEqual(result.get("human_inputs", {}).get("scope"), "仅退票主流程")
        mode_prompts = [
            text
            for text in llm.captured_prompts
            if "【本次生成模式追加指令（高优先级）】" in text
        ]
        self.assertTrue(mode_prompts)
        self.assertIn("【人机协作前置输入（高优先级）】", mode_prompts[0])
        self.assertIn("仅退票主流程", mode_prompts[0])
        self.assertTrue(result.get("human_inputs", {}).get("must_cover"))
        self.assertTrue(result.get("human_inputs", {}).get("risk_tags"))

    def test_constraint_validator_fails_and_feedback_is_used_in_retry(self) -> None:
        if not workflow_graph.LANGGRAPH_AVAILABLE:
            self.skipTest("LangGraph unavailable")

        llm = _ConstraintViolationLLM()

        def fake_context(_: str, retrieval_policy=None) -> str:
            return "上下文"

        result = run_testcase_workflow(
            task_query="验证退票接口",
            get_augmented_context=fake_context,
            llm=llm,
            universal_template="context:{context}\ntask:{task}",
            generation_mode="business_api",
            human_inputs={
                "scope": "退票主流程",
                "exclusions": "退款清算",
                "risk_focus": "库存一致性",
                "priority_modules": ["订单"],
                "run_context": {"confirmed_by_user": True},
            },
            max_iterations=1,
        )

        self.assertIn("constraint_validator", result.get("route_history", []))
        self.assertFalse(bool(result.get("compliance_report", {}).get("pass", True)))
        self.assertEqual(result.get("review_result", {}).get("decision"), "fail")
        mode_prompts = [
            text
            for text in llm.generator_prompts
            if "【本次生成模式追加指令（高优先级）】" in text
        ]
        self.assertGreaterEqual(len(mode_prompts), 2)
        self.assertIn("【上轮评审反馈（必须修复）】", mode_prompts[1])
        self.assertIn("排除项", mode_prompts[1])
        risk_report = result.get("risk_report", {})
        self.assertIsInstance(risk_report, dict)
        risk_items = risk_report.get("items", [])
        self.assertIsInstance(risk_items, list)
        has_constraint_violation = any(
            isinstance(item, dict)
            and str(item.get("category", "")) == "constraint_violation"
            and str(item.get("severity", "")).upper() in {"P0", "P1"}
            for item in risk_items
        )
        self.assertTrue(has_constraint_violation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
