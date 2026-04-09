from __future__ import annotations

import unittest

from src.rag.analysis.attribution_engine import (
    build_current_involved_modules,
    canonicalize_module,
)
from src.rag.analysis.evidence_anchor import build_evidence_anchors
from src.rag.analysis.impact_engine import build_impact_analysis_v2


class TestAnalysisEngines(unittest.TestCase):
    def _sample_context(self) -> str:
        return "\n".join(
            [
                "## 召回来源清单",
                "- [requirement] req.md (file_upload) | doc_key=requirement|file_upload|req.md | feature=manual_order_split | trace=T-1001",
                "- [api_doc] api.md (file_upload) | doc_key=api_doc|file_upload|api.md | trace=T-1001",
                "",
                "### 需求文档",
                "- 片段1 | doc_key=requirement|file_upload|req.md | chunk_index=1 | source_type=requirement | source=req.md | origin=file_upload | modules=手动下单,订单 | domain=票务后台 | downstream=库存/权益 | feature=manual_order_split | trace=T-1001 | related=api_doc|file_upload|api.md",
                "手动下单在订单创建后，需要与库存系统保持一致性。",
                "",
                "### API接口文档",
                "- 片段1 | doc_key=api_doc|file_upload|api.md | chunk_index=2 | source_type=api_doc | source=api.md | origin=file_upload | modules=库存/权益 | upstream=订单 | trace=T-1001",
                "库存扣减接口在下单后触发。",
            ]
        )

    def test_build_evidence_anchors_has_required_fields(self) -> None:
        anchors = build_evidence_anchors(self._sample_context())
        self.assertGreaterEqual(len(anchors), 2)
        first = anchors[0]
        self.assertTrue(str(first.get("anchor_id", "")).strip())
        self.assertTrue(str(first.get("doc_key", "")).strip())
        self.assertIsInstance(first.get("chunk_index"), int)
        self.assertTrue(str(first.get("source_type", "")).strip())
        self.assertTrue(str(first.get("source_name", "")).strip())
        self.assertGreater(float(first.get("source_confidence", 0.0)), 0.0)

    def test_current_involved_requires_direct_evidence(self) -> None:
        anchors = build_evidence_anchors(self._sample_context())
        current = build_current_involved_modules(
            task_query="请评估手动下单流程的订单权限控制与联动影响",
            anchors=anchors,
            max_modules=2,
        )
        self.assertGreaterEqual(len(current), 1)
        self.assertLessEqual(len(current), 2)
        for item in current:
            self.assertIn("module", item)
            self.assertIn("confidence", item)
            self.assertIn("top_evidence", item)
            self.assertIn("evidence_anchor", item)
            self.assertIn("signal_breakdown", item)

    def test_impact_v2_has_linked_modules_with_triplet_constraints(self) -> None:
        anchors = build_evidence_anchors(self._sample_context())
        current = build_current_involved_modules(
            task_query="手动下单新增权限控制，需关注订单联动影响",
            anchors=anchors,
            max_modules=2,
        )
        result = build_impact_analysis_v2(
            task_query="手动下单新增权限控制，需关注订单联动影响",
            anchors=anchors,
            current_involved_modules=current,
        )
        self.assertEqual(result.get("version"), "v2")
        self.assertIn("current_involved_modules", result)
        self.assertIn("potential_linked_modules", result)
        for item in result.get("potential_linked_modules", []):
            self.assertTrue(item.get("trigger_modules"))
            self.assertTrue(item.get("evidence_anchor"))
            self.assertIn(item.get("confidence_level"), {"high", "medium"})

    def test_module_normalization_should_merge_aliases(self) -> None:
        self.assertEqual(canonicalize_module("库存"), "库存/权益")
        self.assertEqual(canonicalize_module("通知"), "通知/消息")
        self.assertEqual(canonicalize_module("权限"), "用户/权限")

    def test_current_requires_direct_task_evidence(self) -> None:
        context = "\n".join(
            [
                "### 需求文档",
                "- 片段1 | doc_key=requirement|file_upload|req.md | chunk_index=1 | source_type=requirement | source=req.md | origin=file_upload | modules=库存/权益 | downstream=支付 | feature=f1 | trace=t1",
                "订单创建后触发库存处理逻辑。",
            ]
        )
        anchors = build_evidence_anchors(context)
        current = build_current_involved_modules(
            task_query="仅验证订单创建流程",
            anchors=anchors,
            max_modules=2,
        )
        modules = [str(item.get("module", "")) for item in current]
        self.assertIn("订单", modules)
        self.assertNotIn("库存/权益", modules)

    def test_linked_direction_should_not_use_upstream_only(self) -> None:
        context = "\n".join(
            [
                "### 需求文档",
                "- 片段1 | doc_key=requirement|file_upload|req.md | chunk_index=1 | source_type=requirement | source=req.md | origin=file_upload | modules=订单 | upstream=支付 | feature=f2 | trace=t2",
                "订单流程需要评估影响。",
            ]
        )
        anchors = build_evidence_anchors(context)
        current = build_current_involved_modules(
            task_query="评估订单改动影响",
            anchors=anchors,
            max_modules=2,
        )
        impact = build_impact_analysis_v2(
            task_query="评估订单改动影响",
            anchors=anchors,
            current_involved_modules=current,
        )
        linked_modules = [str(item.get("module", "")) for item in impact.get("potential_linked_modules", [])]
        self.assertEqual(linked_modules, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
