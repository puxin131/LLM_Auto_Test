from __future__ import annotations

import unittest

from src.rag.analysis.linkage_extractor import build_bidirectional_link_analysis


class TestLinkageExtractor(unittest.TestCase):
    def test_build_bidirectional_link_analysis_outputs_edges_and_trace_refs(self) -> None:
        context = "\n".join(
            [
                "### 需求文档",
                "- 片段1 | doc_key=requirement|file_upload|req.md | chunk_index=1 | source_type=requirement | source=req.md | origin=file_upload | modules=订单 | downstream=库存/权益 | feature=f1 | trace=T-1 | related=api_doc|file_upload|api.md",
                "订单状态变化需要库存联动",
                "### API接口文档",
                "- 片段1 | doc_key=api_doc|file_upload|api.md | chunk_index=2 | source_type=api_doc | source=api.md | origin=file_upload | modules=库存/权益 | upstream=订单 | feature=f1 | trace=T-1",
                "库存扣减接口",
                "### 测试用例",
                "- 片段1 | doc_key=testcase|file_upload|tc.md | chunk_index=1 | source_type=testcase | source=tc.md | origin=file_upload | modules=订单 | trace=T-1",
                "历史回归用例",
            ]
        )
        result = build_bidirectional_link_analysis(
            task_query="评估订单改动跨端影响",
            retrieval_context=context,
        )

        self.assertIsInstance(result, dict)
        edges = result.get("link_edges", [])
        self.assertIsInstance(edges, list)
        self.assertTrue(edges)
        relations = {str(item.get("relation", "")) for item in edges if isinstance(item, dict)}
        self.assertIn("related_doc", relations)
        self.assertIn("shared_trace", relations)

        trace_refs = result.get("trace_refs", {})
        self.assertIsInstance(trace_refs, dict)
        self.assertTrue(trace_refs.get("req_ids"))
        self.assertTrue(trace_refs.get("api_ids"))
        self.assertTrue(trace_refs.get("testcase_ids"))

    def test_module_edges_should_be_bidirectional_pairs(self) -> None:
        context = "\n".join(
            [
                "### 需求文档",
                "- 片段1 | doc_key=requirement|file_upload|req.md | chunk_index=1 | source_type=requirement | source=req.md | origin=file_upload | modules=订单 | downstream=库存/权益 | trace=T-2",
                "订单变更会触发库存联动",
            ]
        )
        result = build_bidirectional_link_analysis(
            task_query="订单模块联动",
            retrieval_context=context,
        )
        edges = result.get("link_edges", [])
        if not isinstance(edges, list):
            self.fail("link_edges should be list")

        pair = set()
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("src_id", ""))
            dst = str(edge.get("dst_id", ""))
            rel = str(edge.get("relation", ""))
            if src.startswith("module:") and dst.startswith("module:"):
                pair.add((src, dst, rel))

        has_downstream = any(rel == "module_downstream" for _, _, rel in pair)
        has_upstream = any(rel == "module_upstream" for _, _, rel in pair)
        self.assertTrue(has_downstream)
        self.assertTrue(has_upstream)


if __name__ == "__main__":
    unittest.main(verbosity=2)

