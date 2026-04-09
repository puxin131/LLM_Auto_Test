from __future__ import annotations

import unittest

from src.rag.analysis.contract_extractor import build_dual_contracts
from src.rag.analysis.integration_coverage_planner import build_integration_coverage_matrix
from src.rag.analysis.mapping_extractor import build_mapping_rules


class TestIntegrationSupport(unittest.TestCase):
    def _context(self) -> str:
        return "\n".join(
            [
                "### 需求文档",
                "- 片段1 | doc_key=requirement|file_upload|internal_req.md | chunk_index=1 | source_type=requirement | source=internal_req.md | origin=file_upload | modules=订单,库存",
                "内部接口 POST /api/order/sync 会触发库存同步，状态码 200/409。",
                "### API接口文档",
                "- 片段1 | doc_key=api_doc|file_upload|erp_api.md | chunk_index=2 | source_type=api_doc | source=erp_api.md | origin=file_upload | modules=商品",
                "三方ERP接口 POST /erp/products/sync，inner_sku -> erpSku，status -> syncStatus。",
            ]
        )

    def test_dual_contracts_split_internal_external(self) -> None:
        data = build_dual_contracts(task_query="验证商品与订单同步", retrieval_context=self._context())
        self.assertIn("internal_contract", data)
        self.assertIn("external_contract", data)
        internal = data["internal_contract"]
        external = data["external_contract"]
        self.assertTrue(any("/api/order/sync" in x for x in internal.get("interfaces", [])))
        self.assertTrue(any("/erp/products/sync" in x for x in external.get("interfaces", [])))

    def test_mapping_rules_extract_pairs(self) -> None:
        data = build_mapping_rules(task_query="inner_sku -> erpSku", retrieval_context=self._context())
        rules = data.get("mapping_rules", [])
        self.assertGreaterEqual(len(rules), 1)
        keys = [str(item.get("rule_key", "")) for item in rules if isinstance(item, dict)]
        self.assertIn("inner_sku->erpSku", keys)

    def test_coverage_matrix_contains_core_domains(self) -> None:
        data = build_integration_coverage_matrix(
            task_query="重点校验商品同步、库存同步、订单同步及超时重试补偿",
            retrieval_context=self._context(),
            current_modules=["商品", "库存", "订单"],
        )
        matrix = data.get("coverage_matrix", [])
        self.assertTrue(isinstance(matrix, list) and len(matrix) > 0)
        domains = {str(item.get("domain", "")) for item in matrix if isinstance(item, dict)}
        self.assertEqual(domains, {"商品", "库存", "订单"})
        selected = [item for item in matrix if isinstance(item, dict) and bool(item.get("selected"))]
        self.assertGreater(len(selected), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

