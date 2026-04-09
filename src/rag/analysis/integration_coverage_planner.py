from __future__ import annotations

import re
from typing import Any, Dict, List

_DOMAINS = ["商品", "库存", "订单"]
_ACTIONS = ["同步", "重试", "补偿", "回滚"]
_OUTCOMES = ["成功", "失败", "超时", "限流", "重复回调"]


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip().lower()


def _contains(text: str, token: str) -> bool:
    return _normalize(token) in _normalize(text)


def _domain_priority(domain: str) -> int:
    if domain == "订单":
        return 0
    if domain == "库存":
        return 1
    return 2


def build_integration_coverage_matrix(
    *,
    task_query: str,
    retrieval_context: str,
    current_modules: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Build a fixed integration coverage matrix for 商品/库存/订单 sync flows.
    """
    text = f"{task_query}\n{retrieval_context}"
    current_modules = current_modules if isinstance(current_modules, list) else []

    rows: List[Dict[str, Any]] = []
    selected_count = 0
    for domain in _DOMAINS:
        for action in _ACTIONS:
            for outcome in _OUTCOMES:
                row_id = f"{domain}-{action}-{outcome}"
                hit_domain = _contains(text, domain)
                hit_action = _contains(text, action)
                has_current = any(_contains(str(module), domain) for module in current_modules)
                selected = bool((hit_domain and hit_action) or (has_current and action in {"同步", "补偿"}))
                if selected:
                    selected_count += 1
                rows.append(
                    {
                        "id": row_id,
                        "domain": domain,
                        "action": action,
                        "outcome": outcome,
                        "selected": selected,
                        "priority": "P0" if selected else "P1",
                    }
                )

    rows.sort(
        key=lambda item: (
            0 if item.get("selected") else 1,
            _domain_priority(str(item.get("domain") or "")),
            str(item.get("action") or ""),
            str(item.get("outcome") or ""),
        )
    )
    summary = {
        "total_cells": len(rows),
        "selected_cells": selected_count,
        "selected_ratio": round(selected_count / max(1, len(rows)), 4),
        "must_cover_domains": _DOMAINS,
    }
    return {"coverage_matrix": rows, "coverage_summary": summary}

