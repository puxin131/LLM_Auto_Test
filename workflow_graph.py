from __future__ import annotations

import inspect
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from src.rag.analysis import (
    build_bidirectional_link_analysis,
    evaluate_constraint_compliance,
    build_dual_contracts,
    build_current_involved_modules,
    build_evidence_anchors,
    build_integration_coverage_matrix,
    build_impact_analysis_v2,
    build_mapping_rules,
)
from src.rag.analysis.badcase_loop import (
    auto_tune_rule_templates_from_replay,
    build_badcase_replay_report,
    load_badcase_rule_templates,
    record_badcase_event,
)

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:
    END = "__END__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    LANGGRAPH_AVAILABLE = False


IntentLabel = Literal["ui", "api", "fallback"]
ReviewDecision = Literal["pass", "fail"]
GeneratorName = Literal["ui_generator", "api_generator", "fallback_generator"]
RetrievalPolicy = Dict[str, Any]
GenerationMode = Literal["business_api", "field_validation"]

PROJECT_ROOT = Path(__file__).resolve().parent


class ReviewScores(TypedDict):
    business_coverage: int
    exception_coverage: int
    assertion_specificity: int
    executability: int
    traceability: int
    redundancy_control: int
    # Backward-compatible aliases for existing UI rendering.
    exception_flow: int
    assertion_clarity: int
    total: int


class ReviewResult(TypedDict):
    decision: ReviewDecision
    scores: ReviewScores
    hard_fail_reasons: List[str]
    comments: List[str]
    missing_points: List[str]
    rewrite_instructions: List[str]


class QAState(TypedDict, total=False):
    request_id: str
    user_requirement_raw: str
    user_requirement_normalized: str

    intent_label: IntentLabel
    intent_confidence: float
    classifier_reason: str

    retrieval_query: str
    retrieval_context: str
    retrieval_context_len: int
    retrieval_meta: Dict[str, Any]
    retrieval_policy: RetrievalPolicy
    human_inputs: Dict[str, Any]
    run_context: Dict[str, Any]
    link_edges: List[Dict[str, Any]]
    trace_refs: Dict[str, List[str]]
    link_summary: Dict[str, Any]
    contracts: Dict[str, Any]
    mapping_rules: Dict[str, Any]
    coverage_matrix: Dict[str, Any]

    active_generator: GeneratorName
    generator_prompt_name: str
    recommended_mode: GenerationMode
    recommended_mode_lock: GenerationMode | Literal[""]
    draft_testcases_md: str

    review_result: ReviewResult
    review_raw_text: str
    review_passed: bool
    review_comments: List[str]
    compliance_report: Dict[str, Any]
    risk_report: Dict[str, Any]

    iteration: int
    max_iterations: int
    route_history: List[str]

    final_testcases_md: str
    final_status: Literal["success", "success_with_warning", "failed"]
    error: Optional[str]


def _normalize_llm_content(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _safe_json_loads(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            pass

    obj = re.search(r"(\{[\s\S]*\})", raw)
    if obj:
        try:
            payload = json.loads(obj.group(1))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_len(text: Any) -> int:
    return len(str(text or ""))


def _safe_int(value: Any, default: int = 0, low: int = 0, high: int = 5) -> int:
    try:
        x = int(value)
    except Exception:
        return default
    if x < low:
        return low
    if x > high:
        return high
    return x


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _parse_human_input_list(value: Any, max_items: int = 8) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,\n，、;/；|]+", str(value))
    items: List[str] = []
    for item in raw_items:
        text = str(item).strip()
        if not text or text in items:
            continue
        items.append(text)
        if len(items) >= max_items:
            break
    return items


def _normalize_human_inputs(
    human_inputs: Dict[str, Any] | None,
    retrieval_policy: RetrievalPolicy | None,
) -> Dict[str, Any]:
    raw = human_inputs if isinstance(human_inputs, dict) else {}
    policy = retrieval_policy if isinstance(retrieval_policy, dict) else {}

    scope = str(raw.get("scope", "")).strip()[:500]
    exclusions = str(raw.get("exclusions", "")).strip()[:500]
    risk_focus = str(raw.get("risk_focus", "")).strip()[:500]
    priority_modules = _parse_human_input_list(raw.get("priority_modules"))
    if not priority_modules:
        priority_modules = _extract_modules_from_policy(policy)

    release = str(raw.get("release", "")).strip()
    if not release:
        release = str(policy.get("release", "")).strip()

    approved_only_raw = raw.get("approved_only")
    if approved_only_raw is None:
        approved_only = bool(policy.get("approved_only", True))
    else:
        approved_only = bool(approved_only_raw)

    scope_items = _parse_human_input_list(scope, max_items=10)
    exclusion_items = _parse_human_input_list(exclusions, max_items=10)
    risk_items = _parse_human_input_list(risk_focus, max_items=10)

    must_cover = _dedupe_terms(priority_modules + [x for x in scope_items if len(str(x).strip()) <= 20])
    must_not_cover = _dedupe_terms(exclusion_items)
    risk_tags = _dedupe_terms(risk_items)

    return {
        "scope": scope,
        "exclusions": exclusions,
        "risk_focus": risk_focus,
        "priority_modules": priority_modules[:8],
        "release": release,
        "approved_only": approved_only,
        "must_cover": must_cover,
        "must_not_cover": must_not_cover,
        "risk_tags": risk_tags,
    }


def _normalize_run_context(human_inputs: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = human_inputs if isinstance(human_inputs, dict) else {}
    run_context_raw = raw.get("run_context", raw)
    if not isinstance(run_context_raw, dict):
        run_context_raw = {}
    confirmed = bool(run_context_raw.get("confirmed_by_user", False))
    confirmation_ts = str(run_context_raw.get("confirmation_ts", "")).strip()
    if confirmed and not confirmation_ts:
        confirmation_ts = _utc_now_iso()
    return {
        "confirmed_by_user": confirmed,
        "confirmation_ts": confirmation_ts,
    }


def _normalize_constraint_term(raw: Any) -> str:
    text = _normalize_text(raw)
    return text


def _dedupe_terms(items: List[str], max_items: int = 10) -> List[str]:
    terms: List[str] = []
    for item in items:
        normalized = _normalize_constraint_term(item)
        if not normalized or normalized in terms:
            continue
        terms.append(normalized)
        if len(terms) >= max_items:
            break
    return terms


def _normalize_generation_mode(value: Any, default: GenerationMode = "business_api") -> GenerationMode:
    text = str(value or "").strip().lower()
    if text == "field_validation":
        return "field_validation"
    if text == "business_api":
        return "business_api"
    return default


def _normalize_optional_generation_mode(value: Any) -> GenerationMode | Literal[""]:
    text = str(value or "").strip().lower()
    if text in {"field_validation", "business_api"}:
        return text  # type: ignore[return-value]
    return ""


def _resolve_effective_generation_mode(base_mode: Any, recommended_mode: Any) -> GenerationMode:
    base = _normalize_generation_mode(base_mode)
    recommended = _normalize_generation_mode(recommended_mode, default=base)
    return recommended


def _recommend_generation_mode(current_mode: Any, review_result: Any) -> GenerationMode:
    fallback_mode = _normalize_generation_mode(current_mode)
    if not isinstance(review_result, dict):
        return fallback_mode

    scores = review_result.get("scores", {})
    if not isinstance(scores, dict):
        scores = {}
    business_coverage = _safe_int(scores.get("business_coverage"), default=3)
    assertion_specificity = _safe_int(
        scores.get("assertion_specificity", scores.get("assertion_clarity")),
        default=3,
    )
    traceability = _safe_int(scores.get("traceability"), default=3)

    text_fragments: List[str] = []
    for key in ("missing_points", "rewrite_instructions", "comments", "hard_fail_reasons"):
        value = review_result.get(key, [])
        if isinstance(value, list):
            text_fragments.extend([str(x) for x in value])
        elif value:
            text_fragments.append(str(value))
    review_text = _normalize_text(" ".join(text_fragments))
    if not review_text:
        return fallback_mode

    field_keywords = [
        "字段",
        "参数",
        "必填",
        "枚举",
        "长度",
        "类型",
        "格式",
        "校验",
        "默认值",
    ]
    business_keywords = [
        "业务流",
        "状态流转",
        "跨模块",
        "联动",
        "上下游",
        "权限",
        "补偿",
        "回滚",
        "一致性",
    ]

    # 结构化信号优先，文本关键字用于加权，不直接决定切换。
    field_signal = 0
    business_signal = 0
    if assertion_specificity <= 2:
        field_signal += 3
    if business_coverage <= 2:
        business_signal += 3
    if traceability <= 2:
        business_signal += 2

    for keyword in field_keywords:
        if keyword in review_text:
            field_signal += 1
    for keyword in business_keywords:
        if keyword in review_text:
            business_signal += 1

    # 切换阈值: 至少达到强信号，且领先另一方向至少 1 分。
    if field_signal >= 3 and field_signal >= business_signal + 1:
        return "field_validation"
    if business_signal >= 3 and business_signal >= field_signal + 1:
        return "business_api"
    return fallback_mode


def _contains_any(text: str, keywords: List[str]) -> bool:
    for keyword in keywords:
        if keyword and keyword in text:
            return True
    return False


def _extract_modules_from_policy(policy: RetrievalPolicy | None) -> List[str]:
    if not isinstance(policy, dict):
        return []
    raw = policy.get("modules", [])
    if isinstance(raw, str):
        items = re.split(r"[,\n，、;/；|]+", raw)
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    modules: List[str] = []
    for item in items:
        value = str(item).strip()
        if value and value not in modules:
            modules.append(value)
    return modules


def _build_impact_analysis(
    *,
    task_query: str,
    retrieval_context: str,
    retrieval_policy: RetrievalPolicy | None,
) -> Dict[str, Any]:
    # retrieval_policy 当前阶段仅保留接口位，推断内核统一基于 anchors + 多信号引擎。
    _ = retrieval_policy

    anchors = build_evidence_anchors(retrieval_context)
    current_involved_modules = build_current_involved_modules(
        task_query=task_query,
        anchors=anchors,
        max_modules=2,
    )

    if not current_involved_modules:
        return {}

    impact_v2 = build_impact_analysis_v2(
        task_query=task_query,
        anchors=anchors,
        current_involved_modules=current_involved_modules,
    )

    potential_linked = impact_v2.get("potential_linked_modules", [])
    if not isinstance(potential_linked, list):
        potential_linked = []

    if not current_involved_modules and not potential_linked:
        return {}

    return impact_v2


def _build_generation_support(
    *,
    task_query: str,
    retrieval_context: str,
) -> Dict[str, Any]:
    anchors = build_evidence_anchors(retrieval_context)
    current_involved_modules = build_current_involved_modules(
        task_query=task_query,
        anchors=anchors,
        max_modules=2,
    )
    current_names = [
        str(item.get("module") or "").strip()
        for item in current_involved_modules
        if isinstance(item, dict) and str(item.get("module") or "").strip()
    ]
    contracts = build_dual_contracts(
        task_query=task_query,
        retrieval_context=retrieval_context,
        anchors=anchors,
    )
    mapping_rules = build_mapping_rules(
        task_query=task_query,
        retrieval_context=retrieval_context,
        anchors=anchors,
    )
    coverage_matrix = build_integration_coverage_matrix(
        task_query=task_query,
        retrieval_context=retrieval_context,
        current_modules=current_names,
    )
    link_analysis = build_bidirectional_link_analysis(
        task_query=task_query,
        retrieval_context=retrieval_context,
    )
    return {
        "contracts": contracts if isinstance(contracts, dict) else {},
        "mapping_rules": mapping_rules if isinstance(mapping_rules, dict) else {},
        "coverage_matrix": coverage_matrix if isinstance(coverage_matrix, dict) else {},
        "link_analysis": link_analysis if isinstance(link_analysis, dict) else {},
    }


def _format_generation_support_block(
    *,
    contracts: Dict[str, Any],
    mapping_rules: Dict[str, Any],
    coverage_matrix: Dict[str, Any],
    link_analysis: Dict[str, Any],
) -> str:
    internal_contract = contracts.get("internal_contract", {}) if isinstance(contracts, dict) else {}
    external_contract = contracts.get("external_contract", {}) if isinstance(contracts, dict) else {}
    contract_summary = str(contracts.get("contract_summary", "")).strip() if isinstance(contracts, dict) else ""
    mapping_items = mapping_rules.get("mapping_rules", []) if isinstance(mapping_rules, dict) else []
    coverage_items = coverage_matrix.get("coverage_matrix", []) if isinstance(coverage_matrix, dict) else []
    coverage_summary = coverage_matrix.get("coverage_summary", {}) if isinstance(coverage_matrix, dict) else {}

    if not isinstance(mapping_items, list):
        mapping_items = []
    if not isinstance(coverage_items, list):
        coverage_items = []
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
    link_edges = link_analysis.get("link_edges", []) if isinstance(link_analysis, dict) else []
    link_summary = link_analysis.get("link_summary", {}) if isinstance(link_analysis, dict) else {}
    trace_refs = link_analysis.get("trace_refs", {}) if isinstance(link_analysis, dict) else {}
    if not isinstance(link_edges, list):
        link_edges = []
    if not isinstance(link_summary, dict):
        link_summary = {}
    if not isinstance(trace_refs, dict):
        trace_refs = {}

    internal_interfaces = internal_contract.get("interfaces", []) if isinstance(internal_contract, dict) else []
    external_interfaces = external_contract.get("interfaces", []) if isinstance(external_contract, dict) else []
    if not isinstance(internal_interfaces, list):
        internal_interfaces = []
    if not isinstance(external_interfaces, list):
        external_interfaces = []

    selected_cells = [item for item in coverage_items if isinstance(item, dict) and bool(item.get("selected"))]
    selected_preview = [
        f"{str(item.get('domain', '-'))}-{str(item.get('action', '-'))}-{str(item.get('outcome', '-'))}"
        for item in selected_cells[:10]
    ]
    mapping_preview = []
    for item in mapping_items[:10]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("rule_key") or "").strip()
        if key:
            mapping_preview.append(key)
    edge_preview = []
    for edge in link_edges[:10]:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src_id", "")).replace("doc:", "").replace("module:", "")
        dst = str(edge.get("dst_id", "")).replace("doc:", "").replace("module:", "")
        rel = str(edge.get("relation", ""))
        if src and dst and rel:
            edge_preview.append(f"{src}->{dst}({rel})")
    req_ids = trace_refs.get("req_ids", [])
    api_ids = trace_refs.get("api_ids", [])
    tc_ids = trace_refs.get("testcase_ids", [])
    if not isinstance(req_ids, list):
        req_ids = []
    if not isinstance(api_ids, list):
        api_ids = []
    if not isinstance(tc_ids, list):
        tc_ids = []

    return (
        "【集成分析输入（高优先级）】\n"
        f"- 双域契约摘要: {contract_summary or '-'}\n"
        f"- 内部接口线索: {', '.join(internal_interfaces[:8]) if internal_interfaces else '-'}\n"
        f"- 三方接口线索: {', '.join(external_interfaces[:8]) if external_interfaces else '-'}\n"
        f"- 字段映射规则: {', '.join(mapping_preview) if mapping_preview else '-'}\n"
        f"- 集成覆盖矩阵(已选): {', '.join(selected_preview) if selected_preview else '-'}\n"
        f"- 覆盖矩阵统计: selected={int(coverage_summary.get('selected_cells', 0) or 0)} / "
        f"total={int(coverage_summary.get('total_cells', 0) or 0)}\n"
        f"- 跨端链路边数量: {int(link_summary.get('total_edges', 0) or 0)}\n"
        f"- 跨端链路预览: {', '.join(edge_preview) if edge_preview else '-'}\n"
        f"- 追踪引用: req={len(req_ids)} api={len(api_ids)} testcase={len(tc_ids)}\n"
    )


def _score_dimension(
    missing: bool,
    task_len: int,
    context_len: int,
    review_score: Optional[int],
) -> bool:
    if not missing:
        return False
    score = 1
    if task_len < 120:
        score += 1
    if context_len < 500:
        score += 1
    if review_score is not None and review_score <= 2:
        score += 1
    return score >= 2


def _safe_review_score(review_result: Any, key: str) -> Optional[int]:
    if not isinstance(review_result, dict):
        return None
    scores = review_result.get("scores", {})
    if not isinstance(scores, dict):
        return None
    if key in scores:
        return _safe_int(scores.get(key))
    return None


def _build_gap_summary(missing_inputs: List[str], coverage_risks: List[str]) -> str:
    head = []
    for item in missing_inputs:
        if item not in head:
            head.append(item)
        if len(head) >= 2:
            break
    if not head and coverage_risks:
        for item in coverage_risks:
            if item not in head:
                head.append(item)
            if len(head) >= 2:
                break
    if not head:
        return ""
    summary = "缺少" + "、".join(head) + "等关键信息，可能导致覆盖不足。"
    return summary[:80]


def _build_rule_gap_hints(
    *,
    task_query: str,
    retrieval_context: str,
    intent_label: str,
    generation_mode: str,
    review_result: Any,
    retrieval_policy: RetrievalPolicy | None,
) -> Dict[str, Any]:
    task_len = len(str(task_query or "").strip())
    context_len = len(str(retrieval_context or "").strip())
    text = _normalize_text(task_query) + "\n" + _normalize_text(retrieval_context)
    modules = _extract_modules_from_policy(retrieval_policy)
    cross_module_relevant = len(modules) >= 2 or "跨模块" in text

    dimensions = [
        {
            "label": "角色/权限边界",
            "risk": "权限差异覆盖不足",
            "suggest": "请补充角色、权限与操作边界说明",
            "keywords": ["角色", "权限", "鉴权", "登录", "管理员", "用户", "租户", "rbac", "权限控制"],
            "score_key": None,
        },
        {
            "label": "异常流/错误码",
            "risk": "异常分支覆盖不足",
            "suggest": "请补充错误码、异常处理与回滚规则",
            "keywords": ["异常", "错误", "失败", "error", "错误码", "回滚", "重试", "超时", "降级", "补偿"],
            "score_key": "exception_coverage",
        },
        {
            "label": "字段约束",
            "risk": "字段校验覆盖不足",
            "suggest": "请补充关键字段的校验规则与取值范围",
            "keywords": ["字段", "参数", "必填", "范围", "长度", "格式", "类型", "枚举", "校验", "约束"],
            "score_key": "assertion_specificity",
        },
        {
            "label": "状态流转",
            "risk": "状态流转覆盖不足",
            "suggest": "请补充关键状态流转与触发条件",
            "keywords": ["状态流转", "状态机", "状态变更", "生命周期", "流转", "状态"],
            "score_key": "traceability",
        },
        {
            "label": "边界条件",
            "risk": "边界条件覆盖不足",
            "suggest": "请补充边界/空值/极限条件说明",
            "keywords": ["边界", "最大", "最小", "为空", "空值", "null", "超长", "上限", "下限"],
            "score_key": "assertion_specificity",
        },
        {
            "label": "跨模块影响",
            "risk": "跨模块联动覆盖不足",
            "suggest": "请补充上下游/跨模块影响与联动场景",
            "keywords": ["跨模块", "联动", "影响", "依赖", "一致性", "同步", "上下游", "关联"],
            "score_key": "traceability",
        },
    ]

    missing_inputs: List[str] = []
    coverage_risks: List[str] = []
    suggested_prompts: List[str] = []

    for dim in dimensions:
        if dim["label"] == "跨模块影响" and not cross_module_relevant:
            continue
        missing = not _contains_any(text, dim["keywords"])
        review_score = _safe_review_score(review_result, dim["score_key"]) if dim["score_key"] else None
        if _score_dimension(missing, task_len, context_len, review_score):
            missing_inputs.append(dim["label"])
            coverage_risks.append(dim["risk"])
            suggested_prompts.append(dim["suggest"])

    if not missing_inputs and not coverage_risks:
        return {}

    summary = _build_gap_summary(missing_inputs, coverage_risks)
    return {
        "gap_summary": summary,
        "missing_inputs": missing_inputs[:6],
        "coverage_risks": coverage_risks[:6],
        "suggested_prompts": suggested_prompts[:6],
        "intent_label": intent_label,
        "generation_mode": generation_mode,
    }


def _normalize_gap_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summary = str(payload.get("gap_summary", "") or "").strip()
    missing_inputs = payload.get("missing_inputs", [])
    coverage_risks = payload.get("coverage_risks", [])
    suggested_prompts = payload.get("suggested_prompts", [])
    if not isinstance(missing_inputs, list):
        missing_inputs = [missing_inputs]
    if not isinstance(coverage_risks, list):
        coverage_risks = [coverage_risks]
    if not isinstance(suggested_prompts, list):
        suggested_prompts = [suggested_prompts]
    missing_inputs = [str(x).strip() for x in missing_inputs if str(x).strip()]
    coverage_risks = [str(x).strip() for x in coverage_risks if str(x).strip()]
    suggested_prompts = [str(x).strip() for x in suggested_prompts if str(x).strip()]
    return {
        "gap_summary": summary[:80],
        "missing_inputs": missing_inputs[:6],
        "coverage_risks": coverage_risks[:6],
        "suggested_prompts": suggested_prompts[:6],
    }


def _merge_gap_hints(rule_hints: Dict[str, Any], llm_hints: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(rule_hints or {})
    llm_payload = _normalize_gap_payload(llm_hints)
    if llm_payload.get("gap_summary"):
        merged["gap_summary"] = llm_payload["gap_summary"]

    for key in ("missing_inputs", "coverage_risks", "suggested_prompts"):
        items = []
        for source in (rule_hints.get(key, []), llm_payload.get(key, [])):
            for item in source or []:
                if item and item not in items:
                    items.append(item)
        merged[key] = items[:6]
    if not merged.get("gap_summary"):
        merged["gap_summary"] = _build_gap_summary(
            merged.get("missing_inputs", []),
            merged.get("coverage_risks", []),
        )
    return merged


def _llm_supplement_gap_hints(
    *,
    task_query: str,
    retrieval_context: str,
    intent_label: str,
    generation_mode: str,
    review_result: Any,
    rule_hints: Dict[str, Any],
    llm: Any,
) -> Dict[str, Any]:
    if not rule_hints or not getattr(llm, "invoke", None):
        return {}
    context = str(retrieval_context or "")
    trimmed_context = context[:1200]
    prompt = (
        "你是测试设计缺口提示的补充助手，仅输出 JSON：\n"
        "{\n"
        '  "gap_summary":"不超过40字",\n'
        '  "missing_inputs":["..."],\n'
        '  "coverage_risks":["..."],\n'
        '  "suggested_prompts":["..."]\n'
        "}\n\n"
        "规则：\n"
        "- 基于已有缺口补充，不重复现有内容\n"
        "- 每类最多补充2条\n"
        "- 若无明显补充，返回空数组，gap_summary 可为空\n"
        "- 仅输出 JSON，不要解释\n\n"
        f"已有缺口：{json.dumps(rule_hints, ensure_ascii=False)}\n"
        f"意图: {intent_label}\n"
        f"生成模式: {generation_mode}\n"
        f"评审结果摘要: {json.dumps(review_result or {}, ensure_ascii=False)}\n"
        f"需求: {task_query}\n"
        f"检索上下文: {trimmed_context}"
    )
    try:
        response = llm.invoke(prompt)
        raw = _normalize_llm_content(response)
        parsed = _safe_json_loads(raw)
        return _normalize_gap_payload(parsed)
    except Exception:
        return {}


def _build_gap_hints(
    *,
    task_query: str,
    retrieval_context: str,
    intent_label: str,
    generation_mode: str,
    review_result: Any,
    retrieval_policy: RetrievalPolicy | None,
    llm: Any,
) -> Dict[str, Any]:
    rule_hints = _build_rule_gap_hints(
        task_query=task_query,
        retrieval_context=retrieval_context,
        intent_label=intent_label,
        generation_mode=generation_mode,
        review_result=review_result,
        retrieval_policy=retrieval_policy,
    )
    if not rule_hints:
        return {}
    llm_hints = _llm_supplement_gap_hints(
        task_query=task_query,
        retrieval_context=retrieval_context,
        intent_label=intent_label,
        generation_mode=generation_mode,
        review_result=review_result,
        rule_hints=rule_hints,
        llm=llm,
    )
    merged = _merge_gap_hints(rule_hints, llm_hints)
    if not merged.get("missing_inputs") and not merged.get("coverage_risks"):
        return {}
    return merged


def _risk_severity_weight(severity: str) -> int:
    text = str(severity or "").upper()
    if text == "P0":
        return 3
    if text == "P1":
        return 2
    return 1


def _to_text_list(values: Any, *, max_items: int = 8) -> List[str]:
    if isinstance(values, list):
        raw = values
    elif values is None:
        raw = []
    else:
        raw = [values]
    items: List[str] = []
    for value in raw:
        text = str(value or "").strip()
        if not text or text in items:
            continue
        items.append(text)
        if len(items) >= max_items:
            break
    return items


def _append_risk_item(
    items: List[Dict[str, Any]],
    *,
    severity: str,
    category: str,
    title: str,
    reason: str,
    evidence: List[str] | None = None,
    suggestions: List[str] | None = None,
) -> None:
    normalized_title = str(title or "").strip()
    normalized_category = str(category or "").strip()
    if not normalized_title or not normalized_category:
        return
    for item in items:
        if (
            str(item.get("category", "")) == normalized_category
            and str(item.get("title", "")) == normalized_title
        ):
            return
    items.append(
        {
            "severity": str(severity or "P2").upper(),
            "category": normalized_category,
            "title": normalized_title,
            "reason": str(reason or "").strip(),
            "evidence": _to_text_list(evidence or [], max_items=6),
            "suggestions": _to_text_list(suggestions or [], max_items=4),
        }
    )


def _read_recent_jsonl_rows(path: Path, *, max_lines: int = 240, max_bytes: int = 256 * 1024) -> List[Dict[str, Any]]:
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
        # 若截断读取，第一行可能是半行 JSON，直接丢弃。
        if read_size < file_size and lines:
            lines = lines[1:]
        rows: List[Dict[str, Any]] = []
        for line in lines[-max_lines:]:
            payload = _safe_json_loads(str(line or "").strip())
            if payload:
                rows.append(payload)
        return rows
    except Exception:
        return []


def _build_badcase_profile(window_size: int = 240) -> Dict[str, Any]:
    rows = _read_recent_jsonl_rows(
        PROJECT_ROOT / "data" / "observation_log.jsonl",
        max_lines=max(30, int(window_size)),
    )
    if not rows:
        return {
            "window_size": 0,
            "overall_bad_rate": 0.0,
            "tag_stats": {},
        }

    bad_statuses = {"failed", "success_with_warning"}
    total = len(rows)
    total_bad = 0
    tag_stats: Dict[str, Dict[str, Any]] = {}
    tag_rules = {
        "context_empty": lambda row: int(row.get("context_len", 0) or 0) <= 0,
        "weak_input": lambda row: bool(row.get("weak_input", False)),
        "gap_hit": lambda row: bool(row.get("gap_hints_hit", False)),
        "impact_high": lambda row: int(row.get("impact_high_count", 0) or 0) > 0,
        "impact_medium_plus": lambda row: int(row.get("impact_medium_count", 0) or 0) > 0,
    }
    for tag in tag_rules:
        tag_stats[tag] = {"sample": 0, "bad": 0, "bad_rate": 0.0}

    for row in rows:
        status = str(row.get("final_status", "")).strip().lower()
        is_bad = status in bad_statuses
        if is_bad:
            total_bad += 1
        for tag, checker in tag_rules.items():
            matched = False
            try:
                matched = bool(checker(row))
            except Exception:
                matched = False
            if not matched:
                continue
            tag_stats[tag]["sample"] += 1
            if is_bad:
                tag_stats[tag]["bad"] += 1

    for tag, stats in tag_stats.items():
        sample = int(stats.get("sample", 0) or 0)
        bad = int(stats.get("bad", 0) or 0)
        stats["bad_rate"] = round((bad / float(sample)) if sample > 0 else 0.0, 4)

    return {
        "window_size": total,
        "overall_bad_rate": round(total_bad / float(total), 4),
        "tag_stats": tag_stats,
    }


def _build_risk_report(
    *,
    retrieval_context_len: int,
    compliance_report: Dict[str, Any],
    gap_hints: Dict[str, Any],
    impact_analysis: Dict[str, Any],
    link_summary: Dict[str, Any],
    trace_refs: Dict[str, Any],
) -> Dict[str, Any]:
    report = compliance_report if isinstance(compliance_report, dict) else {}
    gaps = gap_hints if isinstance(gap_hints, dict) else {}
    impact = impact_analysis if isinstance(impact_analysis, dict) else {}
    summary = link_summary if isinstance(link_summary, dict) else {}
    traces = trace_refs if isinstance(trace_refs, dict) else {}

    items: List[Dict[str, Any]] = []

    if int(retrieval_context_len or 0) <= 0:
        _append_risk_item(
            items,
            severity="P0",
            category="retrieval_quality",
            title="检索上下文为空",
            reason="生成结果缺少可追溯证据，易出现覆盖偏差。",
            evidence=["retrieval_context_len=0"],
            suggestions=["放宽筛选条件并补充关键需求/接口资产后重试。"],
        )

    missing_items = report.get("missing_items", {}) if isinstance(report, dict) else {}
    if not isinstance(missing_items, dict):
        missing_items = {}
    if not bool(report.get("pass", True)):
        must_not_cover_hits = _to_text_list(missing_items.get("must_not_cover_hits", []), max_items=6)
        if must_not_cover_hits:
            _append_risk_item(
                items,
                severity="P0",
                category="constraint_violation",
                title="命中排除项",
                reason="生成内容触发前置排除约束，存在范围偏移风险。",
                evidence=must_not_cover_hits,
                suggestions=["先移除排除项相关场景，再按约束重新生成。"],
            )
        missing_must_cover = _to_text_list(missing_items.get("must_cover", []), max_items=6)
        if missing_must_cover:
            _append_risk_item(
                items,
                severity="P1",
                category="constraint_violation",
                title="必须覆盖项缺失",
                reason="关键覆盖目标未命中，可能导致核心链路遗漏。",
                evidence=missing_must_cover,
                suggestions=["将缺失项加入主流程和异常分支断言。"],
            )
        missing_risks = _to_text_list(missing_items.get("risk_tags", []), max_items=6)
        if missing_risks:
            _append_risk_item(
                items,
                severity="P1",
                category="constraint_violation",
                title="风险关注项覆盖不足",
                reason="风险焦点未覆盖到用例断言，回归价值下降。",
                evidence=missing_risks,
                suggestions=["对每个风险标签增加至少 1 条可执行断言。"],
            )

    coverage_risks = _to_text_list(gaps.get("coverage_risks", []), max_items=6)
    for risk in coverage_risks:
        _append_risk_item(
            items,
            severity="P1",
            category="coverage_gap",
            title=risk,
            reason="缺口检测命中该覆盖风险。",
            evidence=[risk],
            suggestions=["结合缺口建议补充对应场景与断言。"],
        )

    linked_modules = impact.get("potential_linked_modules", []) if isinstance(impact, dict) else []
    if not isinstance(linked_modules, list):
        linked_modules = []
    for module_item in linked_modules[:3]:
        if not isinstance(module_item, dict):
            continue
        module = str(module_item.get("module", "")).strip()
        if not module:
            continue
        confidence = float(module_item.get("confidence", 0.0) or 0.0)
        level = "P1" if confidence >= 0.72 else "P2"
        evidence = []
        trigger_modules = _to_text_list(module_item.get("trigger_modules", []), max_items=3)
        if trigger_modules:
            evidence.append("trigger=" + ",".join(trigger_modules))
        top_evidence = str(module_item.get("top_evidence", "")).strip()
        if top_evidence:
            evidence.append(top_evidence)
        _append_risk_item(
            items,
            severity=level,
            category="cross_module_impact",
            title=f"潜在联动模块: {module}",
            reason=f"存在跨模块影响信号(conf={confidence:.2f})，需补齐上下游联动用例。",
            evidence=evidence,
            suggestions=["补充该模块的接口联动、失败补偿和数据一致性断言。"],
        )

    total_edges = int(summary.get("total_edges", 0) or 0)
    req_count = len(_to_text_list(traces.get("req_ids", []), max_items=20))
    api_count = len(_to_text_list(traces.get("api_ids", []), max_items=20))
    testcase_count = len(_to_text_list(traces.get("testcase_ids", []), max_items=20))
    if total_edges <= 1:
        _append_risk_item(
            items,
            severity="P1",
            category="traceability",
            title="跨端链路证据偏弱",
            reason="链路边数量过少，跨端可追溯性不足。",
            evidence=[f"total_edges={total_edges}"],
            suggestions=["补充需求/API/历史用例资产并重新生成链路。"],
        )
    if req_count == 0 or api_count == 0:
        _append_risk_item(
            items,
            severity="P1",
            category="traceability",
            title="需求或接口追踪引用缺失",
            reason="trace_refs 未形成需求与接口双向锚点，回归定位成本高。",
            evidence=[f"req={req_count}", f"api={api_count}", f"testcase={testcase_count}"],
            suggestions=["至少补齐 1 个需求文档和 1 个接口文档的可引用片段。"],
        )

    rule_templates = load_badcase_rule_templates(project_root=PROJECT_ROOT)
    history_rule = (rule_templates.get("templates", {}) or {}).get("history_badcase", {})
    if not isinstance(history_rule, dict):
        history_rule = {}
    history_enabled = bool(history_rule.get("enabled", True))
    history_min_sample = max(1, int(history_rule.get("min_sample", 12) or 12))
    history_min_bad_rate = max(0.0, min(1.0, float(history_rule.get("min_bad_rate", 0.35) or 0.35)))
    history_min_delta = max(0.0, min(1.0, float(history_rule.get("min_delta", 0.12) or 0.12)))
    history_delta_scale = max(1.0, float(history_rule.get("delta_scale", 24.0) or 24.0))
    history_max_boost = max(0, int(history_rule.get("max_boost", 24) or 24))

    history_profile = _build_badcase_profile(window_size=240)
    history_window_size = int(history_profile.get("window_size", 0) or 0)
    history_overall_bad_rate = float(history_profile.get("overall_bad_rate", 0.0) or 0.0)
    history_tag_stats = history_profile.get("tag_stats", {})
    if not isinstance(history_tag_stats, dict):
        history_tag_stats = {}
    current_history_tags: List[str] = []
    if int(retrieval_context_len or 0) <= 0:
        current_history_tags.append("context_empty")
    if bool(coverage_risks):
        current_history_tags.append("gap_hit")
    if bool(linked_modules):
        current_history_tags.append("impact_medium_plus")

    history_hit_details: List[Dict[str, Any]] = []
    for tag in current_history_tags:
        stats = history_tag_stats.get(tag, {})
        if not isinstance(stats, dict):
            continue
        sample = int(stats.get("sample", 0) or 0)
        bad_rate = float(stats.get("bad_rate", 0.0) or 0.0)
        delta = bad_rate - history_overall_bad_rate
        # 至少有足够样本，且显著高于整体 badcase 水位，才作为动态加权依据。
        if (not history_enabled) or sample < history_min_sample or bad_rate < history_min_bad_rate or delta < history_min_delta:
            continue
        history_hit_details.append(
            {
                "tag": tag,
                "sample": sample,
                "bad_rate": bad_rate,
                "delta": delta,
            }
        )
    history_hit_details.sort(
        key=lambda item: (
            -float(item.get("delta", 0.0) or 0.0),
            -float(item.get("bad_rate", 0.0) or 0.0),
        )
    )
    history_hit_details = history_hit_details[:3]
    if history_hit_details:
        evidence = [
            (
                f"{str(item.get('tag', '-'))}:bad_rate="
                f"{float(item.get('bad_rate', 0.0) or 0.0):.2f}"
                f"(sample={int(item.get('sample', 0) or 0)})"
            )
            for item in history_hit_details
        ]
        _append_risk_item(
            items,
            severity="P1",
            category="history_badcase",
            title="历史 badcase 高发场景命中",
            reason=(
                "当前输入命中历史高失败率标签，建议在生成前补全信息并优先覆盖高风险链路。"
            ),
            evidence=evidence,
            suggestions=[
                "针对命中标签补充需求/接口/异常分支信息后再生成。",
                "对高风险链路增加至少 1 条正向 + 1 条异常断言。",
            ],
        )

    severity_counts = {"P0": 0, "P1": 0, "P2": 0}
    weight_sum = 0
    for item in items:
        severity = str(item.get("severity", "P2")).upper()
        if severity not in severity_counts:
            severity = "P2"
            item["severity"] = severity
        severity_counts[severity] += 1
        weight_sum += _risk_severity_weight(severity)
    base_score = min(100, weight_sum * 10)
    history_score_boost = 0
    for hit in history_hit_details:
        delta = float(hit.get("delta", 0.0) or 0.0)
        history_score_boost += max(2, min(10, int(round(delta * history_delta_scale))))
    history_score_boost = min(history_max_boost, history_score_boost)
    score = min(100, base_score + history_score_boost)
    if severity_counts["P0"] > 0 or score >= 40:
        overall_level = "high"
    elif score >= 18:
        overall_level = "medium"
    else:
        overall_level = "low"

    for index, item in enumerate(items, start=1):
        item["id"] = f"RISK-{index:03d}"

    return {
        "version": "v1",
        "rule_template_version": str(rule_templates.get("version", "v1")),
        "overall_level": overall_level,
        "overall_score": float(score),
        "history_adjustment": {
            "enabled": bool(history_hit_details),
            "base_score": float(base_score),
            "score_boost": float(history_score_boost),
            "adjusted_score": float(score),
            "window_size": history_window_size,
            "overall_bad_rate": float(history_overall_bad_rate),
            "matched_tags": [
                {
                    "tag": str(item.get("tag", "")),
                    "sample": int(item.get("sample", 0) or 0),
                    "bad_rate": float(item.get("bad_rate", 0.0) or 0.0),
                    "delta": float(item.get("delta", 0.0) or 0.0),
                }
                for item in history_hit_details
            ],
            "rule": {
                "enabled": history_enabled,
                "min_sample": history_min_sample,
                "min_bad_rate": history_min_bad_rate,
                "min_delta": history_min_delta,
                "delta_scale": history_delta_scale,
                "max_boost": history_max_boost,
            },
        },
        "severity_counts": severity_counts,
        "items": items[:12],
        "source_summary": {
            "compliance_pass": bool(report.get("pass", True)),
            "gap_risk_count": len(coverage_risks),
            "linked_module_count": len(linked_modules),
            "link_edges": total_edges,
            "trace_req_count": req_count,
            "trace_api_count": api_count,
            "trace_testcase_count": testcase_count,
            "history_window_size": history_window_size,
            "history_overall_bad_rate": float(history_overall_bad_rate),
        },
    }


def _normalize_intent(raw: str) -> IntentLabel:
    text = str(raw or "").strip().lower()
    if text in {"ui", "ui_interaction", "ui交互", "frontend"}:
        return "ui"
    if text in {"api", "api_logic", "接口", "backend"}:
        return "api"
    return "fallback"


def _log_observation(
    *,
    request_id: str,
    task_query: str,
    generation_mode: str,
    intent_label: str,
    final_status: str,
    retrieval_context: str,
    gap_hints: Dict[str, Any],
    impact_analysis: Dict[str, Any],
    risk_report: Dict[str, Any] | None = None,
) -> None:
    try:
        request_id = str(request_id or "") or uuid.uuid4().hex[:12]
        context_len = _safe_len(retrieval_context)
        weak_input = _safe_len(task_query) < 60
        gap_missing = gap_hints.get("missing_inputs", []) if isinstance(gap_hints, dict) else []
        gap_missing = gap_missing if isinstance(gap_missing, list) else [gap_missing]
        gap_hit = bool(gap_missing)

        impact_modules = []
        impact_high = 0
        impact_medium = 0
        if isinstance(impact_analysis, dict):
            for item in impact_analysis.get("current_involved_modules", []) or []:
                if not isinstance(item, dict):
                    continue
                module = str(item.get("module", "")).strip()
                if module:
                    impact_modules.append(module)
                    impact_high += 1
            for item in impact_analysis.get("potential_linked_modules", []) or []:
                if not isinstance(item, dict):
                    continue
                module = str(item.get("module", "")).strip()
                if module:
                    impact_modules.append(module)
                    impact_medium += 1

        relation_consumed = {
            "feature": False,
            "trace": False,
            "domain": False,
            "upstream": False,
            "downstream": False,
            "related": False,
        }
        if isinstance(impact_analysis, dict):
            relation_consumed.update(impact_analysis.get("relation_consumed", {}) or {})

        risk_payload = risk_report if isinstance(risk_report, dict) else {}
        severity_counts = risk_payload.get("severity_counts", {})
        if not isinstance(severity_counts, dict):
            severity_counts = {}
        history_adjustment = risk_payload.get("history_adjustment", {})
        if not isinstance(history_adjustment, dict):
            history_adjustment = {}

        payload = {
            "ts": _utc_now_iso(),
            "request_id": request_id,
            "generation_mode": generation_mode,
            "intent_label": intent_label,
            "final_status": final_status,
            "context_len": context_len,
            "weak_input": weak_input,
            "gap_hints_hit": gap_hit,
            "gap_hints_missing": [str(x) for x in gap_missing if str(x).strip()][:6],
            "impact_hit": bool(impact_modules),
            "impact_modules": impact_modules[:6],
            "impact_high_count": impact_high,
            "impact_medium_count": impact_medium,
            "relation_consumed": relation_consumed,
            "risk_overall_level": str(risk_payload.get("overall_level", "")).strip(),
            "risk_p0_count": int(severity_counts.get("P0", 0) or 0),
            "risk_p1_count": int(severity_counts.get("P1", 0) or 0),
            "risk_history_boost": float(history_adjustment.get("score_boost", 0.0) or 0.0),
        }

        log_path = PROJECT_ROOT / "data" / "observation_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def _heuristic_intent(task_text: str) -> IntentLabel:
    text = str(task_text or "").lower()
    ui_hits = ["ui", "页面", "交互", "按钮", "文案", "前端", "原型", "figma"]
    api_hits = ["api", "接口", "请求", "响应", "参数", "状态码", "鉴权", "幂等", "header"]
    ui_score = sum(1 for k in ui_hits if k in text)
    api_score = sum(1 for k in api_hits if k in text)
    if ui_score > api_score and ui_score > 0:
        return "ui"
    if api_score > ui_score and api_score > 0:
        return "api"
    return "fallback"


def _safe_build_prompt(template: str, context: str, task_query: str) -> str:
    raw = str(template or "")
    if "{context}" in raw and "{task}" in raw:
        try:
            return raw.replace("{context}", context).replace("{task}", task_query)
        except Exception:
            pass
    return (
        "【业务需求上下文】\n"
        f"{context}\n\n"
        "【待测核心逻辑】\n"
        f"{task_query}\n"
    )


def _generate_mode_instruction(mode_key: str) -> str:
    if mode_key == "field_validation":
        return (
            "当前模式: 字段校验用例（Field Validation）。\n"
            "必须重点覆盖必填、类型、长度、枚举、格式、边界值、默认值、兼容性、错误码与错误文案。\n"
            "输出保持结构化业务测试用例，严禁代码。"
        )
    return (
        "当前模式: 业务接口用例（Business API）。\n"
        "聚焦业务状态流转、角色权限、接口联动和上下游影响。\n"
        "输出保持结构化业务测试用例，严禁代码。"
    )


def _classifier_prompt(task_query: str) -> str:
    return (
        "你是测试需求路由器。请输出 JSON：\n"
        "{\n"
        '  "intent_label":"ui|api|fallback",\n'
        '  "is_mixed": true,\n'
        '  "confidence": 0.0,\n'
        '  "reason":"不超过40字",\n'
        '  "focus":["核心关注点1","核心关注点2"]\n'
        "}\n\n"
        "规则：\n"
        "- UI主导：页面交互、可视反馈、文案、多端一致性\n"
        "- API主导：业务状态流转、权限、幂等、一致性、错误码\n"
        "- 混合或不明确：fallback，且 is_mixed=true\n"
        "- 仅输出 JSON，不要解释\n\n"
        f"需求:\n{task_query}"
    )


def _reviewer_prompt(task_query: str, context: str, draft_md: str) -> str:
    return (
        "你是测试用例评审官（LLM Judge）。仅输出 JSON：\n"
        "{\n"
        '  "decision":"pass|fail",\n'
        '  "hard_fail_reasons":["..."],\n'
        '  "scores":{\n'
        '    "business_coverage":0,\n'
        '    "exception_coverage":0,\n'
        '    "assertion_specificity":0,\n'
        '    "executability":0,\n'
        '    "traceability":0,\n'
        '    "redundancy_control":0,\n'
        '    "total":0\n'
        "  },\n"
        '  "comments":["..."],\n'
        '  "missing_points":["..."],\n'
        '  "rewrite_instructions":["..."]\n'
        "}\n\n"
        "硬失败规则（命中任一必须 fail）：\n"
        "- 无异常流覆盖\n"
        "- 断言缺少可验证对象（状态/字段/文案/日志）\n"
        "- 步骤不可执行或缺少关键前置条件\n"
        "- 无证据引用（无法追溯到上下文）\n\n"
        "通过条件建议：\n"
        "- 无 hard_fail_reasons\n"
        "- total >= 24\n"
        "- business_coverage / exception_coverage / assertion_specificity >= 4\n\n"
        "只输出 JSON，不要解释。\n\n"
        f"【需求】\n{task_query}\n\n"
        f"【检索上下文】\n{context[:6000]}\n\n"
        f"【待评审草稿】\n{draft_md[:10000]}"
    )


def _generator_special_instruction(intent: IntentLabel) -> str:
    if intent == "ui":
        return (
            "垂直指令(UI): 优先覆盖页面入口、交互路径、状态反馈、跨语言文案、前后端联动。"
            "重点输出用户可见行为与断言。"
        )
    if intent == "api":
        return (
            "垂直指令(API): 优先覆盖业务接口流转、权限与角色、上下游一致性、错误码与异常场景。"
            "重点输出前置状态与明确断言。"
        )
    return (
        "垂直指令(Fallback): 采用业务全链路兜底策略，兼顾 UI 交互和 API 逻辑，突出跨模块联动。"
    )


def _format_human_inputs_block(human_inputs: Dict[str, Any], run_context: Dict[str, Any]) -> str:
    scope = str(human_inputs.get("scope", "")).strip()
    exclusions = str(human_inputs.get("exclusions", "")).strip()
    risk_focus = str(human_inputs.get("risk_focus", "")).strip()
    priority_modules = human_inputs.get("priority_modules", [])
    if not isinstance(priority_modules, list):
        priority_modules = _parse_human_input_list(priority_modules)
    priority_modules = [str(x).strip() for x in priority_modules if str(x).strip()][:8]
    release = str(human_inputs.get("release", "")).strip()
    approved_only = bool(human_inputs.get("approved_only", True))
    must_cover = human_inputs.get("must_cover", [])
    must_not_cover = human_inputs.get("must_not_cover", [])
    risk_tags = human_inputs.get("risk_tags", [])
    if not isinstance(must_cover, list):
        must_cover = _parse_human_input_list(must_cover)
    if not isinstance(must_not_cover, list):
        must_not_cover = _parse_human_input_list(must_not_cover)
    if not isinstance(risk_tags, list):
        risk_tags = _parse_human_input_list(risk_tags)
    confirmed_by_user = bool(run_context.get("confirmed_by_user", False))
    confirmation_ts = str(run_context.get("confirmation_ts", "")).strip()

    return (
        "【人机协作前置输入（高优先级）】\n"
        f"- 已确认执行: {'是' if confirmed_by_user else '否'}\n"
        f"- 确认时间: {confirmation_ts or '-'}\n"
        f"- 测试范围(scope): {scope or '-'}\n"
        f"- 排除项(exclusions): {exclusions or '-'}\n"
        f"- 风险关注(risk_focus): {risk_focus or '-'}\n"
        f"- 优先模块(priority_modules): {','.join(priority_modules) if priority_modules else '-'}\n"
        f"- 发布版本(release): {release or '-'}\n"
        f"- 仅已审核(approved_only): {'是' if approved_only else '否'}\n"
        f"- 必须覆盖(must_cover): {','.join([str(x) for x in must_cover[:8]]) if must_cover else '-'}\n"
        f"- 必须排除(must_not_cover): {','.join([str(x) for x in must_not_cover[:8]]) if must_not_cover else '-'}\n"
        f"- 风险标签(risk_tags): {','.join([str(x) for x in risk_tags[:8]]) if risk_tags else '-'}\n"
        "请严格按上述边界生成，不得擅自扩展排除项内场景。\n"
    )


def _compose_retrieval_query(
    task_query: str, intent: IntentLabel, review_comments: List[str], iteration: int
) -> str:
    intent_hint_map = {
        "ui": "UI 页面交互 文案 状态提示 联动",
        "api": "API 接口业务逻辑 参数约束 状态流转 错误码",
        "fallback": "业务全链路 跨模块联动 场景覆盖 异常处理",
    }
    review_hint = ""
    if review_comments and iteration > 0:
        review_hint = " ".join(review_comments[:3])
    return f"{intent_hint_map[intent]} {task_query} {review_hint}".strip()


def _build_generator_prompt(
    *,
    base_template: str,
    mode_instruction: str,
    task_query: str,
    context: str,
    intent: IntentLabel,
    review_comments: List[str],
    iteration: int,
    contracts: Dict[str, Any],
    mapping_rules: Dict[str, Any],
    coverage_matrix: Dict[str, Any],
    link_analysis: Dict[str, Any],
    human_inputs: Dict[str, Any],
    run_context: Dict[str, Any],
) -> str:
    base_prompt = _safe_build_prompt(base_template, context=context, task_query=task_query)
    support_block = _format_generation_support_block(
        contracts=contracts,
        mapping_rules=mapping_rules,
        coverage_matrix=coverage_matrix,
        link_analysis=link_analysis,
    )
    output_contract = (
        "【输出结构契约（必须严格遵守）】\n"
        "# 测试范围与假设\n"
        "# 缺失信息清单（如无写“无”）\n"
        "# 功能测试用例\n"
        "# 业务接口测试用例\n"
        "# 集成异常与补偿用例\n"
        "# 用例清单（统一编号）\n"
        "每条用例必须包含：\n"
        "- 用例ID\n"
        "- 场景标题\n"
        "- 风险等级(P0/P1/P2)\n"
        "- 前置数据/状态\n"
        "- 执行步骤（业务动作）\n"
        "- 预期结果（至少覆盖：接口响应、业务状态、数据落库/日志）\n"
        "- 异常分支（如无写“无”）\n"
        "- 关联证据（必须标注来源类型：internal_spec/mapping_rule/third_party_spec）\n"
        "数量与分组约束：\n"
        "- 功能测试用例不少于 12 条\n"
        "- 业务接口测试用例不少于 10 条\n"
        "- 集成异常与补偿用例不少于 6 条（可与前两类重叠）\n"
        "- 商品同步、库存同步、订单同步三个链路必须覆盖\n"
        "质量约束：必须覆盖正常流、边界流、异常流、权限流；禁止模糊断言与伪代码。\n"
    )
    feedback_block = ""
    if review_comments and iteration > 0:
        feedback_block = (
            "\n【上轮评审反馈（必须修复）】\n"
            + "\n".join(f"- {c}" for c in review_comments[:8])
            + "\n"
        )
    human_block = _format_human_inputs_block(human_inputs, run_context)
    return (
        f"{base_prompt}\n\n"
        "【本次生成模式追加指令（高优先级）】\n"
        f"{mode_instruction}\n\n"
        f"{human_block}\n"
        f"{support_block}\n"
        "【智能体垂直指令（高优先级）】\n"
        f"{_generator_special_instruction(intent)}\n"
        f"{output_contract}\n"
        f"{feedback_block}\n"
        "请输出结构化 Markdown 测试用例，不要输出代码。"
    )


def _validate_constraint_compliance(
    *,
    draft_md: str,
    human_inputs: Dict[str, Any],
    llm: Any = None,
) -> Dict[str, Any]:
    report = evaluate_constraint_compliance(
        draft_md=draft_md,
        human_inputs=human_inputs,
        llm=llm,
    )
    missing_items = report.get("missing_items", {})
    if not isinstance(missing_items, dict):
        missing_items = {}
    if "must_cover" not in missing_items:
        missing_items["must_cover"] = list(missing_items.get("include_all", []) or [])
    report["missing_items"] = missing_items

    hit_items = report.get("hit_items", {})
    if not isinstance(hit_items, dict):
        hit_items = {}
    if "must_cover" not in hit_items:
        hit_items["must_cover"] = list(hit_items.get("include_all", []) or [])
    report["hit_items"] = hit_items
    return report


def _validate_review_result(data: Dict[str, Any]) -> ReviewResult:
    scores_raw = data.get("scores", {}) if isinstance(data.get("scores"), dict) else {}
    business_coverage = _safe_int(scores_raw.get("business_coverage"), default=2)
    exception_coverage = _safe_int(
        scores_raw.get("exception_coverage", scores_raw.get("exception_flow")),
        default=2,
    )
    assertion_specificity = _safe_int(
        scores_raw.get("assertion_specificity", scores_raw.get("assertion_clarity")),
        default=2,
    )
    executability = _safe_int(scores_raw.get("executability"), default=2)
    traceability = _safe_int(scores_raw.get("traceability"), default=2)
    redundancy_control = _safe_int(scores_raw.get("redundancy_control"), default=2)
    total_score = (
        int(business_coverage)
        + int(exception_coverage)
        + int(assertion_specificity)
        + int(executability)
        + int(traceability)
        + int(redundancy_control)
    )
    scores: ReviewScores = {
        "business_coverage": business_coverage,
        "exception_coverage": exception_coverage,
        "assertion_specificity": assertion_specificity,
        "executability": executability,
        "traceability": traceability,
        "redundancy_control": redundancy_control,
        # compatibility aliases
        "exception_flow": exception_coverage,
        "assertion_clarity": assertion_specificity,
        "total": total_score,
    }

    hard_fail_reasons = data.get("hard_fail_reasons", [])
    if not isinstance(hard_fail_reasons, list):
        hard_fail_reasons = [str(hard_fail_reasons)]
    hard_fail_reasons = [str(x).strip() for x in hard_fail_reasons if str(x).strip()][:8]

    decision_raw = str(data.get("decision", "")).strip().lower()
    auto_pass = (
        not hard_fail_reasons
        and scores["total"] >= 24
        and scores["business_coverage"] >= 4
        and scores["exception_coverage"] >= 4
        and scores["assertion_specificity"] >= 4
    )
    auto_decision = "pass" if auto_pass else "fail"
    if decision_raw not in {"pass", "fail"}:
        decision_raw = auto_decision
    elif decision_raw == "pass" and not auto_pass:
        decision_raw = "fail"

    comments = data.get("comments", [])
    if not isinstance(comments, list):
        comments = [str(comments)]
    comments = [str(x).strip() for x in comments if str(x).strip()][:8]

    missing_points = data.get("missing_points", [])
    if not isinstance(missing_points, list):
        missing_points = [str(missing_points)]
    missing_points = [str(x).strip() for x in missing_points if str(x).strip()][:8]

    rewrite_instructions = data.get("rewrite_instructions", [])
    if not isinstance(rewrite_instructions, list):
        rewrite_instructions = [str(rewrite_instructions)]
    rewrite_instructions = [str(x).strip() for x in rewrite_instructions if str(x).strip()][:8]

    return {
        "decision": "pass" if decision_raw == "pass" else "fail",
        "scores": scores,
        "hard_fail_reasons": hard_fail_reasons,
        "comments": comments,
        "missing_points": missing_points,
        "rewrite_instructions": rewrite_instructions,
    }


def _call_get_augmented_context(
    get_augmented_context: Callable[..., str],
    query: str,
    retrieval_policy: RetrievalPolicy | None,
) -> str:
    policy = retrieval_policy or {}
    try:
        sig = inspect.signature(get_augmented_context)
        if "retrieval_policy" in sig.parameters:
            return str(get_augmented_context(query, retrieval_policy=policy) or "")
        if len(sig.parameters) >= 2:
            return str(get_augmented_context(query, policy) or "")
        return str(get_augmented_context(query) or "")
    except TypeError:
        return str(get_augmented_context(query) or "")


def _fallback_linear_workflow(
    *,
    task_query: str,
    get_augmented_context: Callable[..., str],
    llm: Any,
    universal_template: str,
    generation_mode: str,
    recommended_mode_lock: str | None,
    human_inputs: Dict[str, Any] | None,
    run_context: Dict[str, Any] | None,
    max_iterations: int,
    request_id: str,
    retrieval_policy: RetrievalPolicy | None = None,
) -> Dict[str, Any]:
    normalized_mode = _normalize_generation_mode(generation_mode)
    normalized_mode_lock = _normalize_optional_generation_mode(recommended_mode_lock)
    normalized_human_inputs = _normalize_human_inputs(human_inputs, retrieval_policy)
    normalized_run_context = run_context if isinstance(run_context, dict) else {}
    normalized_run_context = {
        "confirmed_by_user": bool(normalized_run_context.get("confirmed_by_user", False)),
        "confirmation_ts": str(normalized_run_context.get("confirmation_ts", "")).strip(),
    }
    if normalized_run_context["confirmed_by_user"] and not normalized_run_context["confirmation_ts"]:
        normalized_run_context["confirmation_ts"] = _utc_now_iso()
    context = ""
    retrieval_error = None
    try:
        context = _call_get_augmented_context(get_augmented_context, task_query, retrieval_policy)
    except Exception as exc:
        retrieval_error = f"检索失败，已降级空上下文: {exc}"
    mode_instruction = _generate_mode_instruction(normalized_mode)
    draft = ""
    error_text = retrieval_error
    support_bundle = _build_generation_support(task_query=task_query, retrieval_context=context)
    try:
        prompt = _build_generator_prompt(
            base_template=universal_template,
            mode_instruction=mode_instruction,
            task_query=task_query,
            context=context,
            intent=_heuristic_intent(task_query),
            review_comments=[],
            iteration=0,
            contracts=support_bundle.get("contracts", {}),
            mapping_rules=support_bundle.get("mapping_rules", {}),
            coverage_matrix=support_bundle.get("coverage_matrix", {}),
            link_analysis=support_bundle.get("link_analysis", {}),
            human_inputs=normalized_human_inputs,
            run_context=normalized_run_context,
        )
        response = llm.invoke(prompt)
        draft = _normalize_llm_content(response)
    except Exception as exc:
        base = f"{error_text} | " if error_text else ""
        error_text = f"{base}线性回退流程执行失败: {exc}"

    final_status: Literal["success", "success_with_warning", "failed"] = "success_with_warning"
    if not draft.strip():
        final_status = "failed"
    review_result = {
        "decision": "pass",
        "scores": {
            "business_coverage": 3,
            "exception_coverage": 3,
            "assertion_specificity": 3,
            "executability": 3,
            "traceability": 3,
            "redundancy_control": 3,
            "exception_flow": 3,
            "assertion_clarity": 3,
            "total": 18,
        },
        "hard_fail_reasons": [],
        "comments": ["LangGraph 不可用，已回退线性流程。"],
        "missing_points": [],
        "rewrite_instructions": [],
    }
    compliance_report = _validate_constraint_compliance(
        draft_md=draft,
        human_inputs=normalized_human_inputs,
        llm=llm,
    )
    if not compliance_report.get("pass", True):
        review_result = _validate_review_result(
            {
                "decision": "fail",
                "hard_fail_reasons": ["前置协作约束未满足"],
                "scores": {
                    "business_coverage": 0,
                    "exception_coverage": 0,
                    "assertion_specificity": 0,
                    "executability": 0,
                    "traceability": 0,
                    "redundancy_control": 0,
                    "total": 0,
                },
                "comments": list(compliance_report.get("reasons", [])),
                "missing_points": [
                    *list((compliance_report.get("missing_items", {}) or {}).get("must_cover", [])),
                    *list((compliance_report.get("missing_items", {}) or {}).get("risk_tags", [])),
                ],
                "rewrite_instructions": list(compliance_report.get("rewrite_instructions", [])),
            }
        )
    recommended_mode = (
        _normalize_generation_mode(normalized_mode_lock)
        if normalized_mode_lock
        else _recommend_generation_mode(normalized_mode, review_result)
    )
    intent_label = _heuristic_intent(task_query)
    gap_hints = _build_gap_hints(
        task_query=task_query,
        retrieval_context=context,
        intent_label=intent_label,
        generation_mode=recommended_mode,
        review_result=review_result,
        retrieval_policy=retrieval_policy,
        llm=llm,
    )
    impact_analysis = _build_impact_analysis(
        task_query=task_query,
        retrieval_context=context,
        retrieval_policy=retrieval_policy,
    )
    fallback_link_analysis = support_bundle.get("link_analysis", {}) if isinstance(support_bundle, dict) else {}
    fallback_link_edges = list((fallback_link_analysis or {}).get("link_edges", []) or [])
    fallback_trace_refs = dict((fallback_link_analysis or {}).get("trace_refs", {}) or {})
    fallback_link_summary = dict((fallback_link_analysis or {}).get("link_summary", {}) or {})
    risk_report = _build_risk_report(
        retrieval_context_len=len(context),
        compliance_report=compliance_report if isinstance(compliance_report, dict) else {},
        gap_hints=gap_hints if isinstance(gap_hints, dict) else {},
        impact_analysis=impact_analysis if isinstance(impact_analysis, dict) else {},
        link_summary=fallback_link_summary,
        trace_refs=fallback_trace_refs,
    )
    result = {
        "request_id": request_id,
        "final_testcases_md": draft,
        "retrieval_context": context,
        "retrieval_context_len": len(context),
        "intent_label": intent_label,
        "intent_confidence": 0.55,
        "iteration": 0,
        "max_iterations": max_iterations,
        "route_history": ["fallback_linear"],
        "review_result": review_result,
        "review_passed": review_result.get("decision") == "pass" and bool(draft.strip()),
        "recommended_mode": recommended_mode,
        "recommended_mode_lock": normalized_mode_lock,
        "human_inputs": normalized_human_inputs,
        "run_context": normalized_run_context,
        "compliance_report": compliance_report,
        "final_status": final_status,
        "error": error_text,
        "langgraph_enabled": False,
        "gap_hints": gap_hints,
        "impact_analysis": impact_analysis,
        "risk_report": risk_report,
        "contracts": support_bundle.get("contracts", {}),
        "mapping_rules": support_bundle.get("mapping_rules", {}),
        "coverage_matrix": support_bundle.get("coverage_matrix", {}),
        "link_edges": fallback_link_edges,
        "trace_refs": fallback_trace_refs,
        "link_summary": fallback_link_summary,
    }
    _log_observation(
        request_id=request_id,
        task_query=task_query,
        generation_mode=recommended_mode,
        intent_label=intent_label,
        final_status=final_status,
        retrieval_context=context,
        gap_hints=gap_hints,
        impact_analysis=impact_analysis,
        risk_report=risk_report,
    )
    record_badcase_event(
        request_id=request_id,
        task_query=task_query,
        generation_mode=recommended_mode,
        intent_label=intent_label,
        final_status=final_status,
        risk_report=risk_report,
        route_history=result.get("route_history", []),
        recommended_mode=recommended_mode,
        retrieval_context_len=len(context),
        project_root=PROJECT_ROOT,
    )
    replay_report = build_badcase_replay_report(project_root=PROJECT_ROOT, window_days=30)
    tuning = auto_tune_rule_templates_from_replay(replay_report, project_root=PROJECT_ROOT)
    replay_report["rule_tuning"] = tuning
    result["badcase_replay"] = replay_report
    return result


def run_testcase_workflow(
    *,
    task_query: str,
    get_augmented_context: Callable[..., str],
    llm: Any,
    universal_template: str,
    generation_mode: str,
    recommended_mode_lock: str | None = None,
    human_inputs: Dict[str, Any] | None = None,
    max_iterations: int = 2,
    request_id: str | None = None,
    retrieval_policy: RetrievalPolicy | None = None,
) -> Dict[str, Any]:
    rid = request_id or uuid.uuid4().hex[:12]
    normalized_task = str(task_query or "").strip()
    normalized_generation_mode = _normalize_generation_mode(generation_mode)
    normalized_mode_lock = _normalize_optional_generation_mode(recommended_mode_lock)
    normalized_human_inputs = _normalize_human_inputs(human_inputs, retrieval_policy)
    normalized_run_context = _normalize_run_context(human_inputs)
    if not normalized_task:
        raise ValueError("task_query 不能为空。")

    if not LANGGRAPH_AVAILABLE or StateGraph is None:
        return _fallback_linear_workflow(
            task_query=normalized_task,
            get_augmented_context=get_augmented_context,
            llm=llm,
            universal_template=universal_template,
            generation_mode=generation_mode,
            recommended_mode_lock=recommended_mode_lock,
            human_inputs=normalized_human_inputs,
            run_context=normalized_run_context,
            max_iterations=max_iterations,
            request_id=rid,
            retrieval_policy=retrieval_policy,
        )

    def classifier_node(state: QAState) -> QAState:
        history = list(state.get("route_history", []))
        history.append("classifier")
        try:
            prompt = _classifier_prompt(state.get("user_requirement_normalized", ""))
            response = llm.invoke(prompt)
            raw = _normalize_llm_content(response)
            parsed = _safe_json_loads(raw)

            intent = _normalize_intent(parsed.get("intent_label", ""))
            if parsed.get("intent_label") is None:
                intent = _heuristic_intent(state.get("user_requirement_normalized", ""))

            confidence = parsed.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            reason = str(parsed.get("reason", "")).strip() or "classifier_default"
        except Exception as exc:
            intent = _heuristic_intent(state.get("user_requirement_normalized", ""))
            confidence = 0.0
            reason = f"classifier_error: {exc}"

        generator_map: Dict[IntentLabel, GeneratorName] = {
            "ui": "ui_generator",
            "api": "api_generator",
            "fallback": "fallback_generator",
        }

        return {
            **state,
            "route_history": history,
            "intent_label": intent,
            "intent_confidence": confidence,
            "classifier_reason": reason,
            "active_generator": generator_map[intent],
        }

    def _generator_node(state: QAState, intent: IntentLabel, name: GeneratorName) -> QAState:
        history = list(state.get("route_history", []))
        history.append(name)

        review_comments = list(state.get("review_comments", []))
        iteration = int(state.get("iteration", 0) or 0)
        task = state.get("user_requirement_normalized", "")
        effective_mode = _resolve_effective_generation_mode(
            normalized_generation_mode,
            state.get("recommended_mode")
            if not state.get("recommended_mode_lock")
            else state.get("recommended_mode_lock"),
        )
        mode_instruction = _generate_mode_instruction(effective_mode)
        retrieval_query = _compose_retrieval_query(
            task_query=task,
            intent=intent,
            review_comments=review_comments,
            iteration=iteration,
        )
        try:
            context = _call_get_augmented_context(
                get_augmented_context,
                retrieval_query,
                state.get("retrieval_policy", retrieval_policy or {}),
            )
            support_bundle = _build_generation_support(task_query=task, retrieval_context=context)
            prompt = _build_generator_prompt(
                base_template=universal_template,
                mode_instruction=mode_instruction,
                task_query=task,
                context=context,
                intent=intent,
                review_comments=review_comments,
                iteration=iteration,
                contracts=support_bundle.get("contracts", {}),
                mapping_rules=support_bundle.get("mapping_rules", {}),
                coverage_matrix=support_bundle.get("coverage_matrix", {}),
                link_analysis=support_bundle.get("link_analysis", {}),
                human_inputs=state.get("human_inputs", {}),
                run_context=state.get("run_context", {}),
            )
            response = llm.invoke(prompt)
            draft = _normalize_llm_content(response)
            error = ""
        except Exception as exc:
            context = str(state.get("retrieval_context", "") or "")
            draft = str(state.get("draft_testcases_md", "") or "")
            error = f"{name} 执行失败: {exc}"
            support_bundle = {
                "contracts": state.get("contracts", {}) if isinstance(state.get("contracts"), dict) else {},
                "mapping_rules": state.get("mapping_rules", {})
                if isinstance(state.get("mapping_rules"), dict)
                else {},
                "coverage_matrix": state.get("coverage_matrix", {})
                if isinstance(state.get("coverage_matrix"), dict)
                else {},
                "link_analysis": {
                    "link_edges": state.get("link_edges", []) if isinstance(state.get("link_edges"), list) else [],
                    "trace_refs": state.get("trace_refs", {})
                    if isinstance(state.get("trace_refs"), dict)
                    else {},
                    "link_summary": state.get("link_summary", {})
                    if isinstance(state.get("link_summary"), dict)
                    else {},
                },
            }

        return {
            **state,
            "route_history": history,
            "active_generator": name,
            "generator_prompt_name": name,
            "retrieval_query": retrieval_query,
            "retrieval_context": context,
            "retrieval_context_len": len(context),
            "retrieval_meta": {
                "intent": intent,
                "iteration": iteration,
                "effective_mode": effective_mode,
            },
            "draft_testcases_md": draft,
            "contracts": support_bundle.get("contracts", {}),
            "mapping_rules": support_bundle.get("mapping_rules", {}),
            "coverage_matrix": support_bundle.get("coverage_matrix", {}),
            "link_edges": list((support_bundle.get("link_analysis", {}) or {}).get("link_edges", []) or []),
            "trace_refs": dict((support_bundle.get("link_analysis", {}) or {}).get("trace_refs", {}) or {}),
            "link_summary": dict((support_bundle.get("link_analysis", {}) or {}).get("link_summary", {}) or {}),
            "error": error,
        }

    def ui_generator_node(state: QAState) -> QAState:
        return _generator_node(state, intent="ui", name="ui_generator")

    def api_generator_node(state: QAState) -> QAState:
        return _generator_node(state, intent="api", name="api_generator")

    def fallback_generator_node(state: QAState) -> QAState:
        return _generator_node(state, intent="fallback", name="fallback_generator")

    def constraint_validator_node(state: QAState) -> QAState:
        history = list(state.get("route_history", []))
        history.append("constraint_validator")

        draft = str(state.get("draft_testcases_md", "") or "")
        report = _validate_constraint_compliance(
            draft_md=draft,
            human_inputs=state.get("human_inputs", {}),
            llm=llm,
        )
        if report.get("pass", True):
            return {
                **state,
                "route_history": history,
                "compliance_report": report,
                "error": "",
            }

        review_result = _validate_review_result(
            {
                "decision": "fail",
                "hard_fail_reasons": ["前置协作约束未满足"],
                "scores": {
                    "business_coverage": 0,
                    "exception_coverage": 0,
                    "assertion_specificity": 0,
                    "executability": 0,
                    "traceability": 0,
                    "redundancy_control": 0,
                    "total": 0,
                },
                "comments": list(report.get("reasons", [])),
                "missing_points": [
                    *list((report.get("missing_items", {}) or {}).get("must_cover", [])),
                    *list((report.get("missing_items", {}) or {}).get("risk_tags", [])),
                ],
                "rewrite_instructions": list(report.get("rewrite_instructions", [])),
            }
        )
        feedback = review_result.get("rewrite_instructions", []) or review_result.get("comments", [])
        return {
            **state,
            "route_history": history,
            "compliance_report": report,
            "review_result": review_result,
            "review_passed": False,
            "review_comments": [str(x) for x in feedback if str(x).strip()][:8],
            "error": "",
        }

    def reviewer_node(state: QAState) -> QAState:
        history = list(state.get("route_history", []))
        history.append("reviewer")

        task = state.get("user_requirement_normalized", "")
        context = state.get("retrieval_context", "")
        draft = state.get("draft_testcases_md", "")
        try:
            response = llm.invoke(_reviewer_prompt(task_query=task, context=context, draft_md=draft))
            review_raw = _normalize_llm_content(response)
            parsed = _safe_json_loads(review_raw)
            review_result = _validate_review_result(parsed)
            review_passed = review_result["decision"] == "pass"
            feedback = review_result.get("rewrite_instructions", []) or review_result.get("comments", [])
            recommended_mode = _recommend_generation_mode(
                _resolve_effective_generation_mode(normalized_generation_mode, state.get("recommended_mode")),
                review_result,
            )
            if state.get("recommended_mode_lock"):
                recommended_mode = _normalize_generation_mode(state.get("recommended_mode_lock"))
            error = ""
        except Exception as exc:
            review_raw = ""
            review_result = _validate_review_result(
                {
                    "decision": "fail",
                    "hard_fail_reasons": ["评审节点异常，结果不可信"],
                    "scores": {
                        "business_coverage": 0,
                        "exception_coverage": 0,
                        "assertion_specificity": 0,
                        "executability": 0,
                        "traceability": 0,
                        "redundancy_control": 0,
                        "exception_flow": 0,
                        "assertion_clarity": 0,
                        "total": 0,
                    },
                    "comments": [f"reviewer_error: {exc}"],
                    "missing_points": ["评审节点异常，建议重试。"],
                    "rewrite_instructions": ["请重试生成，检查模型与网络状态。"],
                }
            )
            review_passed = False
            feedback = review_result.get("rewrite_instructions", [])
            recommended_mode = _resolve_effective_generation_mode(
                normalized_generation_mode,
                state.get("recommended_mode"),
            )
            if state.get("recommended_mode_lock"):
                recommended_mode = _normalize_generation_mode(state.get("recommended_mode_lock"))
            error = f"reviewer 执行失败: {exc}"

        return {
            **state,
            "route_history": history,
            "review_raw_text": review_raw,
            "review_result": review_result,
            "review_passed": review_passed,
            "recommended_mode": recommended_mode,
            "review_comments": [str(x) for x in feedback if str(x).strip()][:8],
            "error": error,
        }

    def retry_control_node(state: QAState) -> QAState:
        history = list(state.get("route_history", []))
        history.append("retry_control")
        return {
            **state,
            "route_history": history,
            "iteration": int(state.get("iteration", 0) or 0) + 1,
        }

    def finalize_node(state: QAState) -> QAState:
        history = list(state.get("route_history", []))
        history.append("finalize")

        review_passed = bool(state.get("review_passed", False))
        draft = str(state.get("draft_testcases_md", "")).strip()
        status: Literal["success", "success_with_warning", "failed"] = "success"
        if not draft:
            status = "failed"
        elif not review_passed:
            status = "success_with_warning"

        return {
            **state,
            "route_history": history,
            "final_testcases_md": draft,
            "final_status": status,
        }

    def route_after_classifier(state: QAState) -> str:
        intent = _normalize_intent(state.get("intent_label", "fallback"))
        if intent == "ui":
            return "ui_generator"
        if intent == "api":
            return "api_generator"
        return "fallback_generator"

    def route_after_generator(state: QAState) -> str:
        error_text = str(state.get("error", "")).strip()
        if error_text.lower() in {"none", "null"}:
            error_text = ""
        if error_text:
            return "finalize"
        return "constraint_validator"

    def route_after_constraint_validator(state: QAState) -> str:
        report = state.get("compliance_report", {})
        passed = bool(report.get("pass", True)) if isinstance(report, dict) else True
        if passed:
            return "reviewer"
        return route_after_reviewer(state)

    def route_after_reviewer(state: QAState) -> str:
        if bool(state.get("review_passed", False)):
            return "finalize"
        iteration = int(state.get("iteration", 0) or 0)
        max_iter = int(state.get("max_iterations", 2) or 2)
        if iteration >= max_iter:
            return "finalize"
        return "retry_control"

    def route_after_retry(state: QAState) -> str:
        active = str(state.get("active_generator", "fallback_generator"))
        if active == "ui_generator":
            return "ui_generator"
        if active == "api_generator":
            return "api_generator"
        return "fallback_generator"

    builder: Any = StateGraph(QAState)
    builder.add_node("classifier", classifier_node)
    builder.add_node("ui_generator", ui_generator_node)
    builder.add_node("api_generator", api_generator_node)
    builder.add_node("fallback_generator", fallback_generator_node)
    builder.add_node("constraint_validator", constraint_validator_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("retry_control", retry_control_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("classifier")

    builder.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "ui_generator": "ui_generator",
            "api_generator": "api_generator",
            "fallback_generator": "fallback_generator",
        },
    )
    builder.add_conditional_edges(
        "ui_generator",
        route_after_generator,
        {"constraint_validator": "constraint_validator", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "api_generator",
        route_after_generator,
        {"constraint_validator": "constraint_validator", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "fallback_generator",
        route_after_generator,
        {"constraint_validator": "constraint_validator", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "constraint_validator",
        route_after_constraint_validator,
        {"reviewer": "reviewer", "retry_control": "retry_control", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"retry_control": "retry_control", "finalize": "finalize"},
    )
    builder.add_conditional_edges(
        "retry_control",
        route_after_retry,
        {
            "ui_generator": "ui_generator",
            "api_generator": "api_generator",
            "fallback_generator": "fallback_generator",
        },
    )
    builder.add_edge("finalize", END)

    graph = builder.compile()
    initial_state: QAState = {
        "request_id": rid,
        "user_requirement_raw": task_query,
        "user_requirement_normalized": normalized_task,
        "iteration": 0,
        "max_iterations": max(1, int(max_iterations)),
        "route_history": [],
        "review_comments": [],
        "final_status": "failed",
        "recommended_mode": normalized_generation_mode,
        "recommended_mode_lock": normalized_mode_lock,
        "human_inputs": normalized_human_inputs,
        "run_context": normalized_run_context,
        "retrieval_policy": dict(retrieval_policy or {}),
    }

    try:
        final_state: QAState = graph.invoke(initial_state)
    except Exception as exc:
        fallback = _fallback_linear_workflow(
            task_query=normalized_task,
            get_augmented_context=get_augmented_context,
            llm=llm,
            universal_template=universal_template,
            generation_mode=generation_mode,
            recommended_mode_lock=recommended_mode_lock,
            human_inputs=normalized_human_inputs,
            run_context=normalized_run_context,
            max_iterations=max_iterations,
            request_id=rid,
            retrieval_policy=retrieval_policy,
        )
        fallback["error"] = f"LangGraph 执行异常，已自动回退: {exc}"
        return fallback
    final_mode = _resolve_effective_generation_mode(
        normalized_generation_mode,
        final_state.get("recommended_mode")
        if not final_state.get("recommended_mode_lock")
        else final_state.get("recommended_mode_lock"),
    )
    gap_hints = _build_gap_hints(
        task_query=normalized_task,
        retrieval_context=str(final_state.get("retrieval_context", "") or ""),
        intent_label=str(final_state.get("intent_label", "fallback")),
        generation_mode=final_mode,
        review_result=final_state.get("review_result", {}),
        retrieval_policy=retrieval_policy,
        llm=llm,
    )
    impact_analysis = _build_impact_analysis(
        task_query=normalized_task,
        retrieval_context=str(final_state.get("retrieval_context", "") or ""),
        retrieval_policy=retrieval_policy,
    )
    contracts = final_state.get("contracts", {}) if isinstance(final_state.get("contracts"), dict) else {}
    mapping_rules = (
        final_state.get("mapping_rules", {})
        if isinstance(final_state.get("mapping_rules"), dict)
        else {}
    )
    coverage_matrix = (
        final_state.get("coverage_matrix", {})
        if isinstance(final_state.get("coverage_matrix"), dict)
        else {}
    )
    link_edges = final_state.get("link_edges", []) if isinstance(final_state.get("link_edges"), list) else []
    trace_refs = final_state.get("trace_refs", {}) if isinstance(final_state.get("trace_refs"), dict) else {}
    link_summary = final_state.get("link_summary", {}) if isinstance(final_state.get("link_summary"), dict) else {}
    if not contracts or not mapping_rules or not coverage_matrix or not link_edges:
        support_bundle = _build_generation_support(
            task_query=normalized_task,
            retrieval_context=str(final_state.get("retrieval_context", "") or ""),
        )
        if not contracts:
            contracts = support_bundle.get("contracts", {})
        if not mapping_rules:
            mapping_rules = support_bundle.get("mapping_rules", {})
        if not coverage_matrix:
            coverage_matrix = support_bundle.get("coverage_matrix", {})
        link_analysis = support_bundle.get("link_analysis", {}) if isinstance(support_bundle, dict) else {}
        if not link_edges:
            link_edges = list((link_analysis or {}).get("link_edges", []) or [])
        if not trace_refs:
            trace_refs = dict((link_analysis or {}).get("trace_refs", {}) or {})
        if not link_summary:
            link_summary = dict((link_analysis or {}).get("link_summary", {}) or {})
    risk_report = _build_risk_report(
        retrieval_context_len=int(final_state.get("retrieval_context_len", 0) or 0),
        compliance_report=(
            final_state.get("compliance_report", {})
            if isinstance(final_state.get("compliance_report"), dict)
            else {}
        ),
        gap_hints=gap_hints if isinstance(gap_hints, dict) else {},
        impact_analysis=impact_analysis if isinstance(impact_analysis, dict) else {},
        link_summary=link_summary if isinstance(link_summary, dict) else {},
        trace_refs=trace_refs if isinstance(trace_refs, dict) else {},
    )
    result = {
        "request_id": final_state.get("request_id", rid),
        "intent_label": final_state.get("intent_label", "fallback"),
        "intent_confidence": float(final_state.get("intent_confidence", 0.0) or 0.0),
        "classifier_reason": final_state.get("classifier_reason", ""),
        "retrieval_query": final_state.get("retrieval_query", ""),
        "retrieval_context": final_state.get("retrieval_context", ""),
        "retrieval_context_len": int(final_state.get("retrieval_context_len", 0) or 0),
        "draft_testcases_md": final_state.get("draft_testcases_md", ""),
        "final_testcases_md": final_state.get("final_testcases_md", ""),
        "review_result": final_state.get("review_result", {}),
        "review_passed": bool(final_state.get("review_passed", False)),
        "compliance_report": final_state.get("compliance_report", {}),
        "recommended_mode": final_mode,
        "recommended_mode_lock": _normalize_optional_generation_mode(
            final_state.get("recommended_mode_lock")
        ),
        "human_inputs": _normalize_human_inputs(
            final_state.get("human_inputs", {}),
            retrieval_policy,
        ),
        "run_context": _normalize_run_context(
            {
                "run_context": final_state.get("run_context", {}),
            }
        ),
        "iteration": int(final_state.get("iteration", 0) or 0),
        "max_iterations": int(final_state.get("max_iterations", max_iterations) or max_iterations),
        "route_history": list(final_state.get("route_history", [])),
        "final_status": final_state.get("final_status", "failed"),
        "error": final_state.get("error"),
        "langgraph_enabled": LANGGRAPH_AVAILABLE,
        "gap_hints": gap_hints,
        "impact_analysis": impact_analysis,
        "risk_report": risk_report,
        "contracts": contracts,
        "mapping_rules": mapping_rules,
        "coverage_matrix": coverage_matrix,
        "link_edges": link_edges,
        "trace_refs": trace_refs,
        "link_summary": link_summary,
    }
    _log_observation(
        request_id=result.get("request_id", rid),
        task_query=normalized_task,
        generation_mode=final_mode,
        intent_label=str(result.get("intent_label", "fallback")),
        final_status=str(result.get("final_status", "failed")),
        retrieval_context=str(result.get("retrieval_context", "") or ""),
        gap_hints=gap_hints,
        impact_analysis=impact_analysis,
        risk_report=risk_report,
    )
    record_badcase_event(
        request_id=str(result.get("request_id", rid)),
        task_query=normalized_task,
        generation_mode=final_mode,
        intent_label=str(result.get("intent_label", "fallback")),
        final_status=str(result.get("final_status", "failed")),
        risk_report=risk_report,
        route_history=result.get("route_history", []),
        recommended_mode=str(result.get("recommended_mode", final_mode)),
        retrieval_context_len=int(result.get("retrieval_context_len", 0) or 0),
        project_root=PROJECT_ROOT,
    )
    replay_report = build_badcase_replay_report(project_root=PROJECT_ROOT, window_days=30)
    tuning = auto_tune_rule_templates_from_replay(replay_report, project_root=PROJECT_ROOT)
    replay_report["rule_tuning"] = tuning
    result["badcase_replay"] = replay_report
    return result
