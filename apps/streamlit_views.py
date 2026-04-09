from __future__ import annotations

import hashlib
import json
import os
import tempfile
import traceback
from datetime import datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from uuid import uuid4

import streamlit as st

from apps import testcase_platform as tp
from asset_loader import process_image_to_text_with_meta
from src.rag.analysis.badcase_loop import (
    build_badcase_replay_report,
    list_badcase_rule_template_history,
    prune_badcase_events,
    rollback_badcase_rule_templates,
)
from workflow_graph import run_testcase_workflow


NAV_OPTIONS = ["资产看板", "知识库管理", "用例生成舱", "资产审核"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

STATUS_LABELS = {
    "pending": "待审核",
    "approved": "已入库",
    "rejected": "已驳回",
    "draft": "草稿",
    "unknown": "未知",
}
SOURCE_TYPE_LABELS = {
    "requirement": "需求",
    "testcase": "测试用例",
    "ui": "UI交互",
    "api_doc": "API接口",
    "unknown": "未知",
}
ORIGIN_LABELS = {
    "file": "上传文件",
    "manual_text": "手工录入",
    "feishu_doc": "飞书文档",
    "feishu_board": "飞书画板",
    "figma": "Figma",
    "api_doc_link": "API文档链接",
    "llm_generated": "模型生成",
    "unknown": "未知",
}
GENERATION_MODE_LABELS = {
    "business_api": "业务接口用例",
    "field_validation": "字段校验用例",
}
APPEND_STRATEGY_LABELS = {
    "review_queue": "加入待审核队列",
    "none": "仅生成不入库",
    "direct_append": "直接入库",
}
SYNC_MODE_LABELS = {
    "append": "增量追加",
    "replace_by_source": "按来源替换",
    "rebuild_all": "全量重建",
}
REVIEW_DECISION_LABELS = {
    "pass": "通过",
    "fail": "未通过",
    "unknown": "未知",
}
WORKFLOW_STATUS_LABELS = {
    "success": "成功",
    "success_with_warning": "成功但需关注",
    "failed": "失败",
    "unknown": "未知",
}


def _display_enum(value: Any, mapping: Dict[str, str], default: str = "未知") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    lowered = text.lower()
    if lowered in mapping:
        return mapping[lowered]
    if text in mapping:
        return mapping[text]
    return text


def _display_status(value: Any) -> str:
    return _display_enum(value, STATUS_LABELS)


def _display_source_type(value: Any) -> str:
    return _display_enum(value, SOURCE_TYPE_LABELS)


def _display_origin(value: Any) -> str:
    return _display_enum(value, ORIGIN_LABELS)


def _display_generation_mode(value: Any) -> str:
    return _display_enum(value, GENERATION_MODE_LABELS, default="未指定")


def _display_append_strategy(value: Any) -> str:
    return _display_enum(value, APPEND_STRATEGY_LABELS, default="未指定")


def _display_sync_mode(value: Any) -> str:
    return _display_enum(value, SYNC_MODE_LABELS, default="未指定")


def _display_review_decision(value: Any) -> str:
    return _display_enum(value, REVIEW_DECISION_LABELS, default="-")


def _display_workflow_status(value: Any) -> str:
    return _display_enum(value, WORKFLOW_STATUS_LABELS, default="-")


def _build_label_maps(
    values: List[str], label_fn: Callable[[str], str]
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    value_to_label: Dict[str, str] = {}
    label_to_value: Dict[str, str] = {}
    ordered_labels: List[str] = []
    for value in values:
        label = label_fn(value)
        if label in label_to_value and label_to_value[label] != value:
            suffix = value or "空"
            label = f"{label}（{suffix}）"
        value_to_label[value] = label
        label_to_value[label] = value
        ordered_labels.append(label)
    return value_to_label, label_to_value, ordered_labels


def _inject_ui_theme() -> None:
    # 保留函数入口，当前不注入任何自定义样式，使用 Streamlit 原生视觉。
    return


def _render_page_header(title: str, description: str) -> None:
    st.subheader(title)
    st.caption(description)
    st.divider()


def _to_dataframe_like(rows: List[Dict[str, Any]], columns: List[str]) -> Any:
    try:
        import pandas as pd  # type: ignore

        if rows:
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame(columns=columns)
    except Exception:
        if rows:
            return [{col: row.get(col, "") for col in columns} for row in rows]
        return {col: [] for col in columns}


def _to_chart_frame(rows: List[Dict[str, Any]], columns: List[str], index_col: str) -> Any:
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows, columns=columns)
        if index_col in df.columns:
            return df.set_index(index_col)
        return df
    except Exception:
        return _to_dataframe_like(rows, columns)


def _split_lines(raw: str) -> List[str]:
    return tp._split_lines(raw)


def _is_image_file_name(file_name: str) -> bool:
    suffix = Path(file_name).suffix.lower()
    return suffix in IMAGE_SUFFIXES


def _render_engine_label(engine: str) -> str:
    if engine.startswith("model:"):
        parts = engine.split(":", 2)
        if len(parts) == 3:
            return f"视觉模型识别（{parts[1]} / {parts[2]}）"
        return "视觉模型识别"
    if engine == "local_ocr":
        return "本地 OCR 识别"
    return engine or "未知"


def _process_uploaded_image_to_markdown(uploaded_file: Any) -> Dict[str, str]:
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower() or ".png"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = os.path.abspath(tmp.name)
        return process_image_to_text_with_meta(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _persist_review_queue_or_warn() -> bool:
    return tp._persist_review_queue_or_warn()


def _is_append_effective_success(summary: Dict[str, Any]) -> bool:
    return tp._is_append_effective_success(summary)


def _queue_row_label(item: Dict[str, Any], status_text_map: Dict[str, str]) -> str:
    item_id = str(item.get("id", ""))[-8:]
    status = status_text_map.get(str(item.get("status", "pending")), "待审核")
    created_at = str(item.get("created_at", "-"))
    mode = _display_generation_mode(item.get("generation_mode", "-"))
    return f"{status} | {created_at} | {mode} | {item_id}"


def _normalize_review_scores(scores_raw: Any) -> Dict[str, int]:
    raw = scores_raw if isinstance(scores_raw, dict) else {}
    business_coverage = int(raw.get("business_coverage", 0) or 0)
    exception_coverage = int(raw.get("exception_coverage", raw.get("exception_flow", 0)) or 0)
    assertion_specificity = int(
        raw.get("assertion_specificity", raw.get("assertion_clarity", 0)) or 0
    )
    executability = int(raw.get("executability", 0) or 0)
    traceability = int(raw.get("traceability", 0) or 0)
    redundancy_control = int(raw.get("redundancy_control", 0) or 0)
    total = int(
        raw.get(
            "total",
            business_coverage
            + exception_coverage
            + assertion_specificity
            + executability
            + traceability
            + redundancy_control,
        )
        or 0
    )
    return {
        "business_coverage": business_coverage,
        "exception_coverage": exception_coverage,
        "assertion_specificity": assertion_specificity,
        "executability": executability,
        "traceability": traceability,
        "redundancy_control": redundancy_control,
        "total": total,
    }


def _render_review_result_detail(review_result: Dict[str, Any]) -> None:
    if not isinstance(review_result, dict) or not review_result:
        st.caption("自动评审详情: 暂无（可能为历史记录或外部导入记录）。")
        return

    decision = str(review_result.get("decision", "-")).strip().lower()
    decision_label = "通过" if decision == "pass" else ("不通过" if decision == "fail" else "-")
    hard_fail_reasons = review_result.get("hard_fail_reasons", [])
    if not isinstance(hard_fail_reasons, list):
        hard_fail_reasons = [str(hard_fail_reasons)]
    hard_fail_reasons = [str(x).strip() for x in hard_fail_reasons if str(x).strip()]

    scores = _normalize_review_scores(review_result.get("scores", {}))
    st.caption(f"自动评审结论: {decision_label} | 总分: {scores.get('total', 0)}/30")
    c1, c2, c3 = st.columns(3, gap="small")
    c1.metric("业务覆盖", str(scores["business_coverage"]))
    c2.metric("异常覆盖", str(scores["exception_coverage"]))
    c3.metric("断言明确", str(scores["assertion_specificity"]))
    c4, c5, c6 = st.columns(3, gap="small")
    c4.metric("可执行性", str(scores["executability"]))
    c5.metric("可追溯性", str(scores["traceability"]))
    c6.metric("冗余控制", str(scores["redundancy_control"]))

    if hard_fail_reasons:
        st.warning("硬失败原因: " + " | ".join(hard_fail_reasons[:4]))

    comments = review_result.get("comments", [])
    if not isinstance(comments, list):
        comments = [str(comments)]
    comments = [str(x).strip() for x in comments if str(x).strip()]
    if comments:
        st.caption("评审意见")
        for line in comments[:6]:
            st.write(f"- {line}")

    rewrite = review_result.get("rewrite_instructions", [])
    if not isinstance(rewrite, list):
        rewrite = [str(rewrite)]
    rewrite = [str(x).strip() for x in rewrite if str(x).strip()]
    if rewrite:
        st.caption("改写建议")
        for line in rewrite[:6]:
            st.write(f"- {line}")


def _render_gap_hints(gap_hints: Dict[str, Any]) -> None:
    if not isinstance(gap_hints, dict) or not gap_hints:
        return
    summary = str(gap_hints.get("gap_summary", "")).strip()
    missing_inputs = gap_hints.get("missing_inputs", [])
    coverage_risks = gap_hints.get("coverage_risks", [])
    suggested_prompts = gap_hints.get("suggested_prompts", [])

    if not isinstance(missing_inputs, list):
        missing_inputs = [missing_inputs]
    if not isinstance(coverage_risks, list):
        coverage_risks = [coverage_risks]
    if not isinstance(suggested_prompts, list):
        suggested_prompts = [suggested_prompts]

    missing_inputs = [str(x).strip() for x in missing_inputs if str(x).strip()]
    coverage_risks = [str(x).strip() for x in coverage_risks if str(x).strip()]
    suggested_prompts = [str(x).strip() for x in suggested_prompts if str(x).strip()]

    if not (summary or missing_inputs or coverage_risks or suggested_prompts):
        return

    st.markdown("**测试设计缺口提示**")
    if summary:
        st.write(f"摘要：{summary}")
    if missing_inputs:
        st.caption("缺少信息")
        for line in missing_inputs[:6]:
            st.write(f"- {line}")
    if coverage_risks:
        st.caption("覆盖风险")
        for line in coverage_risks[:6]:
            st.write(f"- {line}")
    if suggested_prompts:
        st.caption("建议补充信息")
        for line in suggested_prompts[:6]:
            st.write(f"- {line}")


def _render_compliance_report(report: Dict[str, Any]) -> None:
    if not isinstance(report, dict) or not report:
        return
    passed = bool(report.get("pass", True))
    score = report.get("score", 0)
    try:
        score = float(score)
    except Exception:
        score = 0.0
    reasons = report.get("reasons", [])
    missing_items = report.get("missing_items", {})
    hit_items = report.get("hit_items", {})
    rewrite_instructions = report.get("rewrite_instructions", [])
    constraints = report.get("constraints", [])
    rule_engine = report.get("rule_engine", {})
    llm_review = report.get("llm_review", {})
    if not isinstance(constraints, list):
        constraints = []
    if not isinstance(rule_engine, dict):
        rule_engine = {}
    if not isinstance(llm_review, dict):
        llm_review = {}

    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(x).strip() for x in reasons if str(x).strip()]
    if not isinstance(rewrite_instructions, list):
        rewrite_instructions = [str(rewrite_instructions)]
    rewrite_instructions = [str(x).strip() for x in rewrite_instructions if str(x).strip()]
    if not isinstance(missing_items, dict):
        missing_items = {}
    if not isinstance(hit_items, dict):
        hit_items = {}

    st.markdown("**前置协作约束合规报告**")
    if passed:
        st.success(f"约束校验通过（{score:.1f}/100）")
    else:
        st.warning(f"约束校验未通过（{score:.1f}/100）")

    rule_pass = bool(rule_engine.get("pass", False))
    rule_score = float(rule_engine.get("score", 0.0) or 0.0)
    llm_used = bool(llm_review.get("used", False))
    llm_pass = bool(llm_review.get("pass", False)) if llm_used else None
    llm_score = float(llm_review.get("score", 0.0) or 0.0) if llm_used else 0.0
    st.caption(
        "双通道结果: "
        f"规则引擎={'通过' if rule_pass else '未通过'}({rule_score:.1f}) | "
        + (
            f"LLM审查={'通过' if llm_pass else '未通过'}({llm_score:.1f})"
            if llm_used
            else "LLM审查=未启用"
        )
    )
    if reasons:
        st.caption("不通过原因")
        for item in reasons[:6]:
            st.write(f"- {item}")
    for key, label in [
        ("must_cover", "未命中必须覆盖项"),
        ("risk_tags", "未覆盖风险标签"),
        ("must_not_cover_hits", "命中排除项"),
    ]:
        values = missing_items.get(key, [])
        if not isinstance(values, list):
            values = [str(values)]
        values = [str(x).strip() for x in values if str(x).strip()]
        if values:
            st.caption(label)
            for item in values[:6]:
                st.write(f"- {item}")
    hit_risks = hit_items.get("risk_tags", [])
    if not isinstance(hit_risks, list):
        hit_risks = [str(hit_risks)]
    hit_risks = [str(x).strip() for x in hit_risks if str(x).strip()]
    if hit_risks:
        st.caption("已覆盖风险标签")
        for item in hit_risks[:6]:
            st.write(f"- {item}")
    if rewrite_instructions:
        st.caption("改写建议")
        for item in rewrite_instructions[:6]:
            st.write(f"- {item}")
    if constraints:
        with st.expander("查看逐条约束评分", expanded=False):
            for item in constraints[:20]:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("id", "-"))
                category = str(item.get("category", "-"))
                term = str(item.get("term", "-"))
                passed_item = bool(item.get("passed", False))
                cscore = float(item.get("score", 0.0) or 0.0)
                threshold = float(item.get("threshold", 0.0) or 0.0)
                matched = item.get("matched_evidence_ids", [])
                if not isinstance(matched, list):
                    matched = [str(matched)]
                matched = [str(x).strip() for x in matched if str(x).strip()]
                st.write(
                    f"- [{cid}] {category} | {'通过' if passed_item else '未通过'} | "
                    f"score={cscore:.2f} threshold={threshold:.2f} | term={term}"
                )
                if matched:
                    st.caption("  evidence: " + ", ".join(matched[:4]))


def _render_risk_report(risk_report: Dict[str, Any]) -> None:
    if not isinstance(risk_report, dict) or not risk_report:
        return
    items = risk_report.get("items", [])
    severity_counts = risk_report.get("severity_counts", {})
    source_summary = risk_report.get("source_summary", {})
    history_adjustment = risk_report.get("history_adjustment", {})
    if not isinstance(items, list):
        items = []
    if not isinstance(severity_counts, dict):
        severity_counts = {}
    if not isinstance(source_summary, dict):
        source_summary = {}
    if not isinstance(history_adjustment, dict):
        history_adjustment = {}
    if not items:
        return

    level = str(risk_report.get("overall_level", "low")).strip().lower()
    score = float(risk_report.get("overall_score", 0.0) or 0.0)
    level_label = {"high": "高", "medium": "中", "low": "低"}.get(level, level)
    p0 = int(severity_counts.get("P0", 0) or 0)
    p1 = int(severity_counts.get("P1", 0) or 0)
    p2 = int(severity_counts.get("P2", 0) or 0)

    st.markdown("**风险说明（结构化）**")
    st.caption(f"总体风险: {level_label} ({score:.1f}/100) | P0={p0} P1={p1} P2={p2}")

    link_edges = int(source_summary.get("link_edges", 0) or 0)
    gap_risk_count = int(source_summary.get("gap_risk_count", 0) or 0)
    linked_module_count = int(source_summary.get("linked_module_count", 0) or 0)
    st.caption(
        f"来源信号: gap={gap_risk_count} | linked_module={linked_module_count} | link_edges={link_edges}"
    )
    if bool(history_adjustment.get("enabled", False)):
        base_score = float(history_adjustment.get("base_score", score) or score)
        boost = float(history_adjustment.get("score_boost", 0.0) or 0.0)
        window_size = int(history_adjustment.get("window_size", 0) or 0)
        st.caption(
            f"历史badcase校准: base={base_score:.1f} + boost={boost:.1f} | window={window_size}"
        )

    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id", "-")).strip()
        severity = str(item.get("severity", "P2")).strip().upper()
        title = str(item.get("title", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not title:
            continue
        st.write(f"- [{severity}] {title} ({rid})")
        if reason:
            st.caption(f"  原因: {reason}")

    with st.expander("查看风险明细", expanded=False):
        for item in items[:12]:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("id", "-")).strip()
            severity = str(item.get("severity", "P2")).strip().upper()
            category = str(item.get("category", "-")).strip()
            title = str(item.get("title", "-")).strip()
            reason = str(item.get("reason", "")).strip()
            evidence = item.get("evidence", [])
            suggestions = item.get("suggestions", [])
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            if not isinstance(suggestions, list):
                suggestions = [str(suggestions)]
            evidence = [str(x).strip() for x in evidence if str(x).strip()]
            suggestions = [str(x).strip() for x in suggestions if str(x).strip()]

            st.write(f"- [{rid}] {severity} | {category} | {title}")
            if reason:
                st.caption("  原因: " + reason)
            if evidence:
                st.caption("  证据: " + " ; ".join(evidence[:4]))
            if suggestions:
                st.caption("  建议: " + " ; ".join(suggestions[:3]))


def _render_badcase_replay_report(
    replay_report: Dict[str, Any],
    *,
    key_prefix: str,
    enable_window_actions: bool = False,
) -> None:
    if not isinstance(replay_report, dict) or not replay_report:
        return
    event_count = int(replay_report.get("event_count", 0) or 0)
    badcase_count = int(replay_report.get("badcase_count", 0) or 0)
    bad_rate = float(replay_report.get("overall_bad_rate", 0.0) or 0.0)
    rule_template = replay_report.get("rule_template", {})
    alerts = replay_report.get("alerts", [])
    hints = replay_report.get("rule_update_hints", [])
    rule_tuning = replay_report.get("rule_tuning", {})
    if not isinstance(rule_template, dict):
        rule_template = {}
    if not isinstance(alerts, list):
        alerts = []
    if not isinstance(hints, list):
        hints = []
    if not isinstance(rule_tuning, dict):
        rule_tuning = {}

    st.markdown("**Badcase 闭环评测**")
    st.caption(f"窗口样本: {event_count} | badcase: {badcase_count} | bad_rate={bad_rate:.2%}")
    template_ver = str(rule_template.get("version", "v1")).strip() or "v1"
    template_ts = str(rule_template.get("updated_at", "")).strip()
    history_count = int(rule_template.get("history_count", 0) or 0)
    st.caption(f"规则模板: version={template_ver} | history={history_count} | updated_at={template_ts or '-'}")
    if alerts:
        st.warning(f"高风险签名告警: {len(alerts)} 条")
        for item in alerts[:4]:
            if not isinstance(item, dict):
                continue
            signature = str(item.get("signature", "-")).strip()
            sample = int(item.get("sample", 0) or 0)
            item_bad_rate = float(item.get("bad_rate", 0.0) or 0.0)
            st.write(f"- {signature} | sample={sample} | bad_rate={item_bad_rate:.2%}")
    if hints:
        st.caption("规则更新建议")
        for item in hints[:4]:
            if not isinstance(item, dict):
                continue
            template = str(item.get("template", "-")).strip()
            tag = str(item.get("reason_tag", "-")).strip()
            suggestion = str(item.get("suggestion", "")).strip()
            st.write(f"- [{template}] tag={tag} | {suggestion}")
    if bool(rule_tuning.get("applied", False)):
        changes = rule_tuning.get("changes", [])
        if not isinstance(changes, list):
            changes = []
        st.success("规则模板已自动微调")
        for change in changes[:3]:
            if not isinstance(change, dict):
                continue
            field = str(change.get("field", "-")).strip()
            old = change.get("old")
            new = change.get("new")
            st.caption(f"  {field}: {old} -> {new}")

    if enable_window_actions:
        c1, c2, c3 = st.columns([2, 1, 1], gap="small")
        with c1:
            keep_days = st.selectbox(
                "Badcase窗口保留天数",
                options=[30, 60, 90, 180],
                index=2,
                key=f"{key_prefix}_badcase_keep_days",
            )
        with c2:
            if st.button("清理历史窗口", key=f"{key_prefix}_badcase_prune"):
                summary = prune_badcase_events(keep_days=int(keep_days))
                st.success(
                    f"清理完成: before={int(summary.get('before', 0))} "
                    f"after={int(summary.get('after', 0))} "
                    f"removed={int(summary.get('removed', 0))}"
                )
                st.rerun()
        with c3:
            if st.button("回滚规则模板", key=f"{key_prefix}_rule_rollback"):
                summary = rollback_badcase_rule_templates()
                if bool(summary.get("applied", False)):
                    st.success(
                        "回滚完成: "
                        f"to_version={str(summary.get('to_version', '-'))} | "
                        f"history={int(summary.get('history_count', 0) or 0)}"
                    )
                else:
                    st.warning("回滚未执行: " + str(summary.get("reason", "unknown")))
                st.rerun()
        history_rows = list_badcase_rule_template_history(limit=3)
        if history_rows:
            st.caption("最近模板历史")
            for row in history_rows[:3]:
                if not isinstance(row, dict):
                    continue
                st.write(
                    "- "
                    + f"version={str(row.get('version', 'v1'))} | "
                    + f"ts={str(row.get('ts', '-')) or '-'}"
                )


def _render_impact_analysis(impact_analysis: Dict[str, Any]) -> None:
    if not isinstance(impact_analysis, dict) or not impact_analysis:
        return
    if str(impact_analysis.get("version", "")).strip() != "v2":
        return

    current_modules = impact_analysis.get("current_involved_modules", [])
    linked_modules = impact_analysis.get("potential_linked_modules", [])
    anchors = impact_analysis.get("evidence_anchors", [])

    if not isinstance(current_modules, list):
        current_modules = []
    if not isinstance(linked_modules, list):
        linked_modules = []
    if not isinstance(anchors, list):
        anchors = []
    if not current_modules and not linked_modules:
        return

    summary = str(impact_analysis.get("impact_summary", "")).strip()

    st.markdown("**潜在影响模块提示**")
    if summary:
        st.write(f"摘要：{summary}")

    if current_modules:
        st.caption("当前需求涉及模块")
        for item in current_modules[:2]:
            if not isinstance(item, dict):
                continue
            module = str(item.get("module", "-")).strip()
            confidence = item.get("confidence")
            evidence = str(item.get("top_evidence", "")).strip()
            label = f"- {module}"
            if isinstance(confidence, (int, float)):
                label += f" ({float(confidence):.2f})"
            if evidence:
                label += f" | {evidence}"
            st.write(label)

    if linked_modules:
        st.caption("潜在联动模块")
        for item in linked_modules[:2]:
            if not isinstance(item, dict):
                continue
            module = str(item.get("module", "-")).strip()
            confidence = item.get("confidence")
            confidence_level = str(item.get("confidence_level", "")).strip()
            impact_type = str(item.get("impact_type", "")).strip()
            evidence = str(item.get("top_evidence", "")).strip()
            triggers = item.get("trigger_modules", [])
            if not isinstance(triggers, list):
                triggers = [str(triggers)]
            triggers = [str(x).strip() for x in triggers if str(x).strip()]
            label = f"- {module}"
            tags: List[str] = []
            if confidence_level:
                tags.append(confidence_level)
            if isinstance(confidence, (int, float)):
                tags.append(f"{float(confidence):.2f}")
            if impact_type:
                tags.append(impact_type)
            if tags:
                label += " (" + "/".join(tags) + ")"
            if triggers:
                label += " | 触发:" + "、".join(triggers)
            if evidence:
                label += f" | {evidence}"
            st.write(label)

    evidence_items: List[Dict[str, Any]] = []
    for group in (current_modules, linked_modules):
        for item in group:
            if not isinstance(item, dict):
                continue
            anchor = item.get("evidence_anchor")
            if isinstance(anchor, dict):
                evidence_items.append(anchor)
    if not evidence_items:
        evidence_items = [a for a in anchors if isinstance(a, dict)]

    if evidence_items:
        with st.expander("查看证据锚点", expanded=False):
            for item in evidence_items[:6]:
                if not isinstance(item, dict):
                    continue
                source_type = str(item.get("source_type", "unknown")).strip() or "unknown"
                source_name = str(item.get("source_name", "unknown")).strip() or "unknown"
                origin = str(item.get("origin", "unknown")).strip() or "unknown"
                chunk_index = item.get("chunk_index")
                doc_key = str(item.get("doc_key", "")).strip()
                modules = item.get("module_tags", [])
                if not isinstance(modules, list):
                    modules = [str(modules)]
                modules = [str(x).strip() for x in modules if str(x).strip()]
                feature_key = str(item.get("feature_key", "")).strip()
                trace_refs = item.get("trace_refs", [])
                if not isinstance(trace_refs, list):
                    trace_refs = [str(trace_refs)]
                trace_refs = [str(x).strip() for x in trace_refs if str(x).strip()]
                source_confidence = item.get("source_confidence")
                extra = []
                if doc_key:
                    extra.append(f"doc_key={doc_key}")
                if isinstance(chunk_index, int):
                    extra.append(f"片段{chunk_index}")
                if modules:
                    extra.append("modules=" + ",".join(modules))
                if feature_key:
                    extra.append(f"feature={feature_key}")
                if trace_refs:
                    extra.append("trace=" + ",".join(trace_refs[:3]))
                if isinstance(source_confidence, (int, float)):
                    extra.append(f"source_conf={float(source_confidence):.2f}")
                source_bucket = _anchor_source_bucket(item)
                extra.append(f"source_tag={source_bucket}")
                extra_text = f" | {' | '.join(extra)}" if extra else ""
                st.write(f"- [{source_type}] {source_name} ({origin}){extra_text}")


def _anchor_source_bucket(anchor: Dict[str, Any]) -> str:
    source_name = str(anchor.get("source_name", "")).strip().lower()
    origin = str(anchor.get("origin", "")).strip().lower()
    doc_key = str(anchor.get("doc_key", "")).strip().lower()
    snippet = str(anchor.get("snippet", "")).strip().lower()
    merged = " ".join([source_name, origin, doc_key, snippet])
    if any(token in merged for token in ("erp", "third", "三方", "external", "供应商", "外部")):
        return "third_party_spec"
    return "internal_spec"


def _render_link_analysis(link_edges: Any, trace_refs: Any, link_summary: Any) -> None:
    if not isinstance(link_edges, list):
        link_edges = []
    if not isinstance(trace_refs, dict):
        trace_refs = {}
    if not isinstance(link_summary, dict):
        link_summary = {}
    if not link_edges and not trace_refs:
        return

    total_edges = int(link_summary.get("total_edges", len(link_edges)) or len(link_edges))
    doc_doc_edges = int(link_summary.get("doc_doc_edges", 0) or 0)
    module_edges = int(link_summary.get("module_edges", 0) or 0)
    relations = link_summary.get("relations", [])
    if not isinstance(relations, list):
        relations = []

    req_ids = trace_refs.get("req_ids", [])
    api_ids = trace_refs.get("api_ids", [])
    tc_ids = trace_refs.get("testcase_ids", [])
    ui_ids = trace_refs.get("ui_ids", [])
    if not isinstance(req_ids, list):
        req_ids = []
    if not isinstance(api_ids, list):
        api_ids = []
    if not isinstance(tc_ids, list):
        tc_ids = []
    if not isinstance(ui_ids, list):
        ui_ids = []

    st.markdown("**跨端双向链路**")
    st.caption(
        f"边数量: {total_edges} | 文档链: {doc_doc_edges} | 模块链: {module_edges} | "
        f"关系: {','.join([str(x) for x in relations[:6]]) if relations else '-'}"
    )
    st.caption(
        f"TraceRefs: req={len(req_ids)} api={len(api_ids)} testcase={len(tc_ids)} ui={len(ui_ids)}"
    )
    with st.expander("查看链路明细", expanded=False):
        for edge in link_edges[:30]:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("src_id", "")).replace("doc:", "").replace("module:", "")
            dst = str(edge.get("dst_id", "")).replace("doc:", "").replace("module:", "")
            relation = str(edge.get("relation", "")).strip()
            confidence = float(edge.get("confidence", 0.0) or 0.0)
            anchor_id = str(edge.get("evidence_anchor_id", "")).strip()
            if not src or not dst or not relation:
                continue
            line = f"- {src} -> {dst} ({relation}) conf={confidence:.2f}"
            if anchor_id:
                line += f" | anchor={anchor_id}"
            st.write(line)


def _render_generation_support(
    contracts: Dict[str, Any], mapping_rules: Dict[str, Any], coverage_matrix: Dict[str, Any]
) -> None:
    if not isinstance(contracts, dict):
        contracts = {}
    if not isinstance(mapping_rules, dict):
        mapping_rules = {}
    if not isinstance(coverage_matrix, dict):
        coverage_matrix = {}

    internal_contract = contracts.get("internal_contract", {})
    external_contract = contracts.get("external_contract", {})
    if not isinstance(internal_contract, dict):
        internal_contract = {}
    if not isinstance(external_contract, dict):
        external_contract = {}

    mapping_items = mapping_rules.get("mapping_rules", [])
    if not isinstance(mapping_items, list):
        mapping_items = []

    matrix_rows = coverage_matrix.get("coverage_matrix", [])
    if not isinstance(matrix_rows, list):
        matrix_rows = []

    summary = str(contracts.get("contract_summary", "")).strip()
    if not summary and not mapping_items and not matrix_rows:
        return

    st.markdown("**集成契约与映射摘要**")
    if summary:
        st.write(f"摘要：{summary}")

    internal_interfaces = internal_contract.get("interfaces", [])
    external_interfaces = external_contract.get("interfaces", [])
    if not isinstance(internal_interfaces, list):
        internal_interfaces = []
    if not isinstance(external_interfaces, list):
        external_interfaces = []
    st.caption("双域契约")
    st.write(
        "- internal_spec: "
        + (", ".join([str(x) for x in internal_interfaces[:6]]) if internal_interfaces else "-")
    )
    st.write(
        "- third_party_spec: "
        + (", ".join([str(x) for x in external_interfaces[:6]]) if external_interfaces else "-")
    )

    if mapping_items:
        st.caption("字段映射规则（mapping_rule）")
        for item in mapping_items[:8]:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source_field", "-")).strip()
            dst = str(item.get("target_field", "-")).strip()
            transform = str(item.get("transform_rule", "")).strip()
            anchor_id = str(item.get("evidence_anchor_id", "")).strip()
            line = f"- [mapping_rule] {src} -> {dst}"
            if transform:
                line += f" | {transform}"
            if anchor_id:
                line += f" | anchor={anchor_id}"
            st.write(line)

    selected = [
        item for item in matrix_rows if isinstance(item, dict) and bool(item.get("selected"))
    ]
    if selected:
        st.caption("集成覆盖矩阵（已选）")
        for item in selected[:12]:
            domain = str(item.get("domain", "-")).strip()
            action = str(item.get("action", "-")).strip()
            outcome = str(item.get("outcome", "-")).strip()
            priority = str(item.get("priority", "P1")).strip()
            st.write(f"- [{priority}] {domain}-{action}-{outcome}")


def _build_import_preview(file_bytes: bytes) -> Dict[str, Any]:
    payload = json.loads(file_bytes.decode("utf-8"))
    raw_items = payload if isinstance(payload, list) else payload.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("JSON 结构无效，必须是数组或包含 items 数组。")

    normalized_items: List[Dict[str, Any]] = []
    invalid_rows = 0
    missing_id_count = 0
    normalized_status_count = 0
    duplicate_ids: set[str] = set()
    seen_ids: set[str] = set()

    for row in raw_items:
        if not isinstance(row, dict):
            invalid_rows += 1
            continue
        item = dict(row)

        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = f"imported_{uuid4().hex[:10]}"
            missing_id_count += 1

        if item_id in seen_ids:
            duplicate_ids.add(item_id)
        seen_ids.add(item_id)

        item_status = str(item.get("status", "pending")).strip().lower()
        if item_status not in {"pending", "approved", "rejected"}:
            item_status = "pending"
            normalized_status_count += 1

        item["id"] = item_id
        item["status"] = item_status
        item.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        normalized_items.append(item)

    dedup_map: Dict[str, Dict[str, Any]] = {}
    for item in normalized_items:
        dedup_map[str(item.get("id", ""))] = item
    deduped_items = list(dedup_map.values())
    deduped_items.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)

    preview_rows = [
        {
            "记录ID": str(item.get("id", ""))[-8:],
            "状态": _display_status(item.get("status", "pending")),
            "生成时间": str(item.get("created_at", "-")),
            "模块": str(item.get("module_text", "")).strip() or "-",
            "需求摘要": str(item.get("task_query", "")).strip()[:30] or "-",
        }
        for item in deduped_items[:20]
    ]

    return {
        "file_sha": hashlib.sha256(file_bytes).hexdigest(),
        "total_rows": len(raw_items),
        "invalid_rows": invalid_rows,
        "valid_rows": len(normalized_items),
        "missing_id_count": missing_id_count,
        "normalized_status_count": normalized_status_count,
        "duplicate_id_count": len(duplicate_ids),
        "deduped_rows": len(deduped_items),
        "items": deduped_items,
        "preview_rows": preview_rows,
    }


def _parse_datetime_safe(raw_time: Any) -> datetime | None:
    text = str(raw_time or "").strip()
    if not text:
        return None

    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")

    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt
        except Exception:
            pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            pass

    return None


def _split_module_text(raw: Any) -> List[str]:
    modules: List[str] = []
    for token in str(raw or "").replace("，", ",").replace("、", ",").split(","):
        value = token.strip()
        if value and value not in modules:
            modules.append(value)
    return modules


def _extract_row_status(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    status = str(
        metadata.get("status")
        or metadata.get("review_status")
        or metadata.get("ingest_status")
        or "unknown"
    ).strip().lower()
    return status or "unknown"


def _extract_row_modules(row: Dict[str, Any]) -> List[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    modules: List[str] = []
    raw_modules = metadata.get("modules")
    if isinstance(raw_modules, list):
        for item in raw_modules:
            for module in _split_module_text(item):
                if module not in modules:
                    modules.append(module)
    elif raw_modules is not None:
        for module in _split_module_text(raw_modules):
            if module not in modules:
                modules.append(module)

    for key in ("module", "module_name", "business_module"):
        for module in _split_module_text(metadata.get(key)):
            if module not in modules:
                modules.append(module)
    return modules


@st.cache_data(show_spinner=False, ttl=60)
def _load_dashboard_snapshot(cache_buster: int = 0) -> Dict[str, Any]:
    _ = cache_buster
    index_data = tp._load_kb_index_data()
    rows = tp._kb_index_to_rows(index_data)
    logs = tp._load_kb_operation_logs(limit=300)
    return {"index_data": index_data, "rows": rows, "logs": logs}


def _render_nav_menu() -> str:
    st.sidebar.title("导航菜单")
    return st.sidebar.radio("导航菜单", NAV_OPTIONS, index=0, key="nav_menu")


def _render_sidebar_runtime_status(
    model_name: str,
    embedding_ok: bool,
    embedding_detail: str,
    chroma_ok: bool,
    chroma_detail: str,
) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("系统状态")
    st.sidebar.write(f"当前模型: `{model_name}`")
    st.sidebar.markdown(
        f"Embedding 模型状态: {':green[已连接]' if embedding_ok else ':red[异常]'}"
    )
    st.sidebar.caption(embedding_detail)
    st.sidebar.markdown(
        f"Chroma 数据库状态: {':green[已连接]' if chroma_ok else ':red[异常]'}"
    )
    st.sidebar.caption(chroma_detail)


def _handle_kb_sync(sync_request: Dict[str, Any]) -> None:
    if not sync_request.get("sync_clicked"):
        return

    assets: List[Dict[str, Any]] = []
    sync_log_extra = {
        "sync_mode": sync_request.get("sync_mode", ""),
        "ingest_status": sync_request.get("kb_ingest_status", ""),
        "ingest_release": (sync_request.get("kb_ingest_release", "") or "").strip(),
        "ingest_modules": (sync_request.get("kb_ingest_modules", "") or "").strip(),
    }

    try:
        kb_module = tp.load_kb_upsert_module()
        base_asset_metadata = tp._build_base_asset_metadata(
            status=sync_request["kb_ingest_status"],
            module_text=sync_request["kb_ingest_modules"],
            release_text=sync_request["kb_ingest_release"],
            trace_refs_text=sync_request["kb_ingest_trace_refs"],
        )

        assets.extend(
            tp._build_assets_from_uploaded_files(
                kb_module,
                sync_request["req_files"],
                source_type="requirement",
                base_metadata=base_asset_metadata,
            )
        )
        req_text = (sync_request["req_text"] or "").strip()
        if req_text:
            assets.append(
                kb_module.build_asset(
                    source_type="requirement",
                    origin="manual_text",
                    source_name=f"manual_requirement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    text=req_text,
                    metadata=dict(base_asset_metadata),
                )
            )
        assets.extend(
            tp._build_assets_from_multiline_refs(
                kb_module,
                sync_request["feishu_docs"],
                source_type="requirement",
                origin="feishu_doc",
                name_prefix="feishu_doc",
                base_metadata=base_asset_metadata,
            )
        )

        assets.extend(
            tp._build_assets_from_uploaded_files(
                kb_module,
                sync_request["tc_files"],
                source_type="testcase",
                base_metadata=base_asset_metadata,
            )
        )
        tc_text = (sync_request["tc_text"] or "").strip()
        if tc_text:
            assets.append(
                kb_module.build_asset(
                    source_type="testcase",
                    origin="manual_text",
                    source_name=f"manual_testcase_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    text=tc_text,
                    metadata=dict(base_asset_metadata),
                )
            )
        assets.extend(
            tp._build_assets_from_multiline_refs(
                kb_module,
                sync_request["feishu_boards"],
                source_type="testcase",
                origin="feishu_board",
                name_prefix="feishu_board",
                base_metadata=base_asset_metadata,
            )
        )

        assets.extend(
            tp._build_assets_from_uploaded_files(
                kb_module,
                sync_request["ui_files"],
                source_type="ui",
                base_metadata=base_asset_metadata,
            )
        )
        ui_text = (sync_request["ui_text"] or "").strip()
        if ui_text:
            assets.append(
                kb_module.build_asset(
                    source_type="ui",
                    origin="manual_text",
                    source_name=f"manual_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    text=ui_text,
                    metadata=dict(base_asset_metadata),
                )
            )
        assets.extend(
            tp._build_assets_from_multiline_refs(
                kb_module,
                sync_request["figma_refs"],
                source_type="ui",
                origin="figma",
                name_prefix="figma",
                base_metadata=base_asset_metadata,
            )
        )

        assets.extend(
            tp._build_assets_from_uploaded_files(
                kb_module,
                sync_request["api_files"],
                source_type="api_doc",
                base_metadata=base_asset_metadata,
            )
        )
        api_text = (sync_request["api_text"] or "").strip()
        if api_text:
            assets.append(
                kb_module.build_asset(
                    source_type="api_doc",
                    origin="manual_text",
                    source_name=f"manual_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    text=api_text,
                    metadata=dict(base_asset_metadata),
                )
            )
        assets.extend(
            tp._build_assets_from_multiline_refs(
                kb_module,
                sync_request["api_doc_links"],
                source_type="api_doc",
                origin="api_doc_link",
                name_prefix="api_doc_link",
                base_metadata=base_asset_metadata,
            )
        )

        if not assets:
            st.sidebar.warning("未检测到可同步资产，请上传文件或填写文本/链接。")
            tp._append_kb_operation_log(
                operation="sync",
                summary={
                    "ok": False,
                    "partial_success": False,
                    "received_assets": 0,
                    "errors": ["未检测到可同步资产"],
                },
                extra=sync_log_extra,
            )
            return

        with st.spinner("正在同步知识库资产到 Chroma..."):
            summary = kb_module.ingest_assets(assets=assets, mode=sync_request["sync_mode"])

        st.session_state["kb_sync_summary"] = summary
        tp._append_kb_operation_log(
            operation="sync",
            summary=summary,
            extra={**sync_log_extra, "assets_count": len(assets)},
        )

        if summary.get("ok"):
            if summary.get("partial_success"):
                st.sidebar.warning(
                    "同步部分成功："
                    f"入库 {summary.get('ingested_assets', 0)} 份，"
                    f"失败 {len(summary.get('errors', []))} 份。"
                )
            else:
                st.sidebar.success(
                    "同步成功："
                    f"入库 {summary.get('ingested_assets', 0)} 份，"
                    f"新增 {summary.get('added_chunks', 0)} 个切片。"
                )
        else:
            st.sidebar.error(
                "同步失败："
                f"成功 {summary.get('ingested_assets', 0)} 份，"
                f"失败 {len(summary.get('errors', []))} 份。"
            )
    except Exception as exc:
        tp._append_kb_operation_log(
            operation="sync",
            summary={
                "ok": False,
                "partial_success": False,
                "received_assets": len(assets),
                "errors": [str(exc)],
            },
            extra=sync_log_extra,
        )
        st.sidebar.error(f"知识库同步异常: {exc}")


def _render_kb_summary_sidebar() -> None:
    kb_summary = st.session_state.get("kb_sync_summary")
    if not kb_summary:
        return

    st.sidebar.divider()
    st.sidebar.caption(tp._kb_summary_caption(kb_summary))
    if kb_summary.get("warnings"):
        st.sidebar.info(
            "同步告警: "
            + " | ".join(kb_summary["warnings"][:2])
            + (" ..." if len(kb_summary["warnings"]) > 2 else "")
        )
    if kb_summary.get("errors"):
        st.sidebar.error(
            "同步错误: "
            + " | ".join(kb_summary["errors"][:2])
            + (" ..." if len(kb_summary["errors"]) > 2 else "")
        )


def render_asset_dashboard(
    model_name: str,
    embedding_ok: bool,
    embedding_detail: str,
    chroma_ok: bool,
    chroma_detail: str,
) -> None:
    _render_sidebar_runtime_status(
        model_name=model_name,
        embedding_ok=embedding_ok,
        embedding_detail=embedding_detail,
        chroma_ok=chroma_ok,
        chroma_detail=chroma_detail,
    )
    _render_page_header("资产看板", "查看核心指标、趋势、动态与异常，避免明细信息淹没决策信号。")

    _, main_col, _ = st.columns([1.2, 8.8, 1.2], gap="small")
    with main_col:
        snapshot = _load_dashboard_snapshot(int(st.session_state.get("kb_data_version", 0) or 0))
        index_data = snapshot.get("index_data", {}) or {}
        rows = snapshot.get("rows", []) or []
        logs = snapshot.get("logs", []) or []

        enriched_rows: List[Dict[str, Any]] = []
        all_modules: List[str] = []
        all_statuses: List[str] = []
        for row in rows:
            synced_dt = _parse_datetime_safe(row.get("synced_at"))
            status = _extract_row_status(row)
            modules = _extract_row_modules(row)

            if status not in all_statuses:
                all_statuses.append(status)
            for module in modules:
                if module not in all_modules:
                    all_modules.append(module)

            enriched = dict(row)
            enriched["_synced_dt"] = synced_dt
            enriched["_status"] = status
            enriched["_modules"] = modules
            enriched["_module_text"] = ",".join(modules) if modules else "-"
            enriched_rows.append(enriched)

        all_types = sorted({str(r.get("source_type", "unknown")) for r in enriched_rows})
        all_modules = sorted(all_modules)
        all_statuses = sorted(all_statuses or ["unknown"])
        _, type_label_to_value, type_labels = _build_label_maps(all_types, _display_source_type)
        _, status_label_to_value, status_labels = _build_label_maps(all_statuses, _display_status)

        with st.container(border=True):
            st.markdown("**全局筛选**")
            f1, f2, f3, f4 = st.columns([2, 3, 3, 2], gap="small")
            with f1:
                range_label = st.selectbox(
                    "时间范围",
                    options=["最近7天", "最近30天", "最近90天", "全部"],
                    index=1,
                    key="dash_range",
                )
            with f2:
                selected_type_labels = st.multiselect(
                    "资产类型",
                    options=type_labels,
                    default=type_labels,
                    key="dash_types",
                )
            with f3:
                selected_modules = st.multiselect(
                    "模块标签",
                    options=all_modules,
                    default=all_modules,
                    key="dash_modules",
                )
            with f4:
                selected_status_labels = st.multiselect(
                    "入库状态",
                    options=status_labels,
                    default=status_labels,
                    key="dash_statuses",
                )

        range_days_map = {"最近7天": 7, "最近30天": 30, "最近90天": 90, "全部": None}
        now = datetime.now()
        days = range_days_map.get(range_label)
        threshold = now - timedelta(days=days) if isinstance(days, int) else None
        selected_type_values = {
            type_label_to_value[label]
            for label in selected_type_labels
            if label in type_label_to_value
        }
        selected_status_values = {
            status_label_to_value[label]
            for label in selected_status_labels
            if label in status_label_to_value
        }

        filtered_rows: List[Dict[str, Any]] = []
        for row in enriched_rows:
            row_dt = row.get("_synced_dt")
            if threshold and (not isinstance(row_dt, datetime) or row_dt < threshold):
                continue
            if selected_type_values and str(row.get("source_type", "unknown")) not in selected_type_values:
                continue
            if selected_status_values and str(row.get("_status", "unknown")) not in selected_status_values:
                continue
            if selected_modules:
                row_modules = row.get("_modules", []) or []
                if not any(m in selected_modules for m in row_modules):
                    continue
            filtered_rows.append(row)

        filtered_logs: List[Dict[str, Any]] = []
        for item in logs:
            log_dt = _parse_datetime_safe(item.get("timestamp"))
            if threshold and (not isinstance(log_dt, datetime) or log_dt < threshold):
                continue
            payload = dict(item)
            payload["_dt"] = log_dt
            filtered_logs.append(payload)

        updated_at = tp._format_time_display(index_data.get("updated_at", ""))
        total_assets = len(filtered_rows)
        total_chunks = sum(int(r.get("chunk_count", 0) or 0) for r in filtered_rows)
        module_covered = len(
            {m for r in filtered_rows for m in (r.get("_modules", []) or []) if str(m).strip()}
        )
        assets_7d = sum(
            1
            for r in filtered_rows
            if isinstance(r.get("_synced_dt"), datetime)
            and r.get("_synced_dt") >= now - timedelta(days=7)
        )
        pending_review = sum(
            1
            for item in (st.session_state.get("review_queue") or [])
            if str(item.get("status", "pending")) == "pending"
        )
        failed_ops_7d = sum(
            1
            for item in logs
            if not bool(item.get("ok"))
            and isinstance(_parse_datetime_safe(item.get("timestamp")), datetime)
            and _parse_datetime_safe(item.get("timestamp")) >= now - timedelta(days=7)
        )

        with st.container(border=True):
            st.markdown("**核心指标**")
            m1, m2, m3, m4, m5, m6 = st.columns(6, gap="small")
            m1.metric("资产总数", str(total_assets))
            m2.metric("向量切片", str(total_chunks))
            m3.metric("覆盖模块", str(module_covered))
            m4.metric("7天新增资产", str(assets_7d))
            m5.metric("待审核积压", str(pending_review))
            m6.metric("7天失败操作", str(failed_ops_7d))
            st.caption(
                f"当前筛选命中 {len(filtered_rows)} / 全量 {len(enriched_rows)} 资产，"
                f"最近全库更新时间：{updated_at}"
            )

        tab_overview, tab_activity, tab_assets, tab_alerts = st.tabs(
            ["总览", "最近动态", "资产明细", "异常告警"]
        )

        with tab_overview:
            with st.container(border=True):
                st.markdown("**趋势与分布**")
                if not filtered_rows:
                    st.info("当前筛选下暂无资产，无法绘制趋势图。")
                else:
                    daily_assets: Dict[str, int] = {}
                    type_dist: Dict[str, int] = {}
                    module_dist: Dict[str, int] = {}
                    for row in filtered_rows:
                        dt = row.get("_synced_dt")
                        if isinstance(dt, datetime):
                            key = dt.strftime("%Y-%m-%d")
                            daily_assets[key] = daily_assets.get(key, 0) + 1
                        source_type = str(row.get("source_type", "unknown"))
                        source_label = _display_source_type(source_type)
                        type_dist[source_label] = type_dist.get(source_label, 0) + 1
                        for module in row.get("_modules", []) or []:
                            module_dist[module] = module_dist.get(module, 0) + 1

                    c_left, c_right = st.columns([1, 1], gap="large")
                    with c_left:
                        st.caption("新增资产趋势（按天）")
                        daily_rows = [
                            {"日期": k, "新增资产数": v}
                            for k, v in sorted(daily_assets.items(), key=lambda x: x[0])
                        ]
                        st.line_chart(
                            _to_chart_frame(daily_rows, ["日期", "新增资产数"], "日期"),
                            height=240,
                        )
                        st.caption("资产类型分布")
                        type_rows = [
                            {"资产类型": k, "资产数": v}
                            for k, v in sorted(type_dist.items(), key=lambda x: x[1], reverse=True)
                        ]
                        st.bar_chart(
                            _to_chart_frame(type_rows, ["资产类型", "资产数"], "资产类型"),
                            height=220,
                        )

                    with c_right:
                        st.caption("模块覆盖 Top10")
                        module_rows = [
                            {"模块": k, "资产数": v}
                            for k, v in sorted(module_dist.items(), key=lambda x: x[1], reverse=True)[:10]
                        ]
                        st.bar_chart(
                            _to_chart_frame(module_rows, ["模块", "资产数"], "模块"),
                            height=240,
                        )
                        top_asset_rows = [
                            {
                                "资产名称": str(r.get("source_name", "-"))[:60],
                                "类型": _display_source_type(r.get("source_type", "-")),
                                "切片数": int(r.get("chunk_count", 0) or 0),
                                "同步时间": tp._format_time_display(r.get("synced_at", "")),
                            }
                            for r in sorted(
                                filtered_rows,
                                key=lambda x: int(x.get("chunk_count", 0) or 0),
                                reverse=True,
                            )[:10]
                        ]
                        st.caption("高切片资产 Top10")
                        st.dataframe(
                            _to_dataframe_like(top_asset_rows, ["资产名称", "类型", "切片数", "同步时间"]),
                            use_container_width=True,
                            hide_index=True,
                        )

        with tab_activity:
            with st.container(border=True):
                st.markdown("**最近动态**")
                limit = st.selectbox(
                    "显示条数",
                    options=[20, 50, 100, 200],
                    index=1,
                    key="dash_logs_limit",
                )
                activity_rows: List[Dict[str, Any]] = []
                for item in filtered_logs[: int(limit)]:
                    extra = item.get("extra", {}) or {}
                    summary = item.get("summary", {}) or {}
                    asset_name = (
                        extra.get("source_name")
                        or extra.get("ingest_release")
                        or summary.get("source_name")
                        or "-"
                    )
                    activity_rows.append(
                        {
                            "时间": tp._format_time_display(item.get("timestamp", "")),
                            "动作": tp._op_type_label(str(item.get("operation", ""))),
                            "状态": tp._op_status_label(item),
                            "资产名称": str(asset_name),
                            "备注": str((summary.get("errors") or ""))[:80] if not item.get("ok") else "-",
                        }
                    )
                st.dataframe(
                    _to_dataframe_like(activity_rows, ["时间", "动作", "状态", "资产名称", "备注"]),
                    use_container_width=True,
                    hide_index=True,
                )
                if not activity_rows:
                    st.info("当前筛选条件下暂无动态记录。")

        with tab_assets:
            with st.container(border=True):
                st.markdown("**资产明细**")
                keyword = st.text_input(
                    "关键字（资产名 / 文档主键 / 模块）",
                    key="dash_assets_keyword",
                    placeholder="例如: 退票、figma、REQ-123",
                ).strip().lower()
                page_size = st.selectbox(
                    "每页条数",
                    options=[20, 50, 100],
                    index=0,
                    key="dash_assets_page_size",
                )
                filtered_detail_rows: List[Dict[str, Any]] = []
                for row in filtered_rows:
                    if keyword:
                        haystack = " ".join(
                            [
                                str(row.get("source_name", "")),
                                str(row.get("doc_key", "")),
                                str(row.get("_module_text", "")),
                            ]
                        ).lower()
                        if keyword not in haystack:
                            continue
                    filtered_detail_rows.append(row)

                total_detail = len(filtered_detail_rows)
                detail_page_count = max(1, ceil(total_detail / int(page_size)))
                detail_page_key = "dash_assets_page_no"
                try:
                    current_page = int(st.session_state.get(detail_page_key, 1))
                except Exception:
                    current_page = 1
                st.session_state[detail_page_key] = min(max(1, current_page), detail_page_count)
                detail_page_no = int(
                    st.number_input(
                        "页码",
                        min_value=1,
                        max_value=detail_page_count,
                        step=1,
                        key=detail_page_key,
                    )
                )
                start = (detail_page_no - 1) * int(page_size)
                end = start + int(page_size)
                page_rows = filtered_detail_rows[start:end]
                st.caption(
                    f"明细共 {total_detail} 条 | 第 {detail_page_no}/{detail_page_count} 页"
                )

                table_rows = [
                    {
                        "资产名称": str(r.get("source_name", "-"))[:60],
                        "模块标签": str(r.get("_module_text", "-")),
                        "资产类型": _display_source_type(r.get("source_type", "-")),
                        "接入来源": _display_origin(r.get("origin", "-")),
                        "切片数": int(r.get("chunk_count", 0) or 0),
                        "状态": _display_status(r.get("_status", "unknown")),
                        "同步时间": tp._format_time_display(r.get("synced_at", "")),
                    }
                    for r in page_rows
                ]
                st.dataframe(
                    _to_dataframe_like(
                        table_rows,
                        ["资产名称", "模块标签", "资产类型", "接入来源", "切片数", "状态", "同步时间"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                if not table_rows:
                    st.info("当前筛选条件下无资产明细。")

                if page_rows:
                    option_map = {
                        f"{str(r.get('source_name', '-'))[:50]} | {str(r.get('doc_key', '-'))[:28]}": str(
                            r.get("doc_key", "")
                        )
                        for r in page_rows
                    }
                    selected_label = st.selectbox(
                        "查看单条详情",
                        options=list(option_map.keys()),
                        key="dash_asset_detail_doc_key",
                    )
                    selected_doc_key = option_map.get(selected_label, "")
                    selected = next(
                        (r for r in page_rows if str(r.get("doc_key", "")) == selected_doc_key),
                        None,
                    )
                    if selected:
                        st.caption("资产元数据")
                        st.json(selected.get("metadata", {}))

        with tab_alerts:
            with st.container(border=True):
                st.markdown("**异常告警**")
                draft_assets = [r for r in filtered_rows if str(r.get("_status")) == "draft"]
                unknown_status_assets = [r for r in filtered_rows if str(r.get("_status")) == "unknown"]
                low_chunk_assets = [
                    r for r in filtered_rows if int(r.get("chunk_count", 0) or 0) <= 1
                ]
                missing_module_assets = [
                    r for r in filtered_rows if not (r.get("_modules") or [])
                ]
                failed_logs = [l for l in filtered_logs if not bool(l.get("ok"))]
                partial_logs = [
                    l for l in filtered_logs if bool(l.get("ok")) and bool(l.get("partial_success"))
                ]

                a1, a2, a3, a4 = st.columns(4, gap="small")
                a1.metric("失败操作", str(len(failed_logs)))
                a2.metric("部分成功", str(len(partial_logs)))
                a3.metric("草稿资产", str(len(draft_assets)))
                a4.metric("缺模块资产", str(len(missing_module_assets)))

                badcase_replay = build_badcase_replay_report(window_days=30)
                _render_badcase_replay_report(
                    badcase_replay if isinstance(badcase_replay, dict) else {},
                    key_prefix="dashboard",
                    enable_window_actions=True,
                )

                alert_rows: List[Dict[str, Any]] = []
                for row in failed_logs[:20]:
                    alert_rows.append(
                        {
                            "告警类型": "操作失败",
                            "对象": tp._op_type_label(str(row.get("operation", ""))),
                            "时间": tp._format_time_display(row.get("timestamp", "")),
                            "详情": "请检查日志详情",
                        }
                    )
                for row in draft_assets[:20]:
                    alert_rows.append(
                        {
                            "告警类型": "草稿资产",
                            "对象": str(row.get("source_name", "-"))[:60],
                            "时间": tp._format_time_display(row.get("synced_at", "")),
                            "详情": "状态=草稿，默认不会参与检索",
                        }
                    )
                for row in unknown_status_assets[:20]:
                    alert_rows.append(
                        {
                            "告警类型": "状态缺失",
                            "对象": str(row.get("source_name", "-"))[:60],
                            "时间": tp._format_time_display(row.get("synced_at", "")),
                            "详情": "建议补齐入库状态元数据",
                        }
                    )
                for row in low_chunk_assets[:20]:
                    alert_rows.append(
                        {
                            "告警类型": "低切片资产",
                            "对象": str(row.get("source_name", "-"))[:60],
                            "时间": tp._format_time_display(row.get("synced_at", "")),
                            "详情": "chunk_count<=1，建议检查解析质量",
                        }
                    )

                st.dataframe(
                    _to_dataframe_like(alert_rows, ["告警类型", "对象", "时间", "详情"]),
                    use_container_width=True,
                    hide_index=True,
                )
                if not alert_rows:
                    st.success("当前筛选下未发现明显异常。")


def render_kb_management(
    model_name: str,
    embedding_ok: bool,
    embedding_detail: str,
    chroma_ok: bool,
    chroma_detail: str,
) -> None:
    _render_sidebar_runtime_status(
        model_name=model_name,
        embedding_ok=embedding_ok,
        embedding_detail=embedding_detail,
        chroma_ok=chroma_ok,
        chroma_detail=chroma_detail,
    )
    _render_page_header("知识库管理", "管理多源资产上传、标签标注与本地向量库入库流程。")
    st.session_state.setdefault("mgmt_image_parse_results", [])
    st.session_state.setdefault("mgmt_image_parse_errors", [])
    st.session_state.setdefault("mgmt_doc_parse_warnings", [])

    mode_options = [
        "1. 增量追加（推荐）: 新来源会入库，已存在同名来源将跳过",
        "2. 按来源替换: 若来源已存在，先删旧数据再重建该来源",
        "3. 全量重建: 清空后仅保留本批次导入内容",
    ]
    mode_map = {
        mode_options[0]: "append",
        mode_options[1]: "replace_by_source",
        mode_options[2]: "rebuild_all",
    }
    status_options = [
        "1. 已入库（可参与检索）",
        "2. 草稿（默认不参与检索）",
    ]
    status_map = {
        status_options[0]: "approved",
        status_options[1]: "draft",
    }

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown("**资产上传与入库配置**")
            mode_label = st.selectbox(
                "同步模式（增量/全量）",
                options=mode_options,
                index=0,
                key="mgmt_sync_mode",
            )
            kb_ingest_status = st.selectbox(
                "本批次入库状态",
                options=status_options,
                index=0,
                key="mgmt_ingest_status",
            )
            kb_ingest_release = st.text_input(
                "本批次版本标识（可选）",
                key="mgmt_ingest_release",
                placeholder="例如: v2026.03, 2026Q1",
            )
            kb_ingest_modules = st.text_input(
                "本批次模块标签（可选，逗号分隔）",
                key="mgmt_ingest_modules",
                placeholder="例如: 订单, 支付, 风控",
            )
            kb_ingest_trace_refs = st.text_area(
                "本批次追踪ID（可选，每行或逗号分隔）",
                key="mgmt_ingest_trace_refs",
                height=70,
                placeholder="例如: REQ-123, API-45, UI-88",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 文档/图片拖拽区")

            st.markdown("**需求资产**")
            req_files = st.file_uploader(
                "需求资产文件（md/txt/csv/doc/docx/pdf/图片）",
                type=["md", "txt", "csv", "doc", "docx", "pdf", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="mgmt_req_files",
            )
            req_text = st.text_area("需求补充文字描述", key="mgmt_req_text", height=70)
            feishu_docs = st.text_area(
                "飞书文档链接/Token（每行一个）",
                key="mgmt_feishu_docs",
                height=70,
                placeholder="https://xxx.feishu.cn/docx/...",
            )

            st.markdown("**测试用例资产**")
            tc_files = st.file_uploader(
                "测试用例资产（md/txt/csv/doc/docx/pdf/图片/xmind）",
                type=["md", "txt", "csv", "doc", "docx", "pdf", "png", "jpg", "jpeg", "xmind"],
                accept_multiple_files=True,
                key="mgmt_tc_files",
            )
            tc_text = st.text_area("测试用例补充描述", key="mgmt_tc_text", height=70)
            feishu_boards = st.text_area(
                "飞书画板链接/Token（每行一个）",
                key="mgmt_feishu_boards",
                height=70,
            )

            st.markdown("**UI交互资产**")
            ui_files = st.file_uploader(
                "UI交互资产（txt/md/doc/docx/pdf/图片）",
                type=["txt", "md", "doc", "docx", "pdf", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="mgmt_ui_files",
            )
            ui_text = st.text_area("UI交互补充描述", key="mgmt_ui_text", height=70)
            figma_refs = st.text_area(
                "Figma 链接/File Key（每行一个）",
                key="mgmt_figma_refs",
                height=70,
            )

            st.markdown("**API接口资产**")
            api_files = st.file_uploader(
                "API接口资产（json/yml/yaml/txt/md/doc/docx/pdf）",
                type=["json", "yml", "yaml", "txt", "md", "doc", "docx", "pdf"],
                accept_multiple_files=True,
                key="mgmt_api_files",
            )
            api_text = st.text_area("API接口补充描述", key="mgmt_api_text", height=70)
            api_doc_links = st.text_area(
                "API接口文档链接（每行一个）",
                key="mgmt_api_links",
                height=70,
                placeholder="https://api.example.com/openapi.yaml",
            )

            btn_parse_col, btn_sync_col = st.columns([1, 1], gap="small")
            parse_clicked = btn_parse_col.button(
                "开始解析",
                use_container_width=True,
                key="mgmt_parse_btn",
            )
            sync_clicked = btn_sync_col.button(
                "同步至本地向量库",
                use_container_width=True,
                type="primary",
                key="mgmt_sync_btn",
            )

            sync_request = {
                "sync_clicked": sync_clicked,
                "parse_clicked": parse_clicked,
                "sync_mode": mode_map[mode_label],
                "req_files": req_files or [],
                "req_text": req_text,
                "feishu_docs": feishu_docs,
                "tc_files": tc_files or [],
                "tc_text": tc_text,
                "feishu_boards": feishu_boards,
                "ui_files": ui_files or [],
                "ui_text": ui_text,
                "figma_refs": figma_refs,
                "api_files": api_files or [],
                "api_text": api_text,
                "api_doc_links": api_doc_links,
                "kb_ingest_status": status_map[kb_ingest_status],
                "kb_ingest_release": kb_ingest_release,
                "kb_ingest_modules": kb_ingest_modules,
                "kb_ingest_trace_refs": kb_ingest_trace_refs,
            }

    parse_requested = bool(sync_request.get("sync_clicked") or sync_request.get("parse_clicked"))

    with col_right:
        with st.container(border=True):
            st.markdown("**本次待入库资产预览**")
            preview_rows: List[Dict[str, Any]] = []

            def _add_file_preview(group: str, source: str, files: List[Any]) -> None:
                for f in files or []:
                    preview_rows.append(
                        {
                            "资产分组": group,
                            "来源": _display_origin(source),
                            "内容": getattr(f, "name", "-"),
                        }
                    )

            _add_file_preview("需求", "file", sync_request["req_files"])
            _add_file_preview("测试用例", "file", sync_request["tc_files"])
            _add_file_preview("UI交互", "file", sync_request["ui_files"])
            _add_file_preview("API接口", "file", sync_request["api_files"])

            if (sync_request["req_text"] or "").strip():
                preview_rows.append(
                    {"资产分组": "需求", "来源": _display_origin("manual_text"), "内容": "需求补充文字"}
                )
            if (sync_request["tc_text"] or "").strip():
                preview_rows.append(
                    {
                        "资产分组": "测试用例",
                        "来源": _display_origin("manual_text"),
                        "内容": "测试用例补充描述",
                    }
                )
            if (sync_request["ui_text"] or "").strip():
                preview_rows.append(
                    {
                        "资产分组": "UI交互",
                        "来源": _display_origin("manual_text"),
                        "内容": "UI交互补充描述",
                    }
                )
            if (sync_request["api_text"] or "").strip():
                preview_rows.append(
                    {
                        "资产分组": "API接口",
                        "来源": _display_origin("manual_text"),
                        "内容": "API接口补充描述",
                    }
                )

            for _ in _split_lines(sync_request["feishu_docs"]):
                preview_rows.append(
                    {"资产分组": "需求", "来源": _display_origin("feishu_doc"), "内容": "飞书文档链接"}
                )
            for _ in _split_lines(sync_request["feishu_boards"]):
                preview_rows.append(
                    {
                        "资产分组": "测试用例",
                        "来源": _display_origin("feishu_board"),
                        "内容": "飞书画板链接",
                    }
                )
            for _ in _split_lines(sync_request["figma_refs"]):
                preview_rows.append(
                    {"资产分组": "UI交互", "来源": _display_origin("figma"), "内容": "Figma 引用"}
                )
            for _ in _split_lines(sync_request["api_doc_links"]):
                preview_rows.append(
                    {
                        "资产分组": "API接口",
                        "来源": _display_origin("api_doc_link"),
                        "内容": "API 文档链接",
                    }
                )

            st.dataframe(
                _to_dataframe_like(preview_rows, ["资产分组", "来源", "内容"]),
                use_container_width=True,
                hide_index=True,
            )
            if not preview_rows:
                st.info("左侧上传文件或填写描述后，此处将显示待入库资产清单。")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**图片资产解析预览**")

            image_candidates: List[tuple[str, Any]] = []
            doc_candidates: List[tuple[str, Any]] = []
            for group, files in (
                ("需求", sync_request["req_files"]),
                ("测试用例", sync_request["tc_files"]),
                ("UI交互", sync_request["ui_files"]),
                ("API接口", sync_request["api_files"]),
            ):
                for uploaded_file in files or []:
                    file_name = str(getattr(uploaded_file, "name", ""))
                    if _is_image_file_name(file_name):
                        image_candidates.append((group, uploaded_file))
                    else:
                        suffix = Path(file_name).suffix.lower()
                        if suffix in {".pdf", ".doc", ".docx"}:
                            doc_candidates.append((group, uploaded_file))

            if parse_requested:
                parsed_results: List[Dict[str, str]] = []
                parse_errors: List[str] = []
                doc_warnings: List[str] = []
                if image_candidates:
                    with st.spinner("视觉大模型正在解析图像，请稍候..."):
                        for group, uploaded_file in image_candidates:
                            file_name = str(getattr(uploaded_file, "name", "unknown"))
                            try:
                                parse_result = _process_uploaded_image_to_markdown(uploaded_file)
                                parsed_results.append(
                                    {
                                        "group": group,
                                        "file_name": file_name,
                                        "markdown": parse_result.get("markdown", ""),
                                        "engine": parse_result.get("engine", ""),
                                        "warning": parse_result.get("warning", ""),
                                    }
                                )
                            except Exception:
                                parse_errors.append(
                                    f"图片解析失败: {file_name}\n\n```text\n{traceback.format_exc()}\n```"
                                )
                else:
                    parse_errors.append("当前未检测到可解析图片（支持 png/jpg/jpeg）。")

                st.session_state["mgmt_image_parse_results"] = parsed_results
                st.session_state["mgmt_image_parse_errors"] = parse_errors
                if doc_candidates:
                    for group, uploaded_file in doc_candidates:
                        file_name = str(getattr(uploaded_file, "name", "unknown"))
                        text, warnings = tp._extract_task_text_from_file(uploaded_file)
                        if warnings:
                            doc_warnings.extend(warnings)
                        elif not str(text or "").strip():
                            doc_warnings.append(f"{file_name}: 文档解析结果为空或不可解析。")
                else:
                    doc_warnings.append("当前未检测到可解析文档（支持 pdf/doc/docx）。")
                st.session_state["mgmt_doc_parse_warnings"] = doc_warnings

            parse_results = st.session_state.get("mgmt_image_parse_results", [])
            parse_errors = st.session_state.get("mgmt_image_parse_errors", [])
            doc_warnings = st.session_state.get("mgmt_doc_parse_warnings", [])

            _, col_preview, _ = st.columns([1, 8, 1])
            with col_preview:
                if parse_results:
                    for idx, item in enumerate(parse_results, start=1):
                        st.markdown(f"**{idx}. [{item['group']}] {item['file_name']}**")
                        st.caption(f"识别方式: {_render_engine_label(str(item.get('engine', '')))}")
                        warning_text = str(item.get("warning", "")).strip()
                        if warning_text:
                            st.warning(f"OCR 提示: {warning_text}")
                        st.markdown(item.get("markdown", "").strip() or "_解析结果为空_")

                if parse_errors:
                    for err in parse_errors:
                        st.error(err)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**文档资产解析提示**")
                if doc_warnings:
                    for warn in doc_warnings:
                        st.error(warn)
                elif not parse_requested:
                    st.info("点击“开始解析”或“同步至本地向量库”后，这里将显示文档解析提示。")
                else:
                    st.caption("未检测到文档解析异常。")

                if not parse_requested and not parse_results and not parse_errors:
                    st.info("点击“开始解析”或“同步至本地向量库”后，这里将显示图片 Markdown 解析结果。")

    if sync_request["sync_clicked"]:
        _handle_kb_sync(sync_request)

    _render_kb_summary_sidebar()
    tp._render_sidebar_kb_operation_logs_panel(limit=20)
    tp._render_sidebar_kb_delete_panel()


def render_generation_hub(
    model_name: str,
    embedding_ok: bool,
    embedding_detail: str,
    chroma_ok: bool,
    chroma_detail: str,
) -> None:
    _render_sidebar_runtime_status(
        model_name=model_name,
        embedding_ok=embedding_ok,
        embedding_detail=embedding_detail,
        chroma_ok=chroma_ok,
        chroma_detail=chroma_detail,
    )

    _render_page_header("用例生成舱", "输入业务逻辑后，自动检索知识库并生成可审核测试用例。")

    with st.container(border=True):
        st.markdown("**生成参数配置**")
        generation_mode_label = st.selectbox(
            "接口用例生成模式",
            options=["1. 业务接口用例（业务流程优先）", "2. 字段校验用例（字段级）"],
            index=0,
            key="generation_mode_label",
        )
        generation_mode_map = {
            "1. 业务接口用例（业务流程优先）": "business_api",
            "2. 字段校验用例（字段级）": "field_validation",
        }
        generation_mode = generation_mode_map[generation_mode_label]
        lock_recommended_mode = st.checkbox(
            "锁定生成模式（禁用自动切换）",
            value=False,
            key="lock_recommended_mode",
            help="开启后，工作流不会根据评审反馈自动切换 business_api/field_validation。",
        )

        append_strategy_label = st.radio(
            "生成结果入库策略",
            options=[
                "1. 加入待审核队列（推荐）",
                "2. 仅生成，不入库",
                "3. 直接追加到知识库（跳过审核）",
            ],
            index=0,
            horizontal=True,
            key="append_strategy_label",
        )
        append_strategy_map = {
            "1. 加入待审核队列（推荐）": "review_queue",
            "2. 仅生成，不入库": "none",
            "3. 直接追加到知识库（跳过审核）": "direct_append",
        }
        append_strategy = append_strategy_map[append_strategy_label]

        cfg1, cfg2 = st.columns([1, 1], gap="large")
        with cfg1:
            retrieval_approved_only = st.checkbox(
                "检索仅使用已入库知识",
                value=True,
                key="retrieval_approved_only",
            )
            retrieval_release_filter = st.text_input(
                "检索版本过滤（可选）",
                key="retrieval_release_filter",
                placeholder="例如: v2026.03",
            )
        with cfg2:
            retrieval_module_filter = st.text_input(
                "检索模块过滤（可选，逗号分隔）",
                key="retrieval_module_filter",
                placeholder="例如: 订单, 支付",
            )
            generation_trace_refs = st.text_input(
                "本次生成追踪ID（可选，逗号分隔）",
                key="generation_trace_refs",
                placeholder="例如: REQ-123, API-45",
            )
            task_query_max_chars_default = tp._normalize_task_query_max_chars(
                os.getenv("TASK_QUERY_MAX_CHARS", "60000")
            )
            task_query_max_chars = int(
                st.number_input(
                    "输入长度上限（字符）",
                    min_value=12000,
                    max_value=200000,
                    value=task_query_max_chars_default,
                    step=2000,
                    key="hub_task_query_max_chars",
                    help="用于控制本次生成时文本+文件解析内容拼接后的最大长度，超出后会截断。",
                )
            )
        with st.expander("前置人机协作（生成前确认）", expanded=True):
            human_scope = st.text_area(
                "测试范围（scope）",
                height=80,
                key="human_scope",
                placeholder="例如：仅覆盖退票主流程+核心异常，不含退款清结算。",
            )
            human_exclusions = st.text_area(
                "排除项（exclusions）",
                height=70,
                key="human_exclusions",
                placeholder="例如：不覆盖历史数据迁移场景。",
            )
            human_risk_focus = st.text_area(
                "风险关注（risk_focus）",
                height=70,
                key="human_risk_focus",
                placeholder="例如：并发退票、状态幂等、库存回补一致性。",
            )
            human_priority_modules = st.text_input(
                "优先模块（priority_modules，可选，逗号分隔）",
                key="human_priority_modules",
                placeholder="默认回退检索模块过滤值",
            )
            human_confirmed_by_user = st.checkbox(
                "我已确认本次范围/排除项/风险关注，允许执行生成",
                value=False,
                key="human_confirmed_by_user",
            )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**核心输入区**")
        _, col_center, _ = st.columns([1, 8, 1])
        with col_center:
            st.markdown("<br>", unsafe_allow_html=True)
            task_core_text = st.text_area(
                "请输入本次待测的核心功能点（文本）",
                height=160,
                placeholder="示例: 增加观演人维度退票次数限制；每个实名观演人最多可发起 X 次退票。",
                key="task_core_text",
            )
            task_extra_text = st.text_area(
                "补充说明（可选）",
                height=100,
                placeholder="可补充边界条件、业务状态、角色差异、时间规则等；可写“模块: 订单, 支付, 风控”增强跨模块检索。",
                key="task_extra_text",
            )
            task_links_text = st.text_area(
                "本次待测关联链接（可选，每行一个）",
                height=100,
                placeholder="需求/接口/原型链接都可填写。",
                key="task_links_text",
            )
            task_input_files = st.file_uploader(
                "本次待测输入资产（可选）",
                type=[
                    "txt",
                    "md",
                    "csv",
                    "json",
                    "yml",
                    "yaml",
                    "doc",
                    "docx",
                    "pdf",
                    "png",
                    "jpg",
                    "jpeg",
                    "xmind",
                ],
                accept_multiple_files=True,
                key="task_input_files",
            )
            generate_clicked = st.button(
                "一键生成业务测试用例（支持多源输入）",
                type="primary",
                use_container_width=True,
            )

    if generate_clicked:
        composed_task_query, task_input_warnings, task_input_stats = tp._compose_task_query(
            core_text=task_core_text,
            extra_text=task_extra_text,
            links_text=task_links_text,
            files=task_input_files or [],
            max_chars=task_query_max_chars,
        )

        if not composed_task_query.strip():
            st.warning("请至少提供一种输入：文本、链接或文件。")
        else:
            st.caption(
                "本次输入汇总: "
                f"{task_input_stats.get('input_chars', 0)} 字符 | "
                f"文件 {task_input_stats.get('file_count', 0)} 个（成功解析 {task_input_stats.get('parsed_file_count', 0)}） | "
                f"链接 {task_input_stats.get('link_count', 0)} 条"
            )
            if task_input_warnings:
                st.info(
                    "输入解析提示: "
                    + " | ".join(task_input_warnings[:2])
                    + (" ..." if len(task_input_warnings) > 2 else "")
                )
            st.caption(
                "检索策略: "
                f"仅已入库={'是' if retrieval_approved_only else '否'} | "
                f"版本={retrieval_release_filter.strip() or '-'} | "
                f"模块={retrieval_module_filter.strip() or '-'}"
            )
            if not str(human_scope or "").strip():
                st.warning("请先填写前置人机协作的测试范围（scope）。")
                st.stop()
            if not bool(human_confirmed_by_user):
                st.warning("请勾选“已确认允许执行生成”后再继续。")
                st.stop()

            generated_markdown = ""
            try:
                retrieval_policy = {
                    "approved_only": retrieval_approved_only,
                    "release": (retrieval_release_filter or "").strip(),
                    "modules": tp._split_tags(retrieval_module_filter or ""),
                    "include_legacy_unlabeled": True,
                }
                priority_modules_value = tp._split_tags(
                    human_priority_modules or retrieval_module_filter or ""
                )
                confirmation_ts = datetime.now().isoformat(timespec="seconds")
                human_inputs_payload = {
                    "scope": str(human_scope or "").strip(),
                    "exclusions": str(human_exclusions or "").strip(),
                    "risk_focus": str(human_risk_focus or "").strip(),
                    "priority_modules": priority_modules_value,
                    "release": (retrieval_release_filter or "").strip(),
                    "approved_only": bool(retrieval_approved_only),
                    "run_context": {
                        "confirmed_by_user": bool(human_confirmed_by_user),
                        "confirmation_ts": confirmation_ts,
                    },
                }

                with st.spinner("正在运行 LangGraph 多智能体工作流..."):
                    get_augmented_context, llm, universal_template = tp.load_pipeline_components()
                    resolved_model_name = (
                        getattr(llm, "model_name", None)
                        or getattr(llm, "model", None)
                        or "unknown"
                    )
                    st.session_state["model_name"] = resolved_model_name
                    workflow_result = run_testcase_workflow(
                        task_query=composed_task_query,
                        get_augmented_context=get_augmented_context,
                        llm=llm,
                        universal_template=universal_template,
                        generation_mode=generation_mode,
                        recommended_mode_lock=(generation_mode if lock_recommended_mode else None),
                        human_inputs=human_inputs_payload,
                        max_iterations=2,
                        retrieval_policy=retrieval_policy,
                    )

                context_length = int(workflow_result.get("retrieval_context_len", 0) or 0)
                st.session_state["context_length"] = context_length
                st.session_state["workflow_last_run"] = workflow_result

                if context_length > 0:
                    st.success(f"检索完成，召回核心业务上下文: {context_length} 字符")
                else:
                    st.warning(
                        "当前工作流检索上下文为空，已基于原始需求生成结果。"
                        "可检查知识库状态或放宽筛选条件后重试。"
                    )

                intent_label = str(workflow_result.get("intent_label", "fallback"))
                iteration = int(workflow_result.get("iteration", 0) or 0)
                max_iterations = int(workflow_result.get("max_iterations", 2) or 2)
                review_result = workflow_result.get("review_result", {}) or {}
                review_decision = str(review_result.get("decision", "-")).strip().lower()
                final_status = str(workflow_result.get("final_status", "success")).strip().lower()
                review_label = _display_review_decision(review_decision)
                status_label = _display_workflow_status(final_status)
                st.caption(
                    "工作流结果: "
                    f"意图={intent_label} | "
                    f"迭代={iteration}/{max_iterations} | "
                    f"推荐模式={str(workflow_result.get('recommended_mode', generation_mode))} | "
                    f"链路边={int((workflow_result.get('link_summary', {}) or {}).get('total_edges', 0) or 0)} | "
                    f"已确认={ '是' if bool((workflow_result.get('run_context') or {}).get('confirmed_by_user', False)) else '否'} | "
                    f"评审={review_label} | "
                    f"状态={status_label}"
                )

                generated_markdown = str(
                    workflow_result.get("final_testcases_md", "")
                    or workflow_result.get("draft_testcases_md", "")
                    or ""
                )
                if not bool(workflow_result.get("langgraph_enabled", True)):
                    st.info("当前环境未启用 LangGraph，已自动回退线性流程。")
                if final_status == "success_with_warning":
                    st.warning("评审未完全通过，达到最大重试后输出当前最优草稿。")
                elif final_status == "failed":
                    st.error("工作流未产出有效结果，请调整输入后重试。")

                if not generated_markdown.strip():
                    st.warning("模型返回为空，请调整输入后重试。")

                st.session_state["generated_markdown"] = generated_markdown
                st.session_state["generated_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")

                if generated_markdown.strip():
                    if append_strategy == "direct_append":
                        try:
                            append_summary = tp._append_generated_markdown_to_kb(
                                markdown_text=generated_markdown,
                                generation_mode=generation_mode,
                                task_query=composed_task_query,
                                generated_at=st.session_state["generated_at"],
                                module_text=retrieval_module_filter,
                                release_text=retrieval_release_filter,
                                trace_refs_text=generation_trace_refs,
                                risk_report=workflow_result.get("risk_report", {}),
                            )
                            if _is_append_effective_success(append_summary):
                                if int(append_summary.get("ingested_assets", 0) or 0) > 0:
                                    st.success("已将本次生成结果直接追加进知识库。")
                                else:
                                    st.success(
                                        "检测到重复内容，本次未重复入库（幂等命中）。"
                                    )
                            elif bool(append_summary.get("blocked_by_policy", False)):
                                st.warning(
                                    "生成成功，但入库被阻断："
                                    + " | ".join(append_summary.get("errors", [])[:2])
                                )
                            else:
                                st.warning(
                                    "生成成功，但追加知识库失败："
                                    + " | ".join(append_summary.get("errors", [])[:2])
                                )
                        except Exception as exc:
                            st.warning(f"生成成功，但追加知识库异常: {exc}")

                    if append_strategy == "review_queue":
                        review_id = f"{st.session_state['generated_at']}_{uuid4().hex[:8]}"
                        route_history = workflow_result.get("route_history", []) or []
                        st.session_state["review_queue"].insert(
                            0,
                            {
                                "id": review_id,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "generated_at": st.session_state["generated_at"],
                                "status": "pending",
                                "generation_mode": generation_mode,
                                "task_query": composed_task_query[:3000],
                                "context_length": st.session_state["context_length"],
                                "content": generated_markdown,
                                "module_text": retrieval_module_filter,
                                "release_text": retrieval_release_filter,
                                "trace_refs_text": generation_trace_refs,
                                "review_result": review_result if isinstance(review_result, dict) else {},
                                "review_passed": bool(workflow_result.get("review_passed", False)),
                                "compliance_report": workflow_result.get("compliance_report", {}),
                                "risk_report": workflow_result.get("risk_report", {}),
                                "human_inputs": workflow_result.get("human_inputs", {}),
                                "run_context": workflow_result.get("run_context", {}),
                                "link_edges": workflow_result.get("link_edges", []),
                                "trace_refs": workflow_result.get("trace_refs", {}),
                                "link_summary": workflow_result.get("link_summary", {}),
                                "workflow_summary": {
                                    "intent_label": str(workflow_result.get("intent_label", "fallback")),
                                    "iteration": int(workflow_result.get("iteration", 0) or 0),
                                    "max_iterations": int(workflow_result.get("max_iterations", 2) or 2),
                                    "recommended_mode": str(
                                        workflow_result.get("recommended_mode", generation_mode)
                                    ),
                                    "recommended_mode_lock": str(
                                        workflow_result.get("recommended_mode_lock", "")
                                    ),
                                    "final_status": str(workflow_result.get("final_status", "unknown")),
                                    "route_history": [str(x) for x in route_history],
                                },
                            },
                        )
                        _persist_review_queue_or_warn()
                        st.success("已加入待审核队列。请到“资产审核”页面逐条审核后再入库。")
            except Exception as exc:
                exc_text = str(exc)
                if (
                    ("all-MiniLM-L6-v2" in exc_text)
                    or ("huggingface.co" in exc_text)
                    or ("EMBEDDING_MODEL_PATH" in exc_text)
                    or ("config.json" in exc_text and "Embedding" in exc_text)
                ):
                    st.error(
                        "检索阶段异常: 本地 Embedding 模型未就绪（all-MiniLM-L6-v2）。\n\n"
                        "请先确认 HuggingFace 本地缓存完整，或设置 EMBEDDING_MODEL_PATH 指向包含 config.json 的本地模型目录。"
                    )
                else:
                    st.error(f"多智能体生成阶段异常: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**结果展示区**")
        _, col_preview, _ = st.columns([1, 8, 1])
        with col_preview:
            workflow_meta = st.session_state.get("workflow_last_run")
            if isinstance(workflow_meta, dict):
                error_text = str(workflow_meta.get("error", "")).strip()
                if error_text.lower() in {"none", "null"}:
                    error_text = ""
                if error_text:
                    st.error(f"工作流异常: {error_text}")
            if st.session_state.get("context_length"):
                st.caption(f"最近一次检索上下文长度: {st.session_state['context_length']} 字符")
            if isinstance(workflow_meta, dict):
                gap_hints = workflow_meta.get("gap_hints", {}) or {}
                _render_gap_hints(gap_hints if isinstance(gap_hints, dict) else {})
                compliance_report = workflow_meta.get("compliance_report", {}) or {}
                _render_compliance_report(
                    compliance_report if isinstance(compliance_report, dict) else {}
                )
                risk_report = workflow_meta.get("risk_report", {}) or {}
                _render_risk_report(risk_report if isinstance(risk_report, dict) else {})
                badcase_replay = workflow_meta.get("badcase_replay", {}) or {}
                _render_badcase_replay_report(
                    badcase_replay if isinstance(badcase_replay, dict) else {},
                    key_prefix="generation_result",
                    enable_window_actions=False,
                )
                impact_analysis = workflow_meta.get("impact_analysis", {}) or {}
                _render_impact_analysis(impact_analysis if isinstance(impact_analysis, dict) else {})
                _render_link_analysis(
                    workflow_meta.get("link_edges", []),
                    workflow_meta.get("trace_refs", {}),
                    workflow_meta.get("link_summary", {}),
                )
                contracts = workflow_meta.get("contracts", {}) or {}
                mapping_rules = workflow_meta.get("mapping_rules", {}) or {}
                coverage_matrix = workflow_meta.get("coverage_matrix", {}) or {}
                _render_generation_support(
                    contracts if isinstance(contracts, dict) else {},
                    mapping_rules if isinstance(mapping_rules, dict) else {},
                    coverage_matrix if isinstance(coverage_matrix, dict) else {},
                )
                intent_label = str(workflow_meta.get("intent_label", "fallback"))
                iteration = int(workflow_meta.get("iteration", 0) or 0)
                max_iterations = int(workflow_meta.get("max_iterations", 2) or 2)
                final_status = str(workflow_meta.get("final_status", "unknown"))
                status_label = _display_workflow_status(final_status)
                run_context = workflow_meta.get("run_context", {}) or {}
                confirmed = bool(run_context.get("confirmed_by_user", False)) if isinstance(run_context, dict) else False
                confirmation_ts = (
                    str(run_context.get("confirmation_ts", "")).strip()
                    if isinstance(run_context, dict)
                    else ""
                )
                st.caption(
                    "多智能体执行摘要: "
                    f"意图={intent_label} | "
                    f"迭代={iteration}/{max_iterations} | "
                    f"前置确认={'是' if confirmed else '否'} | "
                    f"状态={status_label}"
                )
                if confirmed:
                    st.caption(f"确认时间: {confirmation_ts or '-'}")
                with st.expander("查看工作流评审详情", expanded=False):
                    review_result = workflow_meta.get("review_result", {}) or {}
                    _render_review_result_detail(
                        review_result if isinstance(review_result, dict) else {}
                    )
                    route_history = workflow_meta.get("route_history", []) or []
                    st.write("路由轨迹: " + " -> ".join([str(x) for x in route_history]))

            if st.session_state.get("generated_markdown"):
                st.markdown(st.session_state["generated_markdown"])
                download_name = f"TestCases_{st.session_state.get('generated_at', 'latest')}.md"
                st.download_button(
                    label="下载为 Markdown 文件",
                    data=st.session_state["generated_markdown"],
                    file_name=download_name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.info("等待输入业务逻辑，AI 将在此生成推演结果...")


def render_asset_review(
    model_name: str,
    embedding_ok: bool,
    embedding_detail: str,
    chroma_ok: bool,
    chroma_detail: str,
) -> None:
    _render_sidebar_runtime_status(
        model_name=model_name,
        embedding_ok=embedding_ok,
        embedding_detail=embedding_detail,
        chroma_ok=chroma_ok,
        chroma_detail=chroma_detail,
    )

    _render_page_header("资产审核", "审核通过后，测试用例才会正式写入知识库。")

    queue: List[Dict[str, Any]] = st.session_state.get("review_queue") or []
    status_text_map = {
        "pending": "待审核",
        "approved": "已入库",
        "rejected": "已驳回",
    }
    status_value_map = {v: k for k, v in status_text_map.items()}

    pending_count = sum(1 for item in queue if item.get("status") == "pending")
    approved_count = sum(1 for item in queue if item.get("status") == "approved")
    rejected_count = sum(1 for item in queue if item.get("status") == "rejected")

    with st.container(border=True):
        col_waiting, col_done, col_rejected, _ = st.columns([1, 1, 1, 2], gap="small")
        with col_waiting:
            st.metric(label="待审核", value=str(pending_count))
        with col_done:
            st.metric(label="已入库", value=str(approved_count))
        with col_rejected:
            st.metric(label="已驳回", value=str(rejected_count))

    with st.container(border=True):
        st.markdown("**队列备份与恢复**")
        export_payload = {
            "schema_version": tp.REVIEW_QUEUE_SCHEMA_VERSION,
            "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(queue),
            "items": queue,
        }
        col_export, col_import = st.columns([1, 1], gap="large")
        with col_export:
            st.download_button(
                label="导出审核队列（JSON）",
                data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                file_name=f"review_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
                key="review_export_btn",
            )
            st.caption("用于本地备份、跨环境迁移和回滚恢复。")

        with col_import:
            import_mode = st.radio(
                "导入模式",
                options=["合并（按 ID 去重）", "覆盖现有队列"],
                horizontal=True,
                key="review_import_mode",
            )
            import_file = st.file_uploader(
                "导入审核队列 JSON",
                type=["json"],
                key="review_import_file",
            )
            import_bytes = import_file.getvalue() if import_file else b""
            import_sha = hashlib.sha256(import_bytes).hexdigest() if import_bytes else ""

            btn_preview_col, btn_apply_col = st.columns([1, 1], gap="small")
            if btn_preview_col.button("预校验导入文件", use_container_width=True, key="review_import_preview_btn"):
                if not import_file:
                    st.warning("请先上传 JSON 文件。")
                else:
                    try:
                        preview_payload = _build_import_preview(import_bytes)
                        st.session_state["review_import_preview"] = preview_payload
                        st.success("预校验完成。请先确认报告，再点击“确认导入”。")
                    except Exception as exc:
                        st.error(f"预校验失败: {exc}")

            preview_state = st.session_state.get("review_import_preview")
            preview_ready = (
                isinstance(preview_state, dict)
                and bool(import_sha)
                and str(preview_state.get("file_sha", "")) == import_sha
            )

            if preview_state and not preview_ready:
                st.info("检测到导入文件发生变化，请重新点击“预校验导入文件”。")

            if preview_ready:
                st.markdown("**导入预校验报告**")
                p1, p2, p3, p4 = st.columns(4, gap="small")
                p1.metric("原始行数", str(preview_state.get("total_rows", 0)))
                p2.metric("有效行数", str(preview_state.get("valid_rows", 0)))
                p3.metric("去重后行数", str(preview_state.get("deduped_rows", 0)))
                p4.metric("无效行数", str(preview_state.get("invalid_rows", 0)))
                st.caption(
                    "校验信息: "
                    f"缺失ID补全 {preview_state.get('missing_id_count', 0)} 条 | "
                    f"状态归一化 {preview_state.get('normalized_status_count', 0)} 条 | "
                    f"文件内重复ID {preview_state.get('duplicate_id_count', 0)} 个"
                )

                preview_rows = preview_state.get("preview_rows", [])
                st.dataframe(
                    _to_dataframe_like(
                        preview_rows,
                        ["记录ID", "状态", "生成时间", "模块", "需求摘要"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            if btn_apply_col.button("确认导入", use_container_width=True, key="review_import_btn"):
                if not import_file:
                    st.warning("请先上传 JSON 文件。")
                elif not preview_ready:
                    st.warning("请先完成预校验，并确保文件未变化。")
                else:
                    try:
                        imported_items = preview_state.get("items", [])
                        if import_mode == "覆盖现有队列":
                            merged_queue = list(imported_items)
                        else:
                            merged_map = {
                                str(item.get("id", "")): dict(item)
                                for item in queue
                                if isinstance(item, dict) and str(item.get("id", "")).strip()
                            }
                            for item in imported_items:
                                if isinstance(item, dict):
                                    merged_map[str(item.get("id", ""))] = dict(item)
                            merged_queue = list(merged_map.values())

                        merged_queue.sort(
                            key=lambda x: str(x.get("created_at", "")),
                            reverse=True,
                        )
                        st.session_state["review_queue"] = merged_queue
                        if _persist_review_queue_or_warn():
                            st.session_state.pop("review_import_preview", None)
                            st.success(f"导入成功，当前队列共 {len(merged_queue)} 条。")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"导入失败: {exc}")

    if not queue:
        with st.container(border=True):
            st.info("暂无待审核内容。可在“用例生成舱”选择“加入待审核队列（推荐）”。")
        return

    with st.container(border=True):
        st.markdown("**队列筛选与批量操作**")
        c1, c2, c3, c4 = st.columns([2, 3, 2, 2], gap="small")
        with c1:
            selected_status_labels = st.multiselect(
                "审核状态",
                options=list(status_value_map.keys()),
                default=["待审核", "已入库", "已驳回"],
                key="review_status_filter",
            )
            selected_status_values = {
                status_value_map[label]
                for label in selected_status_labels
                if label in status_value_map
            }
        with c2:
            keyword = st.text_input(
                "关键词检索（ID/模块/需求/内容）",
                key="review_keyword_filter",
                placeholder="输入关键字过滤队列",
            ).strip()
        with c3:
            page_size = st.selectbox(
                "每页条数",
                options=[5, 10, 20, 50],
                index=1,
                key="review_page_size",
            )
        with c4:
            sort_order = st.selectbox(
                "排序方式",
                options=["时间倒序", "时间正序"],
                index=0,
                key="review_sort_order",
            )

        def _matches_keyword(item: Dict[str, Any], q: str) -> bool:
            if not q:
                return True
            haystack = " ".join(
                [
                    str(item.get("id", "")),
                    str(item.get("generation_mode", "")),
                    str(item.get("module_text", "")),
                    str(item.get("release_text", "")),
                    str(item.get("task_query", "")),
                    str(item.get("content", ""))[:1200],
                ]
            ).lower()
            return q.lower() in haystack

        filtered_queue = []
        for item in queue:
            status = str(item.get("status", "pending")).strip().lower()
            if selected_status_values and status not in selected_status_values:
                continue
            if not _matches_keyword(item, keyword):
                continue
            filtered_queue.append(item)

        filtered_queue.sort(
            key=lambda x: str(x.get("created_at", "")),
            reverse=(sort_order == "时间倒序"),
        )

        total_filtered = len(filtered_queue)
        page_count = max(1, ceil(total_filtered / int(page_size)))
        page_key = "review_page_no"
        try:
            cached_page = int(st.session_state.get(page_key, 1))
        except Exception:
            cached_page = 1
        st.session_state[page_key] = min(max(1, cached_page), page_count)
        page_no = int(
            st.number_input(
                "页码",
                min_value=1,
                max_value=page_count,
                step=1,
                key=page_key,
            )
        )
        start = (page_no - 1) * int(page_size)
        end = start + int(page_size)
        page_items = filtered_queue[start:end]

        st.caption(
            f"总计 {len(queue)} 条 | 筛选后 {total_filtered} 条 | 当前第 {page_no}/{page_count} 页"
        )

        option_map = {
            _queue_row_label(item, status_text_map): str(item.get("id", ""))
            for item in page_items
        }
        selected_rows = st.multiselect(
            "批量选择（当前页）",
            options=list(option_map.keys()),
            key="review_batch_selection",
        )
        selected_ids = {option_map[label] for label in selected_rows if label in option_map}

        batch_col1, batch_col2, batch_col3, batch_col4 = st.columns([1, 1, 1, 2], gap="small")
        if batch_col1.button(
            "批量通过入库",
            key="review_batch_approve",
            type="primary",
            use_container_width=True,
        ):
            if not selected_ids:
                st.warning("请先选择要批量通过的记录。")
            else:
                st.session_state["review_batch_pending_action"] = {
                    "action": "approve",
                    "ids": sorted(selected_ids),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

        if batch_col2.button(
            "批量驳回",
            key="review_batch_reject",
            use_container_width=True,
        ):
            if not selected_ids:
                st.warning("请先选择要批量驳回的记录。")
            else:
                st.session_state["review_batch_pending_action"] = {
                    "action": "reject",
                    "ids": sorted(selected_ids),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

        if batch_col3.button(
            "批量删除",
            key="review_batch_remove",
            use_container_width=True,
        ):
            if not selected_ids:
                st.warning("请先选择要删除的记录。")
            else:
                st.session_state["review_batch_pending_action"] = {
                    "action": "remove",
                    "ids": sorted(selected_ids),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

        if batch_col4.button(
            "清理已处理记录（仅保留待审核）",
            key="review_cleanup_done",
            use_container_width=True,
        ):
            st.session_state["review_batch_pending_action"] = {
                "action": "cleanup_done",
                "ids": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        pending_action = st.session_state.get("review_batch_pending_action")
        if isinstance(pending_action, dict):
            action = str(pending_action.get("action", "")).strip()
            action_ids = {
                str(v) for v in (pending_action.get("ids") or []) if str(v).strip()
            }
            action_count = len(action_ids)
            action_label_map = {
                "approve": "批量通过入库",
                "reject": "批量驳回",
                "remove": "批量删除",
                "cleanup_done": "清理已处理记录",
            }
            action_label = action_label_map.get(action, action)
            if action == "cleanup_done":
                action_desc = "将移除所有已入库/已驳回记录，仅保留待审核。"
            else:
                action_desc = f"将处理 {action_count} 条记录。"

            st.markdown("<br>", unsafe_allow_html=True)
            st.warning(f"待确认操作：{action_label}。{action_desc}")
            c_confirm, c_cancel, _ = st.columns([1, 1, 3], gap="small")

            if c_confirm.button("确认执行", key="review_batch_confirm_execute", type="primary"):
                current_queue: List[Dict[str, Any]] = st.session_state.get("review_queue") or []
                if action == "approve":
                    approved_new_count = 0
                    duplicate_count = 0
                    failed_count = 0
                    failed_msgs: List[str] = []
                    with st.spinner("正在批量通过并入库..."):
                        for item in current_queue:
                            item_id = str(item.get("id", ""))
                            if item_id not in action_ids:
                                continue
                            if str(item.get("status", "pending")) == "approved":
                                continue

                            content_key = f"review_content_{item_id}"
                            content_text = str(
                                st.session_state.get(content_key, item.get("content", ""))
                            ).strip()
                            if not content_text:
                                failed_count += 1
                                failed_msgs.append(f"{item_id}: 内容为空")
                                continue

                            try:
                                append_summary = tp._append_generated_markdown_to_kb(
                                    markdown_text=content_text,
                                    generation_mode=str(item.get("generation_mode", "-")),
                                    task_query=str(item.get("task_query", "")),
                                    generated_at=str(item.get("generated_at", item.get("created_at", ""))),
                                    review_id=item_id,
                                    module_text=str(item.get("module_text", "")),
                                    release_text=str(item.get("release_text", "")),
                                    trace_refs_text=str(item.get("trace_refs_text", "")),
                                    risk_report=item.get("risk_report", {}),
                                )
                                if _is_append_effective_success(append_summary):
                                    item["status"] = "approved"
                                    item["content"] = content_text
                                    item["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    if int(append_summary.get("ingested_assets", 0) or 0) > 0:
                                        approved_new_count += 1
                                    else:
                                        duplicate_count += 1
                                else:
                                    failed_count += 1
                                    err = (
                                        " | ".join(append_summary.get("errors", [])[:2])
                                        or "入库失败"
                                    )
                                    failed_msgs.append(f"{item_id}: {err}")
                            except Exception as exc:
                                failed_count += 1
                                failed_msgs.append(f"{item_id}: {exc}")

                    st.session_state.pop("review_batch_pending_action", None)
                    if _persist_review_queue_or_warn():
                        st.success(
                            f"批量通过完成：新增入库 {approved_new_count} 条，幂等命中 {duplicate_count} 条，失败 {failed_count} 条。"
                        )
                        if failed_msgs:
                            st.warning("失败明细: " + " | ".join(failed_msgs[:3]))
                        st.rerun()

                elif action == "reject":
                    changed = 0
                    for item in current_queue:
                        item_id = str(item.get("id", ""))
                        if item_id not in action_ids:
                            continue
                        if str(item.get("status", "pending")) == "approved":
                            continue
                        item["status"] = "rejected"
                        item["content"] = str(
                            st.session_state.get(f"review_content_{item_id}", item.get("content", ""))
                        )
                        item["rejected_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        changed += 1
                    st.session_state.pop("review_batch_pending_action", None)
                    if changed and _persist_review_queue_or_warn():
                        st.success(f"已批量驳回 {changed} 条。")
                        st.rerun()
                    if changed == 0:
                        st.info("没有可驳回的记录。")

                elif action == "remove":
                    st.session_state["review_queue"] = [
                        item
                        for item in current_queue
                        if str(item.get("id", "")) not in action_ids
                    ]
                    st.session_state.pop("review_batch_pending_action", None)
                    if _persist_review_queue_or_warn():
                        st.success(f"已删除 {action_count} 条记录。")
                        st.rerun()

                elif action == "cleanup_done":
                    st.session_state["review_queue"] = [
                        item
                        for item in current_queue
                        if str(item.get("status", "pending")) == "pending"
                    ]
                    st.session_state.pop("review_batch_pending_action", None)
                    if _persist_review_queue_or_warn():
                        st.success("已清理已入库/已驳回记录。")
                        st.rerun()

            if c_cancel.button("取消操作", key="review_batch_confirm_cancel", use_container_width=True):
                st.session_state.pop("review_batch_pending_action", None)
                st.info("已取消本次批量操作。")
                st.rerun()

    with st.container(border=True):
        _, col_table_width, _ = st.columns([1, 10, 1])
        with col_table_width:
            if not page_items:
                st.info("当前筛选条件下暂无数据。")
            table_rows = [
                {
                    "记录ID": str(item.get("id", ""))[-8:],
                    "资产名称": (
                        f"生成用例_{_display_generation_mode(item.get('generation_mode', '-'))}_"
                        f"{item.get('generated_at', '-')}"
                    ),
                    "模块标签": str(item.get("module_text", "")).strip() or "-",
                    "关联需求": (str(item.get("task_query", "")).strip()[:30] + "...") if str(item.get("task_query", "")).strip() else "-",
                    "生成时间": str(item.get("created_at", "-")),
                    "审核状态": status_text_map.get(str(item.get("status", "pending")), "待审核"),
                }
                for item in page_items
            ]
            st.dataframe(
                _to_dataframe_like(
                    table_rows,
                    ["记录ID", "资产名称", "模块标签", "关联需求", "生成时间", "审核状态"],
                ),
                use_container_width=True,
                hide_index=True,
            )

    with st.container(border=True):
        st.markdown("**审核操作区（当前页）**")
        remove_ids: set[str] = set()
        for item in page_items:
            item_id = str(item.get("id", ""))
            status = str(item.get("status", "pending"))
            created_at = str(item.get("created_at", "-"))
            generation_mode = str(item.get("generation_mode", "-"))
            generation_mode_label = _display_generation_mode(generation_mode)
            title = f"{status_text_map.get(status, status)} | {created_at} | {generation_mode_label}"

            with st.expander(title, expanded=(status == "pending")):
                st.caption(f"检索上下文长度: {int(item.get('context_length', 0))} 字符")
                task_preview = str(item.get("task_query", "")).strip()
                if task_preview:
                    st.caption("任务快照（截断）")
                    st.code(task_preview[:600], language="text")

                module_text = str(item.get("module_text", "")).strip()
                release_text = str(item.get("release_text", "")).strip()
                if module_text or release_text:
                    st.caption(
                        "元数据: "
                        f"模块={module_text or '-'} | 版本={release_text or '-'}"
                    )

                review_result = item.get("review_result", {})
                with st.expander("查看自动评审详情", expanded=(status == "pending")):
                    _render_review_result_detail(
                        review_result if isinstance(review_result, dict) else {}
                    )
                    workflow_summary = (
                        item.get("workflow_summary", {})
                        if isinstance(item.get("workflow_summary"), dict)
                        else {}
                    )
                    route_history = workflow_summary.get("route_history", []) or []
                    if route_history:
                        st.caption("路由轨迹")
                        st.write(" -> ".join([str(x) for x in route_history]))
                    if workflow_summary:
                        status_label = _display_workflow_status(
                            workflow_summary.get("final_status", "-")
                        )
                        st.caption(
                            "工作流摘要: "
                            f"意图={workflow_summary.get('intent_label', '-')} | "
                            f"迭代={workflow_summary.get('iteration', '-')} / "
                            f"{workflow_summary.get('max_iterations', '-')} | "
                            f"状态={status_label}"
                        )

                content_key = f"review_content_{item_id}"
                if content_key not in st.session_state:
                    st.session_state[content_key] = str(item.get("content", ""))
                edited_content = st.text_area(
                    "审核内容（可删改后再通过入库）",
                    key=content_key,
                    height=260,
                )

                approve_disabled = status == "approved"
                reject_disabled = status == "approved"
                cols = st.columns(3)

                if cols[0].button(
                    "通过并追加到知识库",
                    key=f"review_approve_{item_id}",
                    type="primary",
                    disabled=approve_disabled,
                    use_container_width=True,
                ):
                    if not edited_content.strip():
                        st.warning("审核内容为空，无法入库。")
                    else:
                        try:
                            append_summary = tp._append_generated_markdown_to_kb(
                                markdown_text=edited_content,
                                generation_mode=generation_mode,
                                task_query=task_preview,
                                generated_at=str(item.get("generated_at", created_at)),
                                review_id=item_id,
                                module_text=str(item.get("module_text", "")),
                                release_text=str(item.get("release_text", "")),
                                trace_refs_text=str(item.get("trace_refs_text", "")),
                                risk_report=item.get("risk_report", {}),
                            )
                            if _is_append_effective_success(append_summary):
                                item["status"] = "approved"
                                item["content"] = edited_content
                                item["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                if _persist_review_queue_or_warn():
                                    if int(append_summary.get("ingested_assets", 0) or 0) > 0:
                                        st.success("审核通过，已追加进知识库。")
                                    else:
                                        st.success("审核通过：内容已存在知识库，本次未重复写入。")
                                    st.rerun()
                            elif bool(append_summary.get("blocked_by_policy", False)):
                                st.warning(
                                    "阻断入库: " + " | ".join(append_summary.get("errors", [])[:2])
                                )
                            else:
                                st.error(
                                    "追加失败: "
                                    + " | ".join(append_summary.get("errors", [])[:2])
                                )
                        except Exception as exc:
                            st.error(f"审核入库异常: {exc}")

                if cols[1].button(
                    "驳回",
                    key=f"review_reject_{item_id}",
                    disabled=reject_disabled,
                    use_container_width=True,
                ):
                    item["status"] = "rejected"
                    item["content"] = edited_content
                    item["rejected_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if _persist_review_queue_or_warn():
                        st.info("已驳回，该条不会入库。")
                        st.rerun()

                if cols[2].button(
                    "移除记录",
                    key=f"review_remove_{item_id}",
                    use_container_width=True,
                ):
                    remove_ids.add(item_id)

                if status == "approved":
                    st.success(f"入库时间: {item.get('approved_at', '-')}")
                elif status == "rejected":
                    st.warning(f"驳回时间: {item.get('rejected_at', '-')}")

        if remove_ids:
            st.session_state["review_queue"] = [
                item for item in queue if str(item.get("id", "")) not in remove_ids
            ]
            if _persist_review_queue_or_warn():
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="测试用例生成平台", layout="wide")
    _inject_ui_theme()
    tp._init_session_state()

    st.title("测试用例生成平台")

    nav = _render_nav_menu()

    embedding_ok, embedding_detail = tp._ensure_embedding_env_path()
    chroma_ok, chroma_detail = tp._check_chroma_status()
    model_name = st.session_state.get("model_name", "待初始化")

    if nav == "资产看板":
        render_asset_dashboard(
            model_name=model_name,
            embedding_ok=embedding_ok,
            embedding_detail=embedding_detail,
            chroma_ok=chroma_ok,
            chroma_detail=chroma_detail,
        )
    elif nav == "知识库管理":
        render_kb_management(
            model_name=model_name,
            embedding_ok=embedding_ok,
            embedding_detail=embedding_detail,
            chroma_ok=chroma_ok,
            chroma_detail=chroma_detail,
        )
    elif nav == "用例生成舱":
        render_generation_hub(
            model_name=model_name,
            embedding_ok=embedding_ok,
            embedding_detail=embedding_detail,
            chroma_ok=chroma_ok,
            chroma_detail=chroma_detail,
        )
    else:
        render_asset_review(
            model_name=model_name,
            embedding_ok=embedding_ok,
            embedding_detail=embedding_detail,
            chroma_ok=chroma_ok,
            chroma_detail=chroma_detail,
        )


if __name__ == "__main__":
    main()
