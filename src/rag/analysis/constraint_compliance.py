from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _split_terms(value: Any, max_items: int = 12) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,\n，、;/；|]+", str(value))
    terms: List[str] = []
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        normalized = _normalize_text(text)
        if not normalized or normalized in terms:
            continue
        terms.append(normalized)
        if len(terms) >= max_items:
            break
    return terms


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    tokens: List[str] = []
    seen = set()

    for token in re.findall(r"[a-z0-9_]+", normalized):
        if token not in seen:
            seen.add(token)
            tokens.append(token)

    cn_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    for ch in cn_chars:
        if ch not in seen:
            seen.add(ch)
            tokens.append(ch)
    for i in range(len(cn_chars) - 1):
        bi = cn_chars[i] + cn_chars[i + 1]
        if bi not in seen:
            seen.add(bi)
            tokens.append(bi)
    return tokens


def _char_ngrams(text: str, n: int = 2) -> List[str]:
    cleaned = re.sub(r"\s+", "", _normalize_text(text))
    if not cleaned:
        return []
    if len(cleaned) <= n:
        return [cleaned]
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def _build_evidence_index(draft_md: str, max_items: int = 180) -> List[Dict[str, Any]]:
    lines = str(draft_md or "").splitlines()
    raw_items: List[Tuple[str, str]] = []

    for line_no, raw_line in enumerate(lines, start=1):
        text = str(raw_line).strip()
        if not text:
            continue
        raw_items.append((f"L{line_no}", text))

    para_chunks = re.split(r"\n{2,}", str(draft_md or ""))
    for para_idx, raw_para in enumerate(para_chunks, start=1):
        text = re.sub(r"\s+", " ", str(raw_para).strip())
        if len(text) < 24:
            continue
        raw_items.append((f"P{para_idx}", text))

    evidence: List[Dict[str, Any]] = []
    seen_text = set()
    for evidence_id, text in raw_items:
        norm = _normalize_text(text)
        if not norm or norm in seen_text:
            continue
        seen_text.add(norm)
        item = {
            "id": evidence_id,
            "text": text[:300],
            "normalized": norm,
            "tokens": _tokenize(text),
            "char_ngrams": _char_ngrams(text, n=2),
        }
        evidence.append(item)
        if len(evidence) >= max_items:
            break
    return evidence


def _semantic_score(term: str, evidence: Dict[str, Any]) -> float:
    term_norm = _normalize_text(term)
    if not term_norm:
        return 0.0
    text_norm = str(evidence.get("normalized", ""))
    if not text_norm:
        return 0.0
    if term_norm in text_norm:
        return 1.0

    term_tokens = _tokenize(term_norm)
    text_tokens = evidence.get("tokens", [])
    if not isinstance(text_tokens, list):
        text_tokens = _tokenize(text_norm)
    token_overlap = 0.0
    if term_tokens:
        token_overlap = len(set(term_tokens) & set(text_tokens)) / float(len(set(term_tokens)))

    term_ngrams = _char_ngrams(term_norm, n=2)
    text_ngrams = evidence.get("char_ngrams", [])
    if not isinstance(text_ngrams, list):
        text_ngrams = _char_ngrams(text_norm, n=2)
    char_jaccard = _jaccard(term_ngrams, text_ngrams)

    seq_ratio = SequenceMatcher(None, term_norm, text_norm[:600]).ratio()
    term_cn = set(re.findall(r"[\u4e00-\u9fff]", term_norm))
    text_cn = set(re.findall(r"[\u4e00-\u9fff]", text_norm))
    char_cover = 0.0
    if term_cn:
        char_cover = len(term_cn & text_cn) / float(len(term_cn))

    score = max(
        seq_ratio * 0.88,
        (0.60 * token_overlap) + (0.40 * char_jaccard),
        char_jaccard * 0.92,
        (0.75 * char_cover) + (0.25 * char_jaccard),
    )
    return max(0.0, min(1.0, score))


def build_constraint_dsl(human_inputs: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = human_inputs if isinstance(human_inputs, dict) else {}
    custom = raw.get("constraint_dsl")
    if isinstance(custom, dict):
        include_all = _split_terms(custom.get("include_all", []), max_items=12)
        include_any = _split_terms(custom.get("include_any", []), max_items=12)
        exclude_any = _split_terms(custom.get("exclude_any", []), max_items=12)
        risk_focus = _split_terms(custom.get("risk_focus", []), max_items=12)
        thresholds_raw = custom.get("thresholds", {})
        if not isinstance(thresholds_raw, dict):
            thresholds_raw = {}
    else:
        include_all = _split_terms(raw.get("must_cover", []), max_items=12)
        include_any = []
        exclude_any = _split_terms(raw.get("must_not_cover", []), max_items=12)
        risk_focus = _split_terms(raw.get("risk_tags", []), max_items=12)
        thresholds_raw = {}

    def _f(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    thresholds = {
        "include_min_score": max(0.35, min(0.95, _f(thresholds_raw.get("include_min_score"), 0.52))),
        "exclude_hit_score": max(0.35, min(0.98, _f(thresholds_raw.get("exclude_hit_score"), 0.55))),
        "risk_min_score": max(0.30, min(0.95, _f(thresholds_raw.get("risk_min_score"), 0.48))),
        "risk_min_hits_ratio": max(0.2, min(1.0, _f(thresholds_raw.get("risk_min_hits_ratio"), 0.5))),
    }

    return {
        "dsl_version": "1.0",
        "include_all": include_all,
        "include_any": include_any,
        "exclude_any": exclude_any,
        "risk_focus": risk_focus,
        "thresholds": thresholds,
    }


def _evaluate_term_rules(
    *,
    terms: List[str],
    category: str,
    evidence: List[Dict[str, Any]],
    pass_threshold: float,
    expected_hit: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, term in enumerate(terms, start=1):
        best_score = 0.0
        best_evidence_id = ""
        best_evidence_text = ""
        for item in evidence:
            score = _semantic_score(term, item)
            if score > best_score:
                best_score = score
                best_evidence_id = str(item.get("id", ""))
                best_evidence_text = str(item.get("text", ""))

        hit = best_score >= pass_threshold
        passed = hit if expected_hit else (not hit)
        rows.append(
            {
                "id": f"{category}:{index}",
                "category": category,
                "term": term,
                "threshold": round(pass_threshold, 3),
                "score": round(best_score, 4),
                "passed": bool(passed),
                "expected_hit": bool(expected_hit),
                "matched_evidence_ids": [best_evidence_id] if hit and best_evidence_id else [],
                "best_evidence": {
                    "id": best_evidence_id,
                    "snippet": best_evidence_text[:180],
                },
            }
        )
    return rows


def _llm_json_loads(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
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
            return {}
    obj = re.search(r"(\{[\s\S]*\})", raw)
    if obj:
        try:
            payload = json.loads(obj.group(1))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


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


def _llm_constraint_review(
    *,
    llm: Any,
    dsl: Dict[str, Any],
    rule_rows: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    draft_md: str,
) -> Dict[str, Any]:
    if not getattr(llm, "invoke", None):
        return {"used": False}

    rules_payload = [
        {
            "id": row.get("id"),
            "category": row.get("category"),
            "term": row.get("term"),
            "threshold": row.get("threshold"),
            "rule_score": row.get("score"),
            "rule_passed": row.get("passed"),
        }
        for row in rule_rows[:30]
    ]
    evidence_payload = [
        {
            "id": item.get("id"),
            "text": str(item.get("text", ""))[:160],
        }
        for item in evidence[:40]
    ]
    prompt = (
        "你是约束合规审查器。请基于约束DSL、规则引擎初判、证据索引和测试用例文本，"
        "输出约束合规评审JSON：\n"
        "{\n"
        '  "pass": true,\n'
        '  "score": 0,\n'
        '  "reasons": ["..."],\n'
        '  "rule_assertions": [\n'
        '    {"id":"include_all:1","passed":true,"score":0.0,"evidence_ids":["L12"]}\n'
        "  ]\n"
        "}\n"
        "规则：\n"
        "- 仅输出JSON\n"
        "- score 范围 0-100\n"
        "- evidence_ids 必须来自证据索引\n"
        "- 不要编造不存在的 id\n\n"
        f"DSL: {json.dumps(dsl, ensure_ascii=False)}\n"
        f"规则引擎初判: {json.dumps(rules_payload, ensure_ascii=False)}\n"
        f"证据索引: {json.dumps(evidence_payload, ensure_ascii=False)}\n"
        f"测试用例文本: {str(draft_md or '')[:5000]}"
    )
    try:
        response = llm.invoke(prompt)
        raw = _normalize_llm_content(response)
        parsed = _llm_json_loads(raw)
    except Exception:
        return {"used": False}

    if not parsed:
        return {"used": False}

    score = parsed.get("score", 0)
    try:
        score = float(score)
    except Exception:
        score = 0.0
    score = max(0.0, min(100.0, score))
    passed = bool(parsed.get("pass", False))

    reasons = parsed.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(x).strip() for x in reasons if str(x).strip()][:8]

    valid_ids = {str(item.get("id", "")) for item in evidence}
    assertions_raw = parsed.get("rule_assertions", [])
    if not isinstance(assertions_raw, list):
        assertions_raw = []
    assertions: List[Dict[str, Any]] = []
    for item in assertions_raw[:40]:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("id", "")).strip()
        if not rule_id:
            continue
        raw_evidence_ids = item.get("evidence_ids", [])
        if not isinstance(raw_evidence_ids, list):
            raw_evidence_ids = [raw_evidence_ids]
        evidence_ids = [
            str(eid).strip()
            for eid in raw_evidence_ids
            if str(eid).strip() and str(eid).strip() in valid_ids
        ][:6]
        try:
            item_score = float(item.get("score", 0.0))
        except Exception:
            item_score = 0.0
        assertions.append(
            {
                "id": rule_id,
                "passed": bool(item.get("passed", False)),
                "score": max(0.0, min(1.0, item_score)),
                "evidence_ids": evidence_ids,
            }
        )

    return {
        "used": True,
        "pass": passed,
        "score": round(score, 2),
        "reasons": reasons,
        "rule_assertions": assertions,
    }


def evaluate_constraint_compliance(
    *,
    draft_md: str,
    human_inputs: Dict[str, Any] | None,
    llm: Any = None,
) -> Dict[str, Any]:
    dsl = build_constraint_dsl(human_inputs)
    evidence = _build_evidence_index(draft_md)
    thresholds = dsl.get("thresholds", {})
    include_min_score = float(thresholds.get("include_min_score", 0.52))
    exclude_hit_score = float(thresholds.get("exclude_hit_score", 0.55))
    risk_min_score = float(thresholds.get("risk_min_score", 0.48))
    risk_min_hits_ratio = float(thresholds.get("risk_min_hits_ratio", 0.5))

    include_rows = _evaluate_term_rules(
        terms=dsl.get("include_all", []),
        category="include_all",
        evidence=evidence,
        pass_threshold=include_min_score,
        expected_hit=True,
    )
    include_any_rows = _evaluate_term_rules(
        terms=dsl.get("include_any", []),
        category="include_any",
        evidence=evidence,
        pass_threshold=include_min_score,
        expected_hit=True,
    )
    exclude_rows = _evaluate_term_rules(
        terms=dsl.get("exclude_any", []),
        category="exclude_any",
        evidence=evidence,
        pass_threshold=exclude_hit_score,
        expected_hit=False,
    )
    risk_rows = _evaluate_term_rules(
        terms=dsl.get("risk_focus", []),
        category="risk_focus",
        evidence=evidence,
        pass_threshold=risk_min_score,
        expected_hit=True,
    )

    include_failed = [r for r in include_rows if not bool(r.get("passed"))]
    include_any_passed = [r for r in include_any_rows if bool(r.get("passed"))]
    include_any_required = len(include_any_rows) > 0
    include_any_ok = (not include_any_required) or bool(include_any_passed)
    exclude_hits = [r for r in exclude_rows if not bool(r.get("passed"))]
    risk_hits = [r for r in risk_rows if bool(r.get("passed"))]
    risk_required_count = 0
    if risk_rows:
        risk_required_count = max(1, int(math.ceil(len(risk_rows) * risk_min_hits_ratio)))
    risk_ok = (not risk_rows) or (len(risk_hits) >= risk_required_count)

    include_score = 1.0
    if include_rows:
        include_score = sum(float(r.get("score", 0.0)) for r in include_rows) / len(include_rows)
    include_any_score = 1.0
    if include_any_rows:
        include_any_score = max(float(r.get("score", 0.0)) for r in include_any_rows)
    exclude_score = 1.0
    if exclude_rows:
        exclude_score = max(0.0, 1.0 - (len(exclude_hits) / float(len(exclude_rows))))
    risk_score = 1.0
    if risk_rows:
        risk_score = len(risk_hits) / float(len(risk_rows))

    rule_score = (
        (0.42 * include_score)
        + (0.18 * include_any_score)
        + (0.20 * exclude_score)
        + (0.20 * risk_score)
    ) * 100.0
    rule_score = round(max(0.0, min(100.0, rule_score)), 2)

    rule_pass = (
        not include_failed
        and include_any_ok
        and not exclude_hits
        and risk_ok
    )

    reasons: List[str] = []
    rewrite_instructions: List[str] = []
    if include_failed:
        reasons.append("必须覆盖项未命中")
        rewrite_instructions.append(
            "补齐以下必须覆盖项: "
            + "、".join([str(r.get("term", "")) for r in include_failed[:6]])
        )
    if include_any_required and not include_any_ok:
        reasons.append("至少命中一项覆盖要求未满足")
        rewrite_instructions.append(
            "至少补齐以下任一覆盖项: "
            + "、".join([str(r.get("term", "")) for r in include_any_rows[:6]])
        )
    if exclude_hits:
        reasons.append("命中排除项场景")
        rewrite_instructions.append(
            "删除或替换以下排除项相关内容: "
            + "、".join([str(r.get("term", "")) for r in exclude_hits[:6]])
        )
    if risk_rows and not risk_ok:
        reasons.append("风险关注覆盖不足")
        missing_risk = [r for r in risk_rows if not bool(r.get("passed"))]
        rewrite_instructions.append(
            "补充风险标签覆盖: " + "、".join([str(r.get("term", "")) for r in missing_risk[:6]])
        )

    llm_review = _llm_constraint_review(
        llm=llm,
        dsl=dsl,
        rule_rows=include_rows + include_any_rows + exclude_rows + risk_rows,
        evidence=evidence,
        draft_md=draft_md,
    )
    final_pass = rule_pass
    final_score = rule_score
    if llm_review.get("used"):
        llm_pass = bool(llm_review.get("pass", False))
        llm_score = llm_review.get("score")
        try:
            llm_score_value = float(llm_score)
        except Exception:
            llm_score_value = rule_score
        final_score = round((0.75 * rule_score) + (0.25 * llm_score_value), 2)
        if not llm_pass:
            final_pass = False
            llm_reasons = llm_review.get("reasons", [])
            if isinstance(llm_reasons, list):
                reasons.extend([str(x).strip() for x in llm_reasons if str(x).strip()][:4])
            if not llm_reasons:
                reasons.append("LLM 合规审查未通过")

    missing_items = {
        "include_all": [str(r.get("term", "")) for r in include_failed[:8]],
        "include_any": [] if include_any_ok else [str(r.get("term", "")) for r in include_any_rows[:8]],
        "risk_tags": [str(r.get("term", "")) for r in risk_rows if not bool(r.get("passed"))][:8],
        "must_not_cover_hits": [str(r.get("term", "")) for r in exclude_hits[:8]],
    }
    hit_items = {
        "include_all": [str(r.get("term", "")) for r in include_rows if bool(r.get("passed"))][:8],
        "include_any": [str(r.get("term", "")) for r in include_any_passed[:8]],
        "risk_tags": [str(r.get("term", "")) for r in risk_hits[:8]],
    }

    evidence_index = [
        {"id": str(item.get("id", "")), "text": str(item.get("text", ""))}
        for item in evidence[:40]
    ]
    constraints = (include_rows + include_any_rows + exclude_rows + risk_rows)[:40]

    return {
        "version": "v2",
        "pass": bool(final_pass),
        "score": final_score,
        "reasons": reasons[:8],
        "rewrite_instructions": rewrite_instructions[:8],
        "missing_items": missing_items,
        "hit_items": hit_items,
        "constraint_spec": dsl,
        "constraints": constraints,
        "rule_engine": {
            "pass": bool(rule_pass),
            "score": rule_score,
            "reasons": reasons[:8],
        },
        "llm_review": llm_review,
        "evidence_index": evidence_index,
    }
