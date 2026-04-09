from __future__ import annotations

import os
import re
import time
import json
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Feishu/Lark OpenAPI defaults.
_DEFAULT_BASE_URL = "https://open.feishu.cn"
_TOKEN_CACHE: Dict[str, Any] = {"token": "", "expire_at": 0.0}


def _normalize_base_url() -> str:
    return str(
        os.getenv("FEISHU_BASE_URL")
        or os.getenv("LARK_BASE_URL")
        or _DEFAULT_BASE_URL
    ).rstrip("/")


def _request_timeout() -> float:
    raw = str(os.getenv("FEISHU_REQUEST_TIMEOUT", "20")).strip()
    try:
        return max(5.0, float(raw))
    except Exception:
        return 20.0


def _extract_board_token(reference: str) -> str:
    ref = (reference or "").strip()
    if not ref:
        raise ValueError("空的飞书白板引用。")

    # Direct token input.
    if "http" not in ref and "/" not in ref:
        return ref

    parsed = urlparse(ref)
    query = parse_qs(parsed.query)
    for key in ("token", "board_token", "whiteboard_token", "id"):
        values = query.get(key, [])
        if values and values[0].strip():
            return values[0].strip()

    # URL path extraction: /board/<token> or trailing token-like segment.
    path = parsed.path or ""
    match = re.search(r"/board/([A-Za-z0-9_-]+)", path)
    if match:
        return match.group(1)

    candidates = [seg.strip() for seg in path.split("/") if seg.strip()]
    if candidates:
        tail = candidates[-1]
        # Basic token shape check (avoid super short words).
        if re.fullmatch(r"[A-Za-z0-9_-]{8,}", tail):
            return tail

    raise ValueError(f"无法从飞书白板引用中解析 token: {reference}")


def _extract_doc_token(reference: str) -> str:
    ref = (reference or "").strip()
    if not ref:
        raise ValueError("空的飞书文档引用。")

    if "http" not in ref and "/" not in ref:
        return ref

    parsed = urlparse(ref)
    query = parse_qs(parsed.query)
    for key in ("token", "doc_token", "document_id", "id"):
        values = query.get(key, [])
        if values and values[0].strip():
            return values[0].strip()

    path = parsed.path or ""
    patterns = [
        r"/docx/([A-Za-z0-9_-]{8,})",
        r"/docs?/([A-Za-z0-9_-]{8,})",
        r"/document/([A-Za-z0-9_-]{8,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return match.group(1)

    candidates = [seg.strip() for seg in path.split("/") if seg.strip()]
    if candidates:
        tail = candidates[-1]
        if re.fullmatch(r"[A-Za-z0-9_-]{8,}", tail):
            return tail

    raise ValueError(f"无法从飞书文档引用中解析 token: {reference}")


def _ensure_app_access_token() -> str:
    now = time.time()
    cached_token = str(_TOKEN_CACHE.get("token") or "")
    expire_at = float(_TOKEN_CACHE.get("expire_at") or 0.0)
    if cached_token and expire_at - now > 30:
        return cached_token

    app_id = str(os.getenv("FEISHU_APP_ID") or os.getenv("LARK_APP_ID") or "").strip()
    app_secret = str(
        os.getenv("FEISHU_APP_SECRET") or os.getenv("LARK_APP_SECRET") or ""
    ).strip()
    if not app_id or not app_secret:
        raise RuntimeError(
            "飞书鉴权缺失: 请配置 FEISHU_APP_ID / FEISHU_APP_SECRET。"
        )

    url = f"{_normalize_base_url()}/open-apis/auth/v3/app_access_token/internal"
    payload = {"app_id": app_id, "app_secret": app_secret}
    body = _http_request_json(
        method="POST",
        url=url,
        headers={"Content-Type": "application/json"},
        json_payload=payload,
    )

    if int(body.get("code", -1)) != 0:
        raise RuntimeError(
            f"飞书鉴权失败 code={body.get('code')} msg={body.get('msg')}"
        )

    token = str(body.get("app_access_token") or body.get("tenant_access_token") or "")
    if not token:
        raise RuntimeError("飞书鉴权失败: 未返回 app_access_token。")

    # expire field may be expire(seconds) or absolute.
    expire_seconds = int(body.get("expire", 3600) or 3600)
    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["expire_at"] = now + max(300, expire_seconds)
    return token


def _http_request_json(
    *,
    method: str,
    url: str,
    headers: Dict[str, str] | None = None,
    json_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    body_bytes = None
    req_headers = dict(headers or {})
    if json_payload is not None:
        body_bytes = json.dumps(json_payload).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    request = Request(url=url, data=body_bytes, method=method.upper(), headers=req_headers)
    timeout = _request_timeout()
    try:
        with urlopen(request, timeout=timeout) as resp:  # nosec B310
            payload = resp.read().decode("utf-8", errors="ignore")
            return json.loads(payload)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"网络连接失败: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"请求异常: {exc}") from exc


def _api_get(path: str, token: str, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{_normalize_base_url()}{path}"
    if params:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        if query:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{query}"
    headers = {"Authorization": f"Bearer {token}"}
    body = _http_request_json(method="GET", url=url, headers=headers)

    if int(body.get("code", -1)) != 0:
        raise RuntimeError(
            f"飞书接口失败 path={path} code={body.get('code')} msg={body.get('msg')}"
        )
    return body


def _pull_whiteboard_payload(board_token: str, token: str) -> Tuple[Dict[str, Any], str]:
    """
    Try candidate endpoints for whiteboard payload.
    NOTE:
    - Feishu Board endpoints may vary across tenant versions.
    - We keep a small fallback matrix and return the first successful response.
    """
    attempts: List[Tuple[str, Dict[str, Any] | None]] = [
        (f"/open-apis/board/v1/whiteboards/{board_token}/nodes", {"page_size": 500}),
        (f"/open-apis/board/v1/whiteboards/{board_token}", None),
    ]

    errors: List[str] = []
    for path, params in attempts:
        try:
            payload = _api_get(path, token, params=params)
            return payload, path
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    raise RuntimeError(" | ".join(errors) if errors else "飞书白板拉取失败。")


def _pull_doc_payload(doc_token: str, token: str) -> Tuple[Dict[str, Any], str]:
    attempts = [
        (f"/open-apis/docx/v1/documents/{doc_token}/raw_content", "raw_content"),
        (f"/open-apis/docx/v1/documents/{doc_token}", "document"),
        (f"/open-apis/docx/v1/documents/{doc_token}/blocks", "blocks"),
    ]
    errors: List[str] = []

    for path, mode in attempts:
        try:
            if mode == "blocks":
                page_token = ""
                blocks: List[Dict[str, Any]] = []
                while True:
                    params: Dict[str, Any] = {"page_size": 200}
                    if page_token:
                        params["page_token"] = page_token
                    payload = _api_get(path, token, params=params)
                    data = payload.get("data", {}) or {}
                    items = data.get("items") or data.get("blocks") or []
                    if isinstance(items, list):
                        blocks.extend([i for i in items if isinstance(i, dict)])
                    has_more = bool(data.get("has_more"))
                    page_token = str(data.get("page_token") or data.get("next_page_token") or "").strip()
                    if not has_more or not page_token:
                        break
                if blocks:
                    return {"data": {"items": blocks}}, path
                raise RuntimeError("blocks 接口返回为空。")

            payload = _api_get(path, token)
            data = payload.get("data", {}) or {}
            if mode == "raw_content":
                raw_content = str(data.get("content", "")).strip()
                if raw_content:
                    return {"data": {"raw_content": raw_content}}, path
                raise RuntimeError("raw_content 为空。")
            return payload, path
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    raise RuntimeError(" | ".join(errors) if errors else "飞书文档拉取失败。")


def _collect_texts_from_payload(payload: Any) -> List[str]:
    texts: List[str] = []
    seen = set()

    text_like_keys = {
        "text",
        "title",
        "name",
        "content",
        "plain_text",
        "label",
        "value",
        "description",
        "summary",
    }

    def add(text: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return
        if len(cleaned) < 2:
            return
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return
        if cleaned not in seen:
            seen.add(cleaned)
            texts.append(cleaned)

    def walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = str(key).strip().lower()
                if isinstance(value, str) and key_lower in text_like_keys:
                    add(value)
                walk(value, key_lower)
            return

        if isinstance(node, list):
            for item in node:
                walk(item, parent_key)
            return

        if isinstance(node, str):
            # Only include free text when parent key hints textual value.
            if parent_key in text_like_keys:
                add(node)

    walk(payload)
    return texts


def fetch_feishu_doc_text(reference: str) -> str:
    doc_token = _extract_doc_token(reference)
    access_token = _ensure_app_access_token()
    payload, endpoint = _pull_doc_payload(doc_token, access_token)

    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    raw_content = ""
    if isinstance(data, dict):
        raw_content = str(data.get("raw_content") or data.get("content") or "").strip()

    texts: List[str] = []
    if raw_content:
        compact = re.sub(r"\s+", " ", raw_content).strip()
        if compact:
            texts.append(compact)
    texts.extend(_collect_texts_from_payload(data))

    if not texts:
        raise RuntimeError(
            "飞书文档接口调用成功，但未提取到可用文本。"
            "建议检查文档权限、文档类型，或切换为导出文件入库。"
        )

    preview = "\n".join(f"- {line}" for line in texts[:300])
    return (
        f"Feishu Document Token: {doc_token}\n"
        f"Fetched By Endpoint: {endpoint}\n"
        "Extracted Text:\n"
        f"{preview}"
    )


def fetch_feishu_board_text(reference: str) -> str:
    """
    Fetch Feishu whiteboard content by URL or token.
    Required env:
    - FEISHU_APP_ID
    - FEISHU_APP_SECRET
    Optional env:
    - FEISHU_BASE_URL (default: https://open.feishu.cn)
    - FEISHU_REQUEST_TIMEOUT (seconds)
    """
    board_token = _extract_board_token(reference)
    access_token = _ensure_app_access_token()
    payload, endpoint = _pull_whiteboard_payload(board_token, access_token)

    texts = _collect_texts_from_payload(payload.get("data", payload))
    if not texts:
        raise RuntimeError(
            "飞书白板接口调用成功，但未提取到可用文本。"
            "建议检查白板权限、内容类型，或补充白板导出解析。"
        )

    preview = "\n".join(f"- {line}" for line in texts[:300])
    return (
        f"Feishu Whiteboard Token: {board_token}\n"
        f"Fetched By Endpoint: {endpoint}\n"
        "Extracted Text:\n"
        f"{preview}"
    )
