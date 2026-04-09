from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen


def _figma_api_base() -> str:
    return str(os.getenv("FIGMA_API_BASE_URL") or "https://api.figma.com").rstrip("/")


def _figma_timeout() -> float:
    raw = str(os.getenv("FIGMA_REQUEST_TIMEOUT", "20")).strip()
    try:
        return max(5.0, float(raw))
    except Exception:
        return 20.0


def _figma_token() -> str:
    token = str(os.getenv("FIGMA_ACCESS_TOKEN") or os.getenv("FIGMA_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("缺少 FIGMA_ACCESS_TOKEN（或 FIGMA_TOKEN）。")
    return token


def _extract_figma_ref(reference: str) -> Tuple[str, str]:
    ref = (reference or "").strip()
    if not ref:
        raise ValueError("空的 Figma 引用。")

    if "http" not in ref and "/" not in ref:
        return ref, ""

    parsed = urlparse(ref)
    query = parse_qs(parsed.query)
    node_id = ""
    node_values = query.get("node-id", [])
    if node_values and node_values[0].strip():
        node_id = node_values[0].strip()

    match = re.search(r"/file/([A-Za-z0-9]+)", parsed.path or "")
    if match:
        return match.group(1), node_id

    candidates = [seg.strip() for seg in (parsed.path or "").split("/") if seg.strip()]
    if candidates:
        tail = candidates[-1]
        if re.fullmatch(r"[A-Za-z0-9]{10,}", tail):
            return tail, node_id

    raise ValueError(f"无法解析 Figma file key: {reference}")


def _figma_get(path: str, token: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{_figma_api_base()}{path}"
    if params:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        if query:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{query}"

    req = Request(
        url=url,
        headers={"X-Figma-Token": token, "Content-Type": "application/json"},
        method="GET",
    )
    try:
        with urlopen(req, timeout=_figma_timeout()) as resp:  # nosec B310
            payload = resp.read().decode("utf-8", errors="ignore")
            return json.loads(payload)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Figma HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Figma 网络异常: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Figma 请求异常: {exc}") from exc


def _collect_figma_texts(payload: Any) -> List[str]:
    texts: List[str] = []
    seen = set()

    def add(text: str) -> None:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(cleaned) < 2:
            return
        if cleaned not in seen:
            seen.add(cleaned)
            texts.append(cleaned)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node_type = str(node.get("type", "")).upper()
            if node_type == "TEXT":
                add(node.get("characters", ""))
            add(node.get("name", ""))
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return texts


def fetch_figma_text(reference: str) -> str:
    file_key, node_id = _extract_figma_ref(reference)
    token = _figma_token()

    if node_id:
        payload = _figma_get(f"/v1/files/{file_key}/nodes", token, params={"ids": node_id})
        source_label = f"nodes:{node_id}"
    else:
        payload = _figma_get(f"/v1/files/{file_key}", token)
        source_label = "file"

    texts = _collect_figma_texts(payload)
    if not texts:
        raise RuntimeError(
            "Figma 接口调用成功，但未提取到文本。请检查链接权限、文件内容或节点范围。"
        )

    preview = "\n".join(f"- {line}" for line in texts[:300])
    return (
        f"Figma File Key: {file_key}\n"
        f"Fetched Scope: {source_label}\n"
        "Extracted Text:\n"
        f"{preview}"
    )
