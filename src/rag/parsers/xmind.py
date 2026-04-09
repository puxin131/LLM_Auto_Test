from __future__ import annotations

import io
import json
import re
import zipfile
from typing import List


def _walk_topic(topic: dict, path: List[str], output: List[str]) -> None:
    title = (topic or {}).get("title", "").strip()
    current_path = path[:]
    if title:
        current_path.append(title)
        output.append(" > ".join(current_path))

    children = (topic or {}).get("children", {})
    attached = children.get("attached") or []
    for child in attached:
        _walk_topic(child, current_path, output)


def parse_xmind_bytes(payload: bytes) -> str:
    """Parse .xmind payload into line-based hierarchical text."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())

        if "content.json" in names:
            raw = zf.read("content.json")
            data = json.loads(raw.decode("utf-8", errors="ignore"))
            lines: List[str] = []
            for sheet in data:
                root_topic = (sheet or {}).get("rootTopic") or {}
                _walk_topic(root_topic, [], lines)
            return "\n".join(lines).strip()

        if "content.xml" in names:
            raw = zf.read("content.xml").decode("utf-8", errors="ignore")
            titles = re.findall(r"<title>(.*?)</title>", raw, flags=re.DOTALL)
            cleaned = [re.sub(r"\s+", " ", t).strip() for t in titles]
            cleaned = [c for c in cleaned if c]
            return "\n".join(cleaned).strip()

    raise ValueError("xmind 解析失败: 未找到 content.json 或 content.xml")
