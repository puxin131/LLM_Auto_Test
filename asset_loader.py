from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_env_path)


# 可通过环境变量 IMAGE_ASSET_SOUL_PROMPT 覆盖为你的“灵魂 Prompt”
DEFAULT_SOUL_PROMPT = """
你是一名资深测试架构分析助手。请将输入的 UI 截图、流程图或原型图转写为结构化 Markdown 文档。
输出要求：
1. 仅输出 Markdown 正文，不要解释过程，不要输出 JSON，不要输出代码块围栏。
2. 先给出「页面/图示概览」，再给出「关键元素清单」，最后给出「可测试点建议」。
3. 对无法识别的内容用“[待确认]”标注，不要编造。
4. 保留可用于检索的关键词（按钮名、字段名、流程节点、状态、错误提示等）。
5. 语言使用中文，表达简洁、准确、可落地。
""".strip()

RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class AssetLoaderError(RuntimeError):
    """Base exception for asset_loader."""


class VisionConfigError(AssetLoaderError):
    """Invalid or missing model configuration."""


class VisionRequestError(AssetLoaderError):
    """Model request failed after retries."""


class HttpStatusError(AssetLoaderError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


def process_image_to_text(file_path: str) -> str:
    result = process_image_to_text_with_meta(file_path)
    return result["markdown"]


def process_image_to_text_with_meta(file_path: str) -> Dict[str, str]:
    """
    Convert an image asset (PNG/JPG/...) into normalized Markdown text.
    Processing order:
    1) validate input
    2) model vision parse when config is available
    3) fallback local OCR when model unavailable/fails
    4) normalize markdown and persist cache
    Returns: {"markdown": "...", "engine": "model:openai:gpt-4o-mini|local_ocr", "warning": "..."}
    """
    image_path = Path(file_path).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXT:
        raise ValueError(
            f"不支持的图片类型: {suffix}。支持: {', '.join(sorted(SUPPORTED_IMAGE_EXT))}"
        )

    image_bytes = image_path.read_bytes()
    if not image_bytes:
        raise ValueError(f"图片内容为空: {image_path}")

    soul_prompt = os.getenv("IMAGE_ASSET_SOUL_PROMPT", DEFAULT_SOUL_PROMPT).strip()
    if not soul_prompt:
        raise VisionConfigError("系统指令为空，请配置 IMAGE_ASSET_SOUL_PROMPT。")

    enable_ocr_fallback = _read_bool_env("ASSET_LOADER_ENABLE_OCR_FALLBACK", default=True)

    try:
        provider = _resolve_provider()
        model_name = _resolve_model(provider)
        return _process_with_model(
            image_path=image_path,
            image_bytes=image_bytes,
            provider=provider,
            model_name=model_name,
            soul_prompt=soul_prompt,
        )
    except (AssetLoaderError, TimeoutError, ConnectionError, OSError) as exc:
        if not enable_ocr_fallback:
            raise
        return _process_with_local_ocr(
            image_bytes=image_bytes,
            fallback_reason=str(exc),
        )


def _process_with_model(
    *,
    image_path: Path,
    image_bytes: bytes,
    provider: str,
    model_name: str,
    soul_prompt: str,
) -> Dict[str, str]:
    cache_dir = _resolve_cache_dir()
    cache_key = _build_cache_key(
        image_bytes=image_bytes,
        provider=provider,
        model=model_name,
        soul_prompt=soul_prompt,
    )
    cache_file = cache_dir / f"{cache_key}.md"
    if cache_file.exists():
        cached = cache_file.read_text(encoding="utf-8").strip()
        if cached:
            return {
                "markdown": cached,
                "engine": f"model:{provider}:{model_name}",
                "warning": "",
            }

    mime_type = _guess_mime_type(image_path)
    max_retries = _read_int_env("ASSET_LOADER_MAX_RETRIES", 4, minimum=1, maximum=10)
    timeout_seconds = _read_int_env("ASSET_LOADER_TIMEOUT_SECONDS", 45, minimum=5, maximum=300)
    backoff_seconds = _read_float_env("ASSET_LOADER_BACKOFF_SECONDS", 1.0, minimum=0.0, maximum=30.0)
    jitter_seconds = _read_float_env("ASSET_LOADER_JITTER_SECONDS", 0.2, minimum=0.0, maximum=5.0)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_output = _invoke_vision_api(
                provider=provider,
                model=model_name,
                image_bytes=image_bytes,
                mime_type=mime_type,
                soul_prompt=soul_prompt,
                timeout_seconds=timeout_seconds,
            )
            markdown = _normalize_markdown(raw_output)
            if not markdown:
                raise VisionRequestError("模型返回为空，无法生成 Markdown。")

            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(markdown, encoding="utf-8")
            return {
                "markdown": markdown,
                "engine": f"model:{provider}:{model_name}",
                "warning": "",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            retryable = _is_retryable_error(exc)
            if attempt >= max_retries or not retryable:
                break
            sleep_seconds = backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, jitter_seconds)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    raise VisionRequestError(
        f"图片转 Markdown 失败（provider={provider}, model={model_name}, retries={max_retries}）: {last_error}"
    ) from last_error


def _process_with_local_ocr(
    *,
    image_bytes: bytes,
    fallback_reason: str,
) -> Dict[str, str]:
    cache_dir = _resolve_cache_dir()
    cache_key = _build_cache_key(
        image_bytes=image_bytes,
        provider="local_ocr",
        model="tesseract",
        soul_prompt="",
    )
    cache_file = cache_dir / f"{cache_key}.md"
    if cache_file.exists():
        cached = cache_file.read_text(encoding="utf-8").strip()
        if cached:
            return {
                "markdown": cached,
                "engine": "local_ocr",
                "warning": "",
            }

    ocr_text, ocr_warning = _extract_text_with_local_ocr(image_bytes)
    lines = ["## 图片资产解析（本地 OCR）"]
    if ocr_text.strip():
        lines.append("")
        lines.append(ocr_text.strip())
    else:
        lines.append("")
        lines.append("- 结果: [待确认]")
        lines.append(f"- 说明: {ocr_warning or 'OCR 未识别到有效文本。'}")

    if fallback_reason:
        lines.append("")
        lines.append(f"- 模型通道未使用原因: {fallback_reason}")

    markdown = _normalize_markdown("\n".join(lines))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(markdown, encoding="utf-8")
    return {
        "markdown": markdown,
        "engine": "local_ocr",
        "warning": ocr_warning or "",
    }


def _extract_text_with_local_ocr(image_bytes: bytes) -> tuple[str, Optional[str]]:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return "", "本地 OCR 依赖缺失: 未安装 Pillow。"

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return "", f"图片解析失败: {exc}"

    try:
        import pytesseract  # type: ignore
    except Exception:
        return "", "本地 OCR 依赖缺失: 未安装 pytesseract。"

    for lang in ("chi_sim+eng", "eng"):
        try:
            text = pytesseract.image_to_string(image, lang=lang)
            text = text.strip()
            if text:
                return text, None
        except Exception:
            continue

    return "", "本地 OCR 未识别到有效文本。"


def _resolve_provider() -> str:
    provider = (os.getenv("VISION_PROVIDER") or "").strip().lower()
    if provider in {"openai", "anthropic"}:
        return provider

    if os.getenv("VISION_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"

    raise VisionConfigError(
        "未检测到可用视觉模型配置。请设置 VISION_PROVIDER=openai|anthropic，"
        "并提供对应 API Key（OPENAI_API_KEY 或 ANTHROPIC_API_KEY）。"
    )


def _resolve_model(provider: str) -> str:
    if provider == "openai":
        return (os.getenv("OPENAI_VISION_MODEL") or "gpt-4o-mini").strip()
    if provider == "anthropic":
        return (os.getenv("ANTHROPIC_VISION_MODEL") or "claude-3-5-sonnet-20241022").strip()
    raise VisionConfigError(f"未知 provider: {provider}")


def _resolve_cache_dir() -> Path:
    raw = (os.getenv("ASSET_LOADER_CACHE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent / "data" / "image_md_cache"


def _build_cache_key(image_bytes: bytes, provider: str, model: str, soul_prompt: str) -> str:
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(provider.encode("utf-8"))
    h.update(model.encode("utf-8"))
    h.update(soul_prompt.encode("utf-8"))
    return h.hexdigest()


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime and mime.startswith("image/"):
        return mime
    return "image/png"


def _read_int_env(key: str, default: int, minimum: int, maximum: int) -> int:
    raw = (os.getenv(key) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def _read_float_env(key: str, default: float, minimum: float, maximum: float) -> float:
    raw = (os.getenv(key) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def _read_bool_env(key: str, default: bool = True) -> bool:
    raw = (os.getenv(key) or "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _invoke_vision_api(
    *,
    provider: str,
    model: str,
    image_bytes: bytes,
    mime_type: str,
    soul_prompt: str,
    timeout_seconds: int,
) -> str:
    if provider == "openai":
        return _invoke_openai_vision(
            model=model,
            image_bytes=image_bytes,
            mime_type=mime_type,
            soul_prompt=soul_prompt,
            timeout_seconds=timeout_seconds,
        )
    if provider == "anthropic":
        return _invoke_anthropic_vision(
            model=model,
            image_bytes=image_bytes,
            mime_type=mime_type,
            soul_prompt=soul_prompt,
            timeout_seconds=timeout_seconds,
        )
    raise VisionConfigError(f"未知 provider: {provider}")


def _invoke_openai_vision(
    *,
    model: str,
    image_bytes: bytes,
    mime_type: str,
    soul_prompt: str,
    timeout_seconds: int,
) -> str:
    api_key = (os.getenv("VISION_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise VisionConfigError("缺少 VISION_OPENAI_API_KEY（或 OPENAI_API_KEY）。")

    base_url = (
        os.getenv("VISION_OPENAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    base_url = base_url.rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": soul_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请将该图片解析为结构化 Markdown。"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                ],
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = _http_post_json(url=endpoint, headers=headers, payload=payload, timeout_seconds=timeout_seconds)
    return _extract_openai_text(data)


def _invoke_anthropic_vision(
    *,
    model: str,
    image_bytes: bytes,
    mime_type: str,
    soul_prompt: str,
    timeout_seconds: int,
) -> str:
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise VisionConfigError("缺少 ANTHROPIC_API_KEY。")

    endpoint = "https://api.anthropic.com/v1/messages"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": 0,
        "system": soul_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": "请将该图片解析为结构化 Markdown。"},
                ],
            }
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = _http_post_json(url=endpoint, headers=headers, payload=payload, timeout_seconds=timeout_seconds)
    return _extract_anthropic_text(data)


def _http_post_json(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AssetLoaderError(f"接口返回非 JSON: {raw[:500]}") from exc
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise HttpStatusError(status_code=int(exc.code), body=body) from exc
    except error.URLError as exc:
        raise VisionRequestError(f"网络请求失败: {exc}") from exc


def _extract_openai_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VisionRequestError(f"OpenAI 返回缺少 choices: {data}")

    message = (choices[0] or {}).get("message", {})
    content = message.get("content")
    return _coerce_content_to_text(content)


def _extract_anthropic_text(data: Dict[str, Any]) -> str:
    content = data.get("content")
    if not isinstance(content, list) or not content:
        raise VisionRequestError(f"Anthropic 返回缺少 content: {data}")

    texts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)

    merged = "\n".join(t.strip() for t in texts if t and t.strip()).strip()
    if not merged:
        raise VisionRequestError(f"Anthropic 返回 text 为空: {data}")
    return merged


def _coerce_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    raise VisionRequestError(f"模型 content 格式无法解析: {type(content)}")


def _normalize_markdown(text: str) -> str:
    content = (text or "").strip()
    if not content:
        return ""

    # 去掉常见 markdown fenced code 包裹
    lowered = content.lower()
    if lowered.startswith("```markdown"):
        content = content[len("```markdown") :].strip()
    elif lowered.startswith("```md"):
        content = content[len("```md") :].strip()
    elif lowered.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    # 统一换行，去掉结尾多余空白
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    return content


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, HttpStatusError):
        return exc.status_code in RETRYABLE_HTTP_STATUS
    if isinstance(exc, VisionConfigError):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # urllib error wrapped as VisionRequestError by _http_post_json
    message = str(exc).lower()
    transient_keys = ["timeout", "timed out", "connection reset", "temporary", "503", "429"]
    return any(k in message for k in transient_keys)
