from __future__ import annotations

import io
from typing import Optional, Tuple


def extract_text_from_image_bytes(payload: bytes) -> Tuple[str, Optional[str]]:
    """
    Try OCR from image bytes.
    Returns (text, warning). warning is None when extraction is clean.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return "", "OCR 依赖缺失: 未安装 Pillow，已跳过图片文本提取。"

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        return "", f"图片解析失败: {exc}"

    try:
        import pytesseract  # type: ignore
    except Exception:
        return "", "OCR 依赖缺失: 未安装 pytesseract，已跳过图片文本提取。"

    # 优先中英文混合识别；如果本机无中文语言包，回退英文。
    for lang in ("chi_sim+eng", "eng"):
        try:
            text = pytesseract.image_to_string(image, lang=lang)
            text = text.strip()
            if text:
                return text, None
        except Exception:
            continue

    return "", "OCR 未识别到有效文本，请补充文字描述以提升检索质量。"
