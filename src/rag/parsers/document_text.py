from __future__ import annotations

import html
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import Optional, Tuple


def _clean_text(raw: str) -> str:
    text = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _run_command_extract_text(payload: bytes, suffix: str, command: list[str]) -> str:
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fp:
            fp.write(payload)
            tmp_path = fp.name

        cmd = [arg if arg != "{input}" else tmp_path for arg in command]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(err or f"命令执行失败: {' '.join(cmd)}")

        output = _clean_text(proc.stdout)
        if not output:
            raise RuntimeError("命令执行成功但未提取到文本。")
        return output
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _extract_docx_by_xml(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = zf.namelist()
        target_names = [
            name
            for name in names
            if name.startswith("word/")
            and name.endswith(".xml")
            and not name.startswith("word/_rels/")
        ]

        if not target_names:
            raise ValueError("docx 结构异常: 未找到 word/*.xml。")

        chunks = []
        for name in sorted(target_names):
            xml_text = zf.read(name).decode("utf-8", errors="ignore")
            xml_text = re.sub(r"</w:p>", "\n", xml_text, flags=re.IGNORECASE)
            xml_text = re.sub(r"</w:tr>", "\n", xml_text, flags=re.IGNORECASE)
            pieces = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml_text, flags=re.IGNORECASE | re.DOTALL)
            if pieces:
                chunks.append("\n".join(html.unescape(piece) for piece in pieces))

    result = _clean_text("\n\n".join(chunks))
    if not result:
        raise ValueError("docx 解析成功但文本为空。")
    return result


def _extract_pdf_by_library(payload: bytes) -> str:
    errors = []
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(payload))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        text = _clean_text(text)
        if text:
            return text
        errors.append("pypdf: 文本为空")
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(payload))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        text = _clean_text(text)
        if text:
            return text
        errors.append("PyPDF2: 文本为空")
    except Exception as exc:
        errors.append(f"PyPDF2: {exc}")

    raise RuntimeError("; ".join(errors) or "PDF 库解析失败。")


def _extract_pdf_by_command(payload: bytes) -> str:
    if shutil.which("pdftotext"):
        try:
            return _run_command_extract_text(
                payload=payload,
                suffix=".pdf",
                command=["pdftotext", "{input}", "-"],
            )
        except Exception:
            pass

    if shutil.which("textutil"):
        return _run_command_extract_text(
            payload=payload,
            suffix=".pdf",
            command=["textutil", "-convert", "txt", "-stdout", "{input}"],
        )

    raise RuntimeError("本机缺少可用 PDF 解析器（pypdf/PyPDF2/pdftotext/textutil）。")


def _extract_doc_by_command(payload: bytes, suffix: str) -> str:
    if shutil.which("textutil"):
        return _run_command_extract_text(
            payload=payload,
            suffix=suffix,
            command=["textutil", "-convert", "txt", "-stdout", "{input}"],
        )

    if suffix == ".doc" and shutil.which("antiword"):
        return _run_command_extract_text(
            payload=payload,
            suffix=suffix,
            command=["antiword", "{input}"],
        )

    raise RuntimeError("本机缺少可用 DOC 解析器（textutil/antiword）。")


def extract_text_from_document_bytes(payload: bytes, suffix: str) -> Tuple[str, Optional[str]]:
    """
    Extract text from .pdf/.doc/.docx bytes.
    Returns (text, warning). warning is None when extraction is clean.
    """
    ext = str(suffix or "").lower().strip()
    if ext not in {".pdf", ".doc", ".docx"}:
        return "", f"暂不支持该文档后缀: {ext}"

    if not payload:
        return "", "文档内容为空。"

    if ext == ".docx":
        try:
            return _extract_docx_by_xml(payload), None
        except Exception as xml_exc:
            try:
                text = _extract_doc_by_command(payload, ".docx")
                return text, f"docx 已降级命令解析: {xml_exc}"
            except Exception as cmd_exc:
                return "", f"docx 解析失败: {xml_exc}; 命令解析失败: {cmd_exc}"

    if ext == ".doc":
        try:
            text = _extract_doc_by_command(payload, ".doc")
            return text, None
        except Exception as exc:
            return "", f"doc 解析失败: {exc}"

    # .pdf
    try:
        return _extract_pdf_by_library(payload), None
    except Exception as lib_exc:
        try:
            text = _extract_pdf_by_command(payload)
            return text, f"PDF 已降级命令解析: {lib_exc}"
        except Exception as cmd_exc:
            return "", f"PDF 解析失败: {lib_exc}; 命令解析失败: {cmd_exc}"
