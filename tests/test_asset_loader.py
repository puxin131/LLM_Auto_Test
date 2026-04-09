from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import asset_loader


# 1x1 PNG
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a4mQAAAAASUVORK5CYII="


class TestAssetLoader(unittest.TestCase):
    def _write_png(self, tmpdir: str, name: str = "sample.png") -> Path:
        path = Path(tmpdir) / name
        path.write_bytes(base64.b64decode(PNG_BASE64))
        return path

    def test_process_image_to_text_png_retry_then_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_png(tmpdir)
            cache_dir = Path(tmpdir) / "cache"

            call_count = {"n": 0}

            def fake_invoke(**_: object) -> str:
                call_count["n"] += 1
                if call_count["n"] < 2:
                    raise TimeoutError("transient timeout")
                return "```markdown\n# 页面概览\n- 按钮：提交\n```"

            env = {
                "VISION_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "ASSET_LOADER_CACHE_DIR": str(cache_dir),
                "ASSET_LOADER_MAX_RETRIES": "3",
                "ASSET_LOADER_BACKOFF_SECONDS": "0",
                "ASSET_LOADER_JITTER_SECONDS": "0",
            }
            with patch.dict(os.environ, env, clear=False):
                with patch("asset_loader._invoke_vision_api", side_effect=fake_invoke):
                    result = asset_loader.process_image_to_text(str(image_path))

            self.assertEqual(result, "# 页面概览\n- 按钮：提交")
            self.assertEqual(call_count["n"], 2)
            self.assertTrue(cache_dir.exists())
            self.assertEqual(len(list(cache_dir.glob("*.md"))), 1)

    def test_process_image_to_text_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_png(tmpdir)
            cache_dir = Path(tmpdir) / "cache"

            env = {
                "VISION_PROVIDER": "openai",
                "OPENAI_API_KEY": "test-key",
                "ASSET_LOADER_CACHE_DIR": str(cache_dir),
                "ASSET_LOADER_MAX_RETRIES": "2",
                "ASSET_LOADER_BACKOFF_SECONDS": "0",
                "ASSET_LOADER_JITTER_SECONDS": "0",
            }

            with patch.dict(os.environ, env, clear=False):
                with patch("asset_loader._invoke_vision_api", return_value="## A\n- B") as mocked:
                    first = asset_loader.process_image_to_text(str(image_path))
                    second = asset_loader.process_image_to_text(str(image_path))

            self.assertEqual(first, "## A\n- B")
            self.assertEqual(second, "## A\n- B")
            self.assertEqual(mocked.call_count, 1)

    def test_process_image_to_text_without_provider_config_fallback_local_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_png(tmpdir)
            cache_dir = Path(tmpdir) / "cache"

            with patch.dict(
                os.environ,
                {
                    "VISION_PROVIDER": "",
                    "OPENAI_API_KEY": "",
                    "ANTHROPIC_API_KEY": "",
                    "ASSET_LOADER_CACHE_DIR": str(cache_dir),
                    "ASSET_LOADER_ENABLE_OCR_FALLBACK": "1",
                },
                clear=False,
            ):
                with patch(
                    "asset_loader._extract_text_with_local_ocr",
                    return_value=("OCR 文本内容", None),
                ):
                    result = asset_loader.process_image_to_text_with_meta(str(image_path))

            self.assertEqual(result["engine"], "local_ocr")
            self.assertIn("OCR 文本内容", result["markdown"])

    def test_process_image_to_text_model_fail_then_fallback_local_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_png(tmpdir)
            cache_dir = Path(tmpdir) / "cache"

            with patch.dict(
                os.environ,
                {
                    "VISION_PROVIDER": "openai",
                    "OPENAI_API_KEY": "test-key",
                    "ASSET_LOADER_CACHE_DIR": str(cache_dir),
                    "ASSET_LOADER_MAX_RETRIES": "1",
                    "ASSET_LOADER_ENABLE_OCR_FALLBACK": "1",
                },
                clear=False,
            ):
                with patch(
                    "asset_loader._invoke_vision_api",
                    side_effect=asset_loader.VisionRequestError("upstream failed"),
                ):
                    with patch(
                        "asset_loader._extract_text_with_local_ocr",
                        return_value=("本地 OCR 兜底文本", None),
                    ):
                        result = asset_loader.process_image_to_text_with_meta(str(image_path))

            self.assertEqual(result["engine"], "local_ocr")
            self.assertIn("本地 OCR 兜底文本", result["markdown"])

    def test_process_image_to_text_with_vision_openai_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = self._write_png(tmpdir)
            cache_dir = Path(tmpdir) / "cache"

            with patch.dict(
                os.environ,
                {
                    "VISION_PROVIDER": "openai",
                    "VISION_OPENAI_API_KEY": "vision-key",
                    "VISION_OPENAI_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "OPENAI_API_KEY": "",
                    "ASSET_LOADER_CACHE_DIR": str(cache_dir),
                    "ASSET_LOADER_ENABLE_OCR_FALLBACK": "0",
                },
                clear=False,
            ):
                with patch("asset_loader._invoke_vision_api", return_value="## 视觉解析\n- 成功"):
                    result = asset_loader.process_image_to_text_with_meta(str(image_path))

            self.assertTrue(result["engine"].startswith("model:openai:"))
            self.assertIn("视觉解析", result["markdown"])

    def test_process_image_to_text_invalid_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "bad.txt"
            file_path.write_text("not image", encoding="utf-8")
            with self.assertRaises(ValueError):
                asset_loader.process_image_to_text(str(file_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
