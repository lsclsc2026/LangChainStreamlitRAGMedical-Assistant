import base64
import json
import os
from typing import Any

import requests

from utils.config_handler import psych_conf


class SparkOcrClient:
    """星火 HTTP OCR 客户端（基于兼容 OpenAI 的 chat/completions）。"""

    def __init__(self):
        self.url = psych_conf.get("spark_ocr_url", "").strip()
        self.api_key = (os.getenv("SPARK_OCR_API_KEY") or psych_conf.get("spark_api_key", "")).strip()
        self.model = psych_conf.get("spark_model", "spark-x")
        self.thinking_type = psych_conf.get("spark_thinking_type", "disabled")
        self.temperature = float(psych_conf.get("spark_temperature", 0.1))
        self.top_p = float(psych_conf.get("spark_top_p", 0.9))
        self.max_tokens = int(psych_conf.get("spark_max_tokens", 4096))
        self.timeout_seconds = int(psych_conf.get("spark_timeout_seconds", 120))

    def is_configured(self) -> bool:
        return bool(self.url and self.api_key)

    @staticmethod
    def _extract_text(resp_json: dict[str, Any]) -> str:
        choices = resp_json.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join([p for p in text_parts if p]).strip()
        return str(content).strip()

    def ocr_page_base64(self, image_base64: str, page_index: int) -> str:
        """调用星火接口识别单页图片文本。"""
        if not self.is_configured():
            raise RuntimeError("Spark OCR 未配置：请在 config/psych.yaml 或环境变量 SPARK_OCR_API_KEY 中填写参数")

        # 使用兼容多模态消息格式；如果服务端不支持，会返回错误并走回退链路。
        prompt = (
            "请对这张中文书页图片做高保真OCR，严格逐行输出正文。"
            "不要总结，不要解释，不要添加不存在的内容。"
            "保留必要的段落换行。"
        )

        payload = {
            "model": self.model,
            "user": f"psych_ocr_page_{page_index}",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            "stream": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "thinking": {"type": self.thinking_type},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        response = requests.post(
            self.url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()

        code = body.get("code", 0)
        if code != 0:
            raise RuntimeError(f"Spark OCR 请求失败，code={code}, message={body.get('message', '')}")

        text = self._extract_text(body)
        return text.strip()

    @staticmethod
    def bytes_to_base64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")
