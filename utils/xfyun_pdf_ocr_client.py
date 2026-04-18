import base64
import hashlib
import hmac
import os
import time
from typing import Any

import requests

from utils.config_handler import psych_conf


class XfyunPdfOcrClient:
    """讯飞 PDF OCR 任务式客户端（start/status）。"""

    def __init__(self):
        self.base_url = psych_conf.get("xfyun_base_url", "https://iocr.xfyun.cn/ocrzdq").rstrip("/")
        self.start_url = f"{self.base_url}/v1/pdfOcr/start"
        self.status_url = f"{self.base_url}/v1/pdfOcr/status"

        self.app_id = (os.getenv("XFYUN_OCR_APP_ID") or psych_conf.get("xfyun_app_id", "")).strip()
        self.api_secret = (os.getenv("XFYUN_OCR_API_SECRET") or psych_conf.get("xfyun_api_secret", "")).strip()

        self.export_format = psych_conf.get("xfyun_export_format", "markdown")
        self.poll_interval_seconds = int(psych_conf.get("xfyun_poll_interval_seconds", 5))
        self.timeout_seconds = int(psych_conf.get("xfyun_timeout_seconds", 1800))

    def is_configured(self) -> bool:
        return bool(self.app_id and self.api_secret)

    def _build_headers(self) -> dict[str, str]:
        ts = str(int(time.time()))
        md5_text = hashlib.md5(f"{self.app_id}{ts}".encode("utf-8")).hexdigest()
        sign = base64.b64encode(
            hmac.new(self.api_secret.encode("utf-8"), md5_text.encode("utf-8"), digestmod=hashlib.sha1).digest()
        ).decode("utf-8")
        return {
            "appId": self.app_id,
            "timestamp": ts,
            "signature": sign,
        }

    def start_task(self, pdf_path: str) -> str:
        if not self.is_configured():
            raise RuntimeError("XFYUN OCR 未配置：请填写 xfyun_app_id 与 xfyun_api_secret")

        headers = self._build_headers()
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            data = {"exportFormat": self.export_format}
            response = requests.post(
                self.start_url,
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )
        response.raise_for_status()
        body = response.json()

        if int(body.get("code", 0)) != 0 or not body.get("flag", False):
            raise RuntimeError(f"XFYUN OCR start失败: code={body.get('code')} desc={body.get('desc')}")

        task_no = ((body.get("data") or {}).get("taskNo") or "").strip()
        if not task_no:
            raise RuntimeError("XFYUN OCR start成功但未返回 taskNo")
        return task_no

    def query_status(self, task_no: str) -> dict[str, Any]:
        headers = self._build_headers()
        response = requests.get(
            self.status_url,
            headers=headers,
            params={"taskNo": task_no},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        if int(body.get("code", 0)) != 0 or not body.get("flag", False):
            raise RuntimeError(f"XFYUN OCR status失败: code={body.get('code')} desc={body.get('desc')}")
        return body

    def wait_until_finished(self, task_no: str) -> dict[str, Any]:
        start_time = time.time()
        while True:
            body = self.query_status(task_no)
            data = body.get("data") or {}
            status = (data.get("status") or "").upper()
            if status in {"FINISH", "ANY_FAILED", "FAILED", "STOP"}:
                return data
            if (time.time() - start_time) > self.timeout_seconds:
                raise TimeoutError(f"XFYUN OCR任务超时: taskNo={task_no}")
            time.sleep(max(5, self.poll_interval_seconds))

    @staticmethod
    def _download_text(url: str) -> str:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()

        # markdown/plain/json 文件优先按文本解码。
        if "text" in content_type or "json" in content_type or url.lower().endswith((".md", ".txt", ".json")):
            return response.content.decode("utf-8", errors="ignore").strip()

        # 未知类型尝试 utf-8 解码。
        return response.content.decode("utf-8", errors="ignore").strip()

    def fetch_result_text(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """返回每页文本，格式: [{page, text}]。"""
        page_list = data.get("pageList") or []
        results: list[dict[str, Any]] = []

        for page_item in page_list:
            status = (page_item.get("status") or "").upper()
            if status != "FINISH":
                continue
            down_url = (page_item.get("downUrl") or "").strip()
            if not down_url:
                continue
            text = self._download_text(down_url)
            if text:
                results.append({"page": page_item.get("pageNum"), "text": text})

        if results:
            return results

        # 如果 pageList 没有可用内容，则尝试任务级 downUrl。
        task_down_url = (data.get("downUrl") or "").strip()
        if task_down_url:
            task_text = self._download_text(task_down_url)
            if task_text:
                return [{"page": None, "text": task_text}]

        return []
