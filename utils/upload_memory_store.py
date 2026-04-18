import json
import os
from datetime import datetime

from utils.path_tool import get_abs_path


UPLOAD_MEMORY_STORE_PATH = get_abs_path("storage/upload_memories.json")


def _ensure_store_dir() -> None:
    os.makedirs(os.path.dirname(UPLOAD_MEMORY_STORE_PATH), exist_ok=True)


def _load_all() -> dict:
    _ensure_store_dir()
    if not os.path.exists(UPLOAD_MEMORY_STORE_PATH):
        return {}
    with open(UPLOAD_MEMORY_STORE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _save_all(data: dict) -> None:
    _ensure_store_dir()
    with open(UPLOAD_MEMORY_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_upload_memory(session_id: str, filename: str, summary: str, saved_path: str = "") -> dict:
    data = _load_all()
    item = {
        "filename": filename,
        "summary": summary,
        "saved_path": saved_path,
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    data.setdefault(session_id, []).append(item)
    _save_all(data)
    return item


def get_upload_memory(session_id: str) -> list[dict]:
    data = _load_all()
    return data.get(session_id, [])
