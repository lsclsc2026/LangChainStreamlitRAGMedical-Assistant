import os
import uuid

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from rag.vector_store import VectorStoreService
from utils.file_handler import clean_text, pdf_loader
from utils.path_tool import get_abs_path
from utils.upload_memory_store import add_upload_memory, get_upload_memory


app = FastAPI(title="langchain-agent-backend", version="1.0.0")
vector_store = VectorStoreService()

UPLOAD_DIR = get_abs_path("storage/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _admin_auth(x_admin_token: str = Header(default="")):
    expected = os.getenv("ADMIN_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN 未配置")
    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="管理员鉴权失败")


def _summarize_uploaded_file(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return clean_text(f.read())[:800]
    if lower.endswith(".pdf"):
        docs = pdf_loader(path)
        text = "\n".join(doc.page_content for doc in docs[:2])
        return clean_text(text)[:800]
    return "不支持自动摘要，仅记录文件名。"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_user_file(session_id: str, file: UploadFile = File(...)):
    filename = file.filename or "upload.bin"
    save_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        summary = _summarize_uploaded_file(save_path)
    except Exception:
        summary = "文件已上传，摘要提取失败。"

    memory_item = add_upload_memory(
        session_id=session_id,
        filename=filename,
        summary=summary,
        saved_path=save_path,
    )
    return {"ok": True, "memory": memory_item}


@app.get("/api/memory/{session_id}/uploads")
def list_upload_memories(session_id: str):
    return {"ok": True, "items": get_upload_memory(session_id)}


@app.get("/api/admin/knowledge/files")
def list_knowledge_files(_: None = Depends(_admin_auth)):
    data_path = get_abs_path("data")
    files = []
    for root, _, filenames in os.walk(data_path):
        for name in filenames:
            if name.lower().endswith((".txt", ".pdf")):
                abs_path = os.path.join(root, name)
                files.append(os.path.relpath(abs_path, data_path))
    return {"ok": True, "files": sorted(files)}


@app.post("/api/admin/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...), _: None = Depends(_admin_auth)):
    data_path = get_abs_path("data")
    os.makedirs(data_path, exist_ok=True)
    filename = file.filename or "knowledge.txt"
    save_path = os.path.join(data_path, filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "saved": save_path}


@app.delete("/api/admin/knowledge/file")
def delete_knowledge_file(relative_path: str, _: None = Depends(_admin_auth)):
    data_path = get_abs_path("data")
    full_path = os.path.normpath(os.path.join(data_path, relative_path))
    if not full_path.startswith(os.path.normpath(data_path)):
        raise HTTPException(status_code=400, detail="非法路径")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    os.remove(full_path)
    return JSONResponse({"ok": True})


@app.post("/api/admin/knowledge/rebuild")
def rebuild_knowledge(_: None = Depends(_admin_auth)):
    vector_store.reset_store(clear_md5=True)
    vector_store.load_document(force_reload=True)
    return {"ok": True, "message": "知识库重建完成"}
