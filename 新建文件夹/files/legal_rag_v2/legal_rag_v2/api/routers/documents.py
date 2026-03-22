"""
文档管理接口：上传、列表、删除。
"""
import os
import sys
import json
import shutil
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from scripts.ingest_docx import parse_docx, write_to_chroma
from rag.retriever import save_bm25_index, BM25_INDEX_PATH
from api.dependencies import reset_qa_chain

router = APIRouter(tags=["文档管理"])

# 上传文件临时存放目录
UPLOAD_DIR      = "./data/raw_docx"
INGESTED_RECORD = "./data/ingested.json"
INGEST_STATUS   = {}   # 记录当前摄入任务状态，key 为文件名


# ── 数据模型 ──────────────────────────────────────────────

class DocumentInfo(BaseModel):
    filename:     str
    file_hash:    str
    article_count: int = 0
    status:       str  # ingested / pending


class IngestStatusResponse(BaseModel):
    filename: str
    status:   str    # processing / done / failed
    message:  str = ""


# ── 工具函数 ──────────────────────────────────────────────

def load_record() -> dict:
    if os.path.exists(INGESTED_RECORD):
        with open(INGESTED_RECORD, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_record(record: dict) -> None:
    os.makedirs("./data", exist_ok=True)
    with open(INGESTED_RECORD, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def compute_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 后台摄入任务 ──────────────────────────────────────────

def _do_ingest(filepath: str, filename: str) -> None:
    """后台执行摄入，完成后重置 QAChain 单例加载最新索引。"""
    global INGEST_STATUS
    INGEST_STATUS[filename] = {"status": "processing", "message": "正在解析文档..."}
    try:
        docs = parse_docx(filepath)
        if not docs:
            INGEST_STATUS[filename] = {"status": "failed", "message": "未解析出任何条文，请检查文档格式"}
            return

        INGEST_STATUS[filename] = {"status": "processing", "message": f"正在向量化 {len(docs)} 个条文..."}
        write_to_chroma(docs)
        save_bm25_index(_load_all_docs_for_bm25())

        # 更新 hash 记录
        record = load_record()
        record[filename] = {
            "hash":          compute_hash(filepath),
            "article_count": len(docs),
        }
        save_record(record)

        # 重置 QAChain，下次请求时自动加载最新索引
        reset_qa_chain()

        INGEST_STATUS[filename] = {
            "status":  "done",
            "message": f"摄入完成，共 {len(docs)} 个条文",
        }
    except Exception as e:
        INGEST_STATUS[filename] = {"status": "failed", "message": str(e)}


def _load_all_docs_for_bm25() -> list:
    """重新解析所有已摄入文档，用于重建 BM25 索引。"""
    record  = load_record()
    all_docs = []
    for fname in record:
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.exists(fpath):
            try:
                all_docs.extend(parse_docx(fpath))
            except Exception:
                pass
    return all_docs


# ── 接口 ─────────────────────────────────────────────────

@router.get("", response_model=List[DocumentInfo])
def list_documents():
    """获取已摄入文档列表。"""
    record = load_record()
    result = []
    for fname, info in record.items():
        # 兼容旧格式（只存 hash 字符串）
        if isinstance(info, str):
            info = {"hash": info, "article_count": 0}
        result.append(DocumentInfo(
            filename      = fname,
            file_hash     = info.get("hash", ""),
            article_count = info.get("article_count", 0),
            status        = "ingested",
        ))
    return result


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    上传 Word 文档并触发后台摄入。
    立即返回，摄入进度通过 /upload/status/{filename} 查询。
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="仅支持 .docx 格式")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, file.filename)

    # 保存文件到本地
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    # 查重：hash 相同则跳过
    new_hash = compute_hash(filepath)
    record   = load_record()
    existing = record.get(file.filename, {})
    if isinstance(existing, dict) and existing.get("hash") == new_hash:
        return {"message": f"文件未变化，跳过摄入：{file.filename}", "skipped": True}

    # 后台异步摄入
    background_tasks.add_task(_do_ingest, filepath, file.filename)
    return {"message": f"已开始摄入：{file.filename}", "skipped": False}


@router.get("/upload/status/{filename}", response_model=IngestStatusResponse)
def get_ingest_status(filename: str):
    """查询指定文件的摄入进度。"""
    info = INGEST_STATUS.get(filename)
    if not info:
        # 不在进度表里，说明已完成或未上传过
        record = load_record()
        if filename in record:
            return IngestStatusResponse(filename=filename, status="done", message="已完成")
        raise HTTPException(status_code=404, detail="未找到该文件的摄入记录")
    return IngestStatusResponse(filename=filename, **info)


@router.delete("/{filename}")
def delete_document(filename: str):
    """
    删除文档：
    1. 从 ChromaDB 删除对应条文的向量
    2. 从本地磁盘删除文件
    3. 从摄入记录中移除
    4. 重建 BM25 索引
    """
    import chromadb
    record = load_record()
    if filename not in record:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 从 ChromaDB 删除（按 source_file 元数据过滤）
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        col    = client.get_collection(CHROMA_COLLECTION)
        results = col.get(where={"source_file": filename})
        if results["ids"]:
            col.delete(ids=results["ids"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB 删除失败：{e}")

    # 删除本地文件
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    # 更新摄入记录
    del record[filename]
    save_record(record)

    # 重建 BM25 索引
    remaining_docs = _load_all_docs_for_bm25()
    if remaining_docs:
        save_bm25_index(remaining_docs)
    elif os.path.exists(BM25_INDEX_PATH):
        os.remove(BM25_INDEX_PATH)

    # 重置 QAChain
    reset_qa_chain()

    return {"message": f"已删除：{filename}"}
