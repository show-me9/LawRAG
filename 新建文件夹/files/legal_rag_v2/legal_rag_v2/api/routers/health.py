"""
健康检查接口。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter
from pydantic import BaseModel

from config import CHROMA_PERSIST_DIR, SILICONFLOW_API_KEY
from rag.retriever import BM25_INDEX_PATH

router = APIRouter(tags=["健康检查"])


class HealthResponse(BaseModel):
    status: str             # ok / degraded
    chroma_ready: bool      # ChromaDB 索引是否存在
    bm25_ready: bool        # BM25 索引是否存在
    api_key_set: bool       # API Key 是否已配置
    message: str


@router.get("/health", response_model=HealthResponse)
def health_check():
    chroma_ready = (
        os.path.exists(CHROMA_PERSIST_DIR)
        and len(os.listdir(CHROMA_PERSIST_DIR)) > 0
    )
    bm25_ready   = os.path.exists(BM25_INDEX_PATH)
    api_key_set  = bool(SILICONFLOW_API_KEY)

    all_ok = chroma_ready and bm25_ready and api_key_set
    return HealthResponse(
        status      = "ok" if all_ok else "degraded",
        chroma_ready= chroma_ready,
        bm25_ready  = bm25_ready,
        api_key_set = api_key_set,
        message     = "服务就绪" if all_ok else "请先完成数据摄入并配置 API Key",
    )
