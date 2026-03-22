"""
配置管理接口：读取和更新系统配置。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests as http_requests

router = APIRouter(tags=["配置管理"])


# ── 数据模型 ──────────────────────────────────────────────

class SettingsResponse(BaseModel):
    llm_model:       str
    embedding_model: str
    reranker_model:  str
    vector_search_k: int
    bm25_search_k:   int
    rerank_top_k:    int
    bm25_weight:     float
    vector_weight:   float
    api_key_set:     bool   # 只返回是否已设置，不返回明文


class SettingsUpdateRequest(BaseModel):
    api_key:         Optional[str]   = None
    llm_model:       Optional[str]   = None
    embedding_model: Optional[str]   = None
    reranker_model:  Optional[str]   = None
    vector_search_k: Optional[int]   = None
    bm25_search_k:   Optional[int]   = None
    rerank_top_k:    Optional[int]   = None
    bm25_weight:     Optional[float] = None
    vector_weight:   Optional[float] = None


class ConnectTestResponse(BaseModel):
    success: bool
    message: str


# ── 接口 ─────────────────────────────────────────────────

@router.get("", response_model=SettingsResponse)
def get_settings():
    """获取当前系统配置（API Key 仅返回是否已设置）。"""
    import config as cfg
    return SettingsResponse(
        llm_model       = cfg.LLM_MODEL,
        embedding_model = cfg.EMBEDDING_MODEL,
        reranker_model  = cfg.RERANKER_MODEL,
        vector_search_k = cfg.VECTOR_SEARCH_K,
        bm25_search_k   = cfg.BM25_SEARCH_K,
        rerank_top_k    = cfg.RERANK_TOP_K,
        bm25_weight     = cfg.ENSEMBLE_WEIGHTS[0],
        vector_weight   = cfg.ENSEMBLE_WEIGHTS[1],
        api_key_set     = bool(cfg.SILICONFLOW_API_KEY),
    )


@router.post("")
def update_settings(req: SettingsUpdateRequest):
    """
    更新系统配置，写入 .env 文件并热更新 config 模块。
    注意：修改模型或检索参数后，需重新摄入或重启服务才能完全生效。
    """
    import config as cfg

    env_path = ".env"
    env_lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            env_lines = f.readlines()

    def set_env(key: str, value: str):
        """更新或追加 .env 中的键值对。"""
        for i, line in enumerate(env_lines):
            if line.startswith(f"{key}="):
                env_lines[i] = f"{key}={value}\n"
                return
        env_lines.append(f"{key}={value}\n")

    # 写入 .env 并热更新 config
    if req.api_key is not None:
        set_env("SILICONFLOW_API_KEY", req.api_key)
        cfg.SILICONFLOW_API_KEY = req.api_key
        os.environ["SILICONFLOW_API_KEY"] = req.api_key

    if req.llm_model is not None:
        set_env("LLM_MODEL", req.llm_model)
        cfg.LLM_MODEL = req.llm_model

    if req.embedding_model is not None:
        set_env("EMBEDDING_MODEL", req.embedding_model)
        cfg.EMBEDDING_MODEL = req.embedding_model

    if req.reranker_model is not None:
        set_env("RERANKER_MODEL", req.reranker_model)
        cfg.RERANKER_MODEL = req.reranker_model

    if req.vector_search_k is not None:
        cfg.VECTOR_SEARCH_K = req.vector_search_k

    if req.bm25_search_k is not None:
        cfg.BM25_SEARCH_K = req.bm25_search_k

    if req.rerank_top_k is not None:
        cfg.RERANK_TOP_K = req.rerank_top_k

    if req.bm25_weight is not None and req.vector_weight is not None:
        cfg.ENSEMBLE_WEIGHTS = [req.bm25_weight, req.vector_weight]

    # 持久化到 .env
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(env_lines)

    return {"message": "配置已更新"}


@router.get("/test-connection", response_model=ConnectTestResponse)
def test_connection():
    """测试 SiliconFlow API 连接是否正常。"""
    import config as cfg

    if not cfg.SILICONFLOW_API_KEY:
        return ConnectTestResponse(success=False, message="未配置 API Key")

    try:
        resp = http_requests.get(
            f"{cfg.SILICONFLOW_BASE_URL}/models",
            headers={"Authorization": f"Bearer {cfg.SILICONFLOW_API_KEY}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return ConnectTestResponse(success=True, message="连接成功")
        else:
            return ConnectTestResponse(
                success=False,
                message=f"连接失败，状态码：{resp.status_code}"
            )
    except Exception as e:
        return ConnectTestResponse(success=False, message=f"连接异常：{e}")
