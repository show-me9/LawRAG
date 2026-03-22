"""
智能问答接口，支持 session_id 会话隔离。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List
import uuid

from api.dependencies import get_qa_chain, remove_session

router = APIRouter(tags=["智能问答"])


# ── 数据模型 ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="会话 ID，前端生成后持久化保存，同一对话每次请求传相同值"
    )


class SourceItem(BaseModel):
    citation:    str
    source_file: str
    snippet:     str


class ChatResponse(BaseModel):
    answer:          str
    sources:         List[SourceItem]
    rewritten_query: str
    session_id:      str


class ClearRequest(BaseModel):
    session_id: str
    clear_long_term: bool = False   # 是否同时清空长期记忆（用户画像）


class MemoryResponse(BaseModel):
    session_id:   str
    long_term:    dict   # 长期记忆内容
    has_short_term: bool # 短期记忆是否有内容


# ── 接口 ─────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    发送问题，返回回答及引用条文。
    同一 session_id 的请求共享短期记忆（会话上下文）。
    不同 session_id 完全隔离，互不影响。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        chain  = get_qa_chain(req.session_id)
        result = chain.ask(req.question)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"索引未就绪，请先摄入文档：{e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    return ChatResponse(
        answer          = result["answer"],
        sources         = [SourceItem(**s) for s in result["sources"]],
        rewritten_query = result["rewritten_query"],
        session_id      = req.session_id,
    )


@router.post("/clear")
def clear_memory(req: ClearRequest):
    """
    清空记忆。
    - 默认只清空短期记忆（当前对话上下文），开始新一轮对话
    - clear_long_term=true 时同时清空长期记忆（用户画像重置）
    """
    try:
        chain = get_qa_chain(req.session_id)
        chain.clear_short_term()
        if req.clear_long_term:
            chain.clear_long_term()
            return {"message": "短期记忆与长期记忆已清空"}
        return {"message": "短期记忆已清空，长期记忆保留"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
def close_session(session_id: str):
    """
    销毁会话实例，释放服务端短期记忆。
    用户关闭浏览器或退出时调用。长期记忆已持久化，不受影响。
    """
    remove_session(session_id)
    return {"message": f"会话 {session_id} 已关闭"}


@router.get("/memory/{session_id}", response_model=MemoryResponse)
def get_memory(session_id: str):
    """查看指定会话的记忆状态（调试用）。"""
    try:
        chain = get_qa_chain(session_id)
        return MemoryResponse(
            session_id    = session_id,
            long_term     = chain.long_term.get_raw(),
            has_short_term= not chain.short_term.is_empty(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
