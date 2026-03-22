"""
智能问答接口。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from api.dependencies import get_qa_chain

router = APIRouter(tags=["智能问答"])


# ── 数据模型 ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    citation:    str
    source_file: str
    snippet:     str


class ChatResponse(BaseModel):
    answer:          str
    sources:         List[SourceItem]
    rewritten_query: str


# ── 接口 ─────────────────────────────────────────────────

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    发送问题，返回回答及引用条文。
    多轮对话记忆由服务端 QAChain 单例维护，无需客户端传递历史。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        chain  = get_qa_chain()
        result = chain.ask(req.question)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"索引未就绪，请先摄入文档：{e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败：{e}")

    return ChatResponse(
        answer          = result["answer"],
        sources         = [SourceItem(**s) for s in result["sources"]],
        rewritten_query = result["rewritten_query"],
    )


@router.post("/clear")
def clear_memory():
    """清空服务端对话记忆，开始新一轮对话。"""
    try:
        chain = get_qa_chain()
        chain.clear_memory()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "对话记忆已清空"}
