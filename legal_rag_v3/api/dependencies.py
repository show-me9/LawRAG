"""
全局共享依赖：QAChain 单例。
FastAPI 启动时初始化一次，所有请求共用同一个实例。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chain import LegalQAChain

_qa_chain: LegalQAChain | None = None


def get_qa_chain() -> LegalQAChain:
    """
    返回 QAChain 单例。
    若尚未初始化（索引不存在）则抛出异常，由路由层捕获返回 503。
    """
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = LegalQAChain()
    return _qa_chain


def reset_qa_chain() -> None:
    """重置单例（摄入新文档后调用，使检索器加载最新索引）。"""
    global _qa_chain
    _qa_chain = None
