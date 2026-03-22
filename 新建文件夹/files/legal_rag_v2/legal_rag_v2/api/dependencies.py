"""
全局依赖：按 session_id 管理 QAChain 实例，实现会话隔离。
每个 session_id 对应独立的短期记忆，长期记忆由 LongTermMemory 自行持久化。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chain import LegalQAChain

# session_id -> LegalQAChain 实例
_sessions: dict[str, LegalQAChain] = {}


def get_qa_chain(session_id: str) -> LegalQAChain:
    """
    返回指定 session_id 的 QAChain 实例。
    不存在则新建，已存在则复用（短期记忆连续）。
    """
    if session_id not in _sessions:
        _sessions[session_id] = LegalQAChain(session_id=session_id)
    return _sessions[session_id]


def remove_session(session_id: str) -> None:
    """销毁指定会话实例，释放短期记忆。长期记忆已持久化到磁盘，不受影响。"""
    if session_id in _sessions:
        del _sessions[session_id]


def list_sessions() -> list[str]:
    """返回当前活跃的所有 session_id 列表。"""
    return list(_sessions.keys())
