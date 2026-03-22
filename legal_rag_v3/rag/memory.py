"""
对话记忆管理。
LangChain 1.2.x 中 ConversationSummaryBufferMemory 已废弃，
改为手动维护消息列表，超出窗口时调用 LLM 对旧历史进行摘要压缩。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    LLM_MODEL,
    MAX_HISTORY_TURNS,
)


def get_llm() -> ChatOpenAI:
    """
    获取 SiliconFlow LLM 实例。
    LangChain 1.2.x: ChatOpenAI 从 langchain_openai 导入，
    参数名改为 api_key / base_url。
    """
    if not SILICONFLOW_API_KEY:
        raise ValueError("未找到 SILICONFLOW_API_KEY，请在 .env 文件中配置。")
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0.1,
        max_tokens=2048,
    )


class ConversationMemory:
    """
    手动管理的对话记忆。
    - 保留最近 MAX_HISTORY_TURNS 轮完整原始对话
    - 超出后调用 LLM 将旧历史压缩为摘要，摘要以 SystemMessage 形式保留
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # 存储消息列表，元素为 (HumanMessage | AIMessage)
        self._messages: list = []
        # 历史摘要（当消息超出窗口时生成）
        self._summary: str = ""

    def add_turn(self, human: str, ai: str) -> None:
        """保存一轮对话。"""
        self._messages.append(HumanMessage(content=human))
        self._messages.append(AIMessage(content=ai))
        self._compress_if_needed()

    def _compress_if_needed(self) -> None:
        """
        如果消息数超过 MAX_HISTORY_TURNS * 2，
        将最早的一半历史压缩为摘要。
        """
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self._messages) <= max_messages:
            return

        # 取出需要压缩的旧消息
        to_compress = self._messages[: max_messages // 2]
        self._messages = self._messages[max_messages // 2:]

        # 格式化旧消息为文本
        history_text = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
            for m in to_compress
        )

        # 调用 LLM 生成摘要
        summary_prompt = (
            f"请将以下对话历史压缩为简洁的摘要（不超过200字），"
            f"保留关键法律问题和结论：\n\n{history_text}"
        )
        if self._summary:
            summary_prompt = (
                f"已有摘要：{self._summary}\n\n"
                f"请结合以上摘要，将新增对话一并压缩更新：\n\n{history_text}"
            )

        try:
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            self._summary = response.content
        except Exception as e:
            print(f"[警告] 历史压缩失败，保留原始消息：{e}")
            self._messages = to_compress + self._messages

    def get_history_str(self) -> str:
        """返回格式化的对话历史字符串，注入 Prompt。"""
        parts = []
        if self._summary:
            parts.append(f"[早期对话摘要]\n{self._summary}")
        for m in self._messages:
            role = "用户" if isinstance(m, HumanMessage) else "助手"
            parts.append(f"{role}: {m.content}")
        return "\n".join(parts)

    def get_messages(self) -> list:
        """返回原始消息列表（用于问题改写）。"""
        return list(self._messages)

    def clear(self) -> None:
        """清空所有记忆。"""
        self._messages = []
        self._summary  = ""

    def is_empty(self) -> bool:
        return len(self._messages) == 0 and not self._summary
