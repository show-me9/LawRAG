"""
记忆管理模块。

短期记忆：单次会话内的对话上下文，会话结束后清空。
         - 保留最近 MAX_HISTORY_TURNS 轮原始消息
         - 超出后对早期消息做摘要压缩（仍服务于同一会话）

长期记忆：跨会话持久化，存储用户画像与偏好。
         - 每次新会话开始时加载，注入 System Prompt
         - 对话过程中动态更新，写回 JSON 文件
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    LLM_MODEL,
    MAX_HISTORY_TURNS,
)

LONG_TERM_MEMORY_DIR = "./data/user_memory"


# ── LLM 实例 ──────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    if not SILICONFLOW_API_KEY:
        raise ValueError("未找到 SILICONFLOW_API_KEY，请在 .env 文件中配置。")
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0.1,
        max_tokens=2048,
    )


# ══════════════════════════════════════════════════════════
# 长期记忆
# ══════════════════════════════════════════════════════════

class LongTermMemory:
    """
    跨会话持久化的用户记忆，以 session_id 为 key。
    存储结构：
    {
        "background":  "用户职业背景、身份等",
        "preferences": "偏好的回答风格、关注领域等",
        "notes":       "其他值得记住的信息"
    }
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._path = os.path.join(LONG_TERM_MEMORY_DIR, f"{session_id}.json")
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"background": "", "preferences": "", "notes": ""}

    def _save(self) -> None:
        os.makedirs(LONG_TERM_MEMORY_DIR, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def to_prompt_str(self) -> str:
        """格式化为字符串，注入 System Prompt。"""
        parts = []
        if self._data.get("background"):
            parts.append(f"用户背景：{self._data['background']}")
        if self._data.get("preferences"):
            parts.append(f"回答偏好：{self._data['preferences']}")
        if self._data.get("notes"):
            parts.append(f"其他信息：{self._data['notes']}")
        return "\n".join(parts)

    def update(self, llm: ChatOpenAI, recent_conversation: str) -> None:
        """
        根据最近对话提取用户画像并更新。
        每轮对话结束后调用，不阻塞主流程。
        """
        existing = self.to_prompt_str()
        prompt = f"""请根据以下对话内容，提取或更新用户的画像信息。
只提取有持久价值的信息（如职业背景、关注领域、偏好风格），不要记录具体问题内容。
以 JSON 格式输出，包含三个字段：background、preferences、notes。
只输出 JSON，不要有任何其他文字。

已有画像：
{existing if existing else '（暂无）'}

最近对话：
{recent_conversation}

输出：
{{"background": "...", "preferences": "...", "notes": "..."}}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content.strip().replace("```json", "").replace("```", "").strip()
            updated = json.loads(text)
            for key in ("background", "preferences", "notes"):
                if updated.get(key):
                    self._data[key] = updated[key]
            self._save()
        except Exception as e:
            print(f"[长期记忆] 更新失败（不影响正常使用）：{e}")

    def get_raw(self) -> dict:
        return dict(self._data)

    def clear(self) -> None:
        self._data = {"background": "", "preferences": "", "notes": ""}
        self._save()


# ══════════════════════════════════════════════════════════
# 短期记忆
# ══════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    单次会话内的对话上下文记忆。
    会话结束后实例销毁，不持久化。
    """

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._messages: list = []
        self._summary: str = ""

    def add_turn(self, human: str, ai: str) -> None:
        self._messages.append(HumanMessage(content=human))
        self._messages.append(AIMessage(content=ai))
        self._compress_if_needed()

    def _compress_if_needed(self) -> None:
        """超出窗口时压缩早期消息，释放 context 空间，仍服务于同一会话。"""
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self._messages) <= max_messages:
            return

        to_compress    = self._messages[: max_messages // 2]
        self._messages = self._messages[max_messages // 2:]

        history_text = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
            for m in to_compress
        )
        base_prompt = f"请将以下对话压缩为简洁摘要（不超过150字），保留关键法律问题和结论：\n\n{history_text}"
        if self._summary:
            base_prompt = (
                f"已有摘要：{self._summary}\n\n"
                f"请结合以上摘要，将新增对话一并压缩更新：\n\n{history_text}"
            )
        try:
            response = self.llm.invoke([HumanMessage(content=base_prompt)])
            self._summary = response.content
        except Exception as e:
            print(f"[短期记忆] 压缩失败，保留原始消息：{e}")
            self._messages = to_compress + self._messages

    def get_history_str(self) -> str:
        parts = []
        if self._summary:
            parts.append(f"[早期对话摘要]\n{self._summary}")
        for m in self._messages:
            role = "用户" if isinstance(m, HumanMessage) else "助手"
            parts.append(f"{role}: {m.content}")
        return "\n".join(parts)

    def get_recent_str(self) -> str:
        """返回最近 2 轮对话文本，供长期记忆提取使用。"""
        recent = self._messages[-4:] if len(self._messages) >= 4 else self._messages
        return "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
            for m in recent
        )

    def is_empty(self) -> bool:
        return len(self._messages) == 0 and not self._summary

    def clear(self) -> None:
        self._messages = []
        self._summary  = ""
