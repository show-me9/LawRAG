"""
LangChain 1.2.x 对话链。
每个 session_id 独立实例化，短期记忆会话隔离，长期记忆跨会话持久化。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser

from rag.memory import get_llm, ShortTermMemory, LongTermMemory
from rag.prompt import build_qa_prompt, build_rewrite_prompt, format_context
from rag.retriever import build_retriever


class LegalQAChain:
    """
    法律智能问答链，每个 session_id 独立一个实例。

    短期记忆：ShortTermMemory —— 会话内上下文，实例销毁后清空
    长期记忆：LongTermMemory  —— 用户画像，跨会话持久化到磁盘
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

        print(f"[Session {session_id}] 初始化 LLM...")
        self.llm = get_llm()

        print(f"[Session {session_id}] 初始化记忆...")
        self.short_term = ShortTermMemory(self.llm)          # 短期：会话内
        self.long_term  = LongTermMemory(session_id)          # 长期：跨会话

        print(f"[Session {session_id}] 加载检索索引...")
        self.retriever = build_retriever()

        self.rewrite_chain = build_rewrite_prompt() | self.llm | StrOutputParser()
        self.qa_prompt     = build_qa_prompt()

        # 加载已有长期记忆，输出提示
        lt = self.long_term.to_prompt_str()
        if lt:
            print(f"[Session {session_id}] 已加载长期记忆：\n{lt}")
        print(f"[Session {session_id}] 初始化完成\n")

    def _build_system_prompt_extra(self) -> str:
        """将长期记忆拼入 System Prompt 的用户信息区。"""
        lt = self.long_term.to_prompt_str()
        if not lt:
            return ""
        return f"\n## 用户信息（请据此调整回答风格）\n{lt}"

    def _rewrite_question(self, question: str) -> str:
        """指代消解：首轮直接返回原始问题。"""
        if self.short_term.is_empty():
            return question
        rewritten = self.rewrite_chain.invoke({
            "chat_history": self.short_term.get_history_str(),
            "question":     question,
        })
        return rewritten.strip()

    def ask(self, question: str) -> dict:
        """
        核心问答方法。
        Returns:
            {"answer": str, "sources": list[dict], "rewritten_query": str}
        """
        # Step 1：问题改写
        rewritten_q = self._rewrite_question(question)

        # Step 2：混合检索 + Reranker
        docs = self.retriever.invoke(rewritten_q)

        # Step 3：构建上下文
        context = format_context(docs)

        # Step 4：拼入长期记忆到 Prompt
        system_extra = self._build_system_prompt_extra()

        # Step 5：调用 LLM
        qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        answer = qa_chain.invoke({
            "context":       context,
            "chat_history":  self.short_term.get_history_str(),
            "question":      question,
            "system_extra":  system_extra,
        })

        # Step 6：更新短期记忆
        self.short_term.add_turn(human=question, ai=answer)

        # Step 7：异步更新长期记忆（提取用户画像）
        recent = self.short_term.get_recent_str()
        self.long_term.update(self.llm, recent)

        # Step 8：整理引用来源
        sources = [
            {
                "citation":    doc.metadata.get("citation", "未知来源"),
                "source_file": doc.metadata.get("source_file", ""),
                "snippet":     doc.page_content[:150].strip() + "...",
            }
            for doc in docs
        ]

        return {
            "answer":          answer,
            "sources":         sources,
            "rewritten_query": rewritten_q,
        }

    def clear_short_term(self) -> None:
        """清空短期记忆（开始新对话），长期记忆保留。"""
        self.short_term.clear()
        print(f"[Session {self.session_id}] 短期记忆已清空")

    def clear_long_term(self) -> None:
        """清空长期记忆（用户画像重置）。"""
        self.long_term.clear()
        print(f"[Session {self.session_id}] 长期记忆已清空")
