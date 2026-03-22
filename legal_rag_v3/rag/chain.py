"""
LangChain 1.2.x 对话链组装。
启动时只从磁盘加载索引，不读取任何原始文档。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser

from rag.memory import get_llm, ConversationMemory
from rag.prompt import build_qa_prompt, build_rewrite_prompt, format_context
from rag.retriever import build_retriever


class LegalQAChain:
    """
    法律智能问答链。

    用法：
        qa = LegalQAChain()       # 无需传入文档，直接从磁盘加载索引
        result = qa.ask("合同解除需要满足哪些条件？")
        print(result["answer"])
    """

    def __init__(self):
        print("正在初始化 LLM...")
        self.llm = get_llm()

        print("正在初始化对话记忆...")
        self.memory = ConversationMemory(self.llm)

        print("正在加载检索索引（向量 + BM25 + Reranker）...")
        self.retriever = build_retriever()   # 直接从磁盘加载，无需文档列表

        self.rewrite_chain = build_rewrite_prompt() | self.llm | StrOutputParser()
        self.qa_prompt = build_qa_prompt()

        print("初始化完成！\n")

    def _rewrite_question(self, question: str) -> str:
        """结合对话历史进行指代消解，首轮直接返回原始问题。"""
        if self.memory.is_empty():
            return question
        rewritten = self.rewrite_chain.invoke({
            "chat_history": self.memory.get_history_str(),
            "question":     question,
        })
        return rewritten.strip()

    def ask(self, question: str) -> dict:
        """
        核心问答方法。
        Returns:
            {"answer": str, "sources": list[dict], "rewritten_query": str}
        """
        # Step 1：问题改写（指代消解）
        rewritten_q = self._rewrite_question(question)

        # Step 2：混合检索 + Reranker 精排
        docs = self.retriever.invoke(rewritten_q)

        # Step 3：构建带引用标记的上下文
        context = format_context(docs)

        # Step 4：调用 LLM 生成回答
        qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        answer = qa_chain.invoke({
            "context":      context,
            "chat_history": self.memory.get_history_str(),
            "question":     question,
        })

        # Step 5：保存本轮对话
        self.memory.add_turn(human=question, ai=answer)

        # Step 6：整理引用来源
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

    def clear_memory(self) -> None:
        """清空对话记忆，开始新一轮对话。"""
        self.memory.clear()
        print("对话记忆已清空。")
