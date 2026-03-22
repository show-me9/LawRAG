"""
Prompt 模板设计。
system_extra 占位符用于注入长期记忆（用户画像）。
"""
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEMPLATE = """你是一位专业的法律智能问答助手，专门负责解答中国法律相关问题。{system_extra}

## 回答规则
1. **只基于提供的法律条文作答**，不得凭空捏造法条内容。
2. **每一个核心观点必须引用对应条文**，引用格式：【来源】
3. 如果检索到的条文不足以回答问题，明确告知用户，并说明可能需要查阅哪类法律。
4. 回答语言简洁专业，避免歧义。
5. 涉及具体案件时，提示用户咨询专业律师。

## 参考条文
{context}

## 对话历史
{chat_history}
"""

REWRITE_TEMPLATE = """你是一个法律问答助手。请根据以下对话历史，
将用户的最新问题改写为一个完整、独立、可以直接用于检索的问题。

要求：
- 消解所有代词（他、她、他们、这、那、此等）
- 补充必要的上下文（当事方、法律关系、事件背景）
- 不要回答问题，只输出改写后的问题
- 如果问题已经足够完整，直接原样输出

对话历史：
{chat_history}

用户最新问题：{question}

改写后的完整问题："""


def build_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human",  "{question}"),
    ])


def build_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("human", REWRITE_TEMPLATE),
    ])


def format_context(docs: list) -> str:
    if not docs:
        return "（未检索到相关条文）"
    parts = []
    for i, doc in enumerate(docs, 1):
        citation = doc.metadata.get("citation", "未知来源")
        text     = doc.page_content.strip()
        parts.append(f"[条文{i}]【{citation}】\n{text}")
    return "\n\n".join(parts)
