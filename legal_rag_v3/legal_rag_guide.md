# 基于 LangChain 的法律智能问答系统 — 详细实现指南

> 技术栈：LangChain · MinerU · ChromaDB · SiliconFlow (Embedding / Reranker / LLM) · BM25

---

## 目录

1. [项目结构](#1-项目结构)
2. [环境依赖安装](#2-环境依赖安装)
3. [配置管理](#3-配置管理)
4. [数据摄入层](#4-数据摄入层)
   - 4.1 MinerU PDF 转 Markdown
   - 4.2 Markdown 解析与分块
   - 4.3 向量化写入 ChromaDB
5. [检索层](#5-检索层)
   - 5.1 SiliconFlow Embedding 封装
   - 5.2 混合检索（向量 + BM25）
   - 5.3 Reranker 精排
6. [对话记忆与问题改写](#6-对话记忆与问题改写)
7. [Prompt 设计与引用条文](#7-prompt-设计与引用条文)
8. [LangChain 对话链组装](#8-langchain-对话链组装)
9. [主程序入口](#9-主程序入口)
10. [关键注意事项与调优建议](#10-关键注意事项与调优建议)

---

## 1. 项目结构

```
legal_rag/
├── data/
│   ├── raw_pdfs/              # 原始 PDF 文件
│   ├── markdown/              # MinerU 转换后的 Markdown 文件
│   └── chroma_db/             # ChromaDB 本地持久化目录
├── rag/
│   ├── __init__.py
│   ├── embeddings.py          # SiliconFlow Embedding 封装
│   ├── retriever.py           # 混合检索 + Reranker 封装
│   ├── memory.py              # 对话记忆封装
│   ├── prompt.py              # Prompt 模板
│   └── chain.py               # LangChain 对话链组装
├── scripts/
│   ├── pdf_to_md.py           # 批量调用 MinerU 转换 PDF
│   └── ingest.py              # 数据摄入主脚本
├── config.py                  # 统一配置
├── main.py                    # 程序入口（命令行交互）
└── requirements.txt
```

---

## 2. 环境依赖安装

```bash
# Python 依赖
pip install langchain langchain-community langchain-core
pip install chromadb
pip install rank-bm25
pip install openai          # SiliconFlow 兼容 OpenAI SDK
pip install python-dotenv
pip install tqdm

# MinerU（PDF 转 Markdown）
pip install magic-pdf[full]

# 可选：本地 embedding（完全离线方案）
# pip install sentence-transformers
```

`requirements.txt`：

```
langchain>=0.2.0
langchain-community>=0.2.0
langchain-core>=0.2.0
chromadb>=0.5.0
rank-bm25>=0.2.2
openai>=1.0.0
python-dotenv>=1.0.0
tqdm>=4.66.0
magic-pdf[full]>=0.7.0
```

---

## 3. 配置管理

`config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

# ── SiliconFlow ──────────────────────────────────────────
SILICONFLOW_API_KEY  = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 模型名称（按需替换为 SiliconFlow 上实际可用的模型）
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
LLM_MODEL       = "deepseek-ai/DeepSeek-V2.5"   # 或其他 SiliconFlow 上的 LLM

# ── ChromaDB ─────────────────────────────────────────────
CHROMA_PERSIST_DIR  = "./data/chroma_db"
CHROMA_COLLECTION   = "legal_docs"

# ── 检索参数 ─────────────────────────────────────────────
VECTOR_SEARCH_K   = 20    # 向量检索召回数
BM25_SEARCH_K     = 20    # BM25 召回数
RERANK_TOP_K      = 5     # Reranker 最终保留数
ENSEMBLE_WEIGHTS  = [0.4, 0.6]   # [BM25权重, 向量权重]

# ── 对话记忆 ─────────────────────────────────────────────
MAX_TOKEN_LIMIT   = 2000  # 记忆压缩阈值（token 数）

# ── 路径 ─────────────────────────────────────────────────
RAW_PDF_DIR  = "./data/raw_pdfs"
MARKDOWN_DIR = "./data/markdown"
```

`.env` 文件（不要提交到 git）：

```
SILICONFLOW_API_KEY=your_api_key_here
```

---

## 4. 数据摄入层

### 4.1 MinerU PDF 转 Markdown

`scripts/pdf_to_md.py`

```python
"""
批量将 raw_pdfs/ 下的 PDF 用 MinerU 转换为 Markdown，
输出到 markdown/ 目录。
本脚本为一次性离线处理，运行完成后无需重复执行。
"""
import os
import subprocess
from tqdm import tqdm
from config import RAW_PDF_DIR, MARKDOWN_DIR

def convert_pdf_to_md(pdf_path: str, output_dir: str) -> None:
    """调用 MinerU CLI 转换单个 PDF。"""
    cmd = [
        "magic-pdf",
        "-p", pdf_path,
        "-o", output_dir,
        "-m", "auto"          # auto 模式：自动选择解析方式（OCR / 原生文本）
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[警告] 转换失败: {pdf_path}\n{result.stderr}")

def batch_convert(pdf_dir: str, md_dir: str) -> None:
    os.makedirs(md_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    print(f"共发现 {len(pdf_files)} 个 PDF，开始转换...")

    for fname in tqdm(pdf_files):
        pdf_path = os.path.join(pdf_dir, fname)
        # MinerU 会在 output_dir/<pdf_name>/ 下生成 .md 文件
        convert_pdf_to_md(pdf_path, md_dir)

    print("PDF → Markdown 转换完成！")

if __name__ == "__main__":
    batch_convert(RAW_PDF_DIR, MARKDOWN_DIR)
```

> **说明**：MinerU 输出的目录结构为 `markdown/<pdf_name>/<pdf_name>.md`，
> 后续摄入脚本会递归查找所有 `.md` 文件。

---

### 4.2 Markdown 解析与分块

法律文档的 Markdown 层级通常为：

```
# 中华人民共和国合同法          ← 法规名
## 第三章 合同的效力             ← 章节
### 第四十四条                   ← 条文（基本切分单位）
依法成立的合同，自成立时生效...
```

`scripts/ingest.py`（分块部分）

```python
import os
import glob
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document

# 按 Markdown 标题层级切分
HEADERS_TO_SPLIT = [
    ("#",   "law_name"),
    ("##",  "chapter"),
    ("###", "article"),
]

def find_md_files(md_dir: str) -> list[str]:
    """递归查找所有 Markdown 文件。"""
    return glob.glob(os.path.join(md_dir, "**", "*.md"), recursive=True)

def load_and_split(md_dir: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False   # 保留标题文本，方便引用时回显
    )

    all_docs: list[Document] = []
    md_files = find_md_files(md_dir)
    print(f"发现 {len(md_files)} 个 Markdown 文件")

    for md_path in md_files:
        source_name = os.path.basename(os.path.dirname(md_path))  # 用父目录名作为来源
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content)

        for chunk in chunks:
            text = chunk.page_content.strip()

            # 过滤过短的 chunk（通常是纯标题行或空行）
            if len(text) < 30:
                continue

            # 补充文件级元数据
            chunk.metadata["source_file"] = source_name
            chunk.metadata["full_path"]   = md_path

            # 构建用于检索展示的引用字符串
            law   = chunk.metadata.get("law_name", "")
            chap  = chunk.metadata.get("chapter",  "")
            art   = chunk.metadata.get("article",  "")
            chunk.metadata["citation"] = " · ".join(filter(None, [law, chap, art]))

            all_docs.append(chunk)

    print(f"共切分出 {len(all_docs)} 个有效 chunk")
    return all_docs
```

---

### 4.3 向量化写入 ChromaDB

`scripts/ingest.py`（写入部分）

```python
import chromadb
from langchain_community.vectorstores import Chroma
from rag.embeddings import get_embedding_model
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, MARKDOWN_DIR

def build_vectorstore(docs: list[Document]) -> Chroma:
    """将切分好的文档向量化并写入 ChromaDB。"""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding_model,
    )

    # 分批写入，避免单次请求超时或触发 API 限流
    batch_size = 100
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  向量化进度：{min(i + batch_size, total)}/{total}")

    print("ChromaDB 写入完成！")
    return vectorstore

def verify_vectorstore() -> None:
    """验证写入结果。"""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    col = client.get_collection(CHROMA_COLLECTION)
    print(f"ChromaDB 中共有 {col.count()} 条向量记录")

# ── 主流程 ──────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_and_split(MARKDOWN_DIR)
    build_vectorstore(docs)
    verify_vectorstore()
```

> **运行摄入脚本**：
> ```bash
> # 第一步：PDF → Markdown（一次性）
> python scripts/pdf_to_md.py
>
> # 第二步：Markdown → ChromaDB（一次性）
> python scripts/ingest.py
> ```

---

## 5. 检索层

### 5.1 SiliconFlow Embedding 封装

`rag/embeddings.py`

```python
from langchain_community.embeddings import OpenAIEmbeddings
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, EMBEDDING_MODEL

def get_embedding_model() -> OpenAIEmbeddings:
    """
    SiliconFlow 兼容 OpenAI 接口，直接复用 OpenAIEmbeddings，
    替换 base_url 和 api_key 即可。
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
    )
```

---

### 5.2 混合检索（向量 + BM25）

`rag/retriever.py`

```python
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever, Document

from rag.embeddings import get_embedding_model
from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    VECTOR_SEARCH_K, BM25_SEARCH_K,
    ENSEMBLE_WEIGHTS, RERANK_TOP_K,
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, RERANKER_MODEL
)

def build_ensemble_retriever(all_docs: list[Document]) -> EnsembleRetriever:
    """
    构建混合检索器：向量检索 + BM25 关键词检索，通过 EnsembleRetriever 融合。
    all_docs 为摄入时切分的完整文档列表，用于初始化 BM25 索引。
    """
    # 向量检索器
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    vectorstore = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embedding_model(),
    )
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": VECTOR_SEARCH_K}
    )

    # BM25 关键词检索器（法律条文编号精确匹配效果好）
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = BM25_SEARCH_K

    # RRF 融合：法律场景适当提高 BM25 权重
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=ENSEMBLE_WEIGHTS,      # [BM25, 向量]
    )
    return ensemble
```

---

### 5.3 Reranker 精排

SiliconFlow 的 Reranker 通过 HTTP 接口调用，这里封装为 LangChain 兼容的 `BaseRetriever`。

`rag/retriever.py`（续）

```python
import requests
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

class SiliconFlowReranker(BaseRetriever):
    """
    调用 SiliconFlow Reranker API 对召回结果精排，
    返回 Top-K 最相关文档。
    """
    base_retriever: EnsembleRetriever
    top_k: int = RERANK_TOP_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # Step 1: 混合检索召回候选文档
        candidates = self.base_retriever.get_relevant_documents(query)
        if not candidates:
            return []

        # Step 2: 调用 SiliconFlow Reranker
        payload = {
            "model": RERANKER_MODEL,
            "query": query,
            "documents": [doc.page_content for doc in candidates],
            "top_n": self.top_k,
            "return_documents": False,   # 只返回索引和分数，减少传输
        }
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{SILICONFLOW_BASE_URL}/rerank",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        # Step 3: 按 Reranker 分数排序，取 Top-K
        reranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        top_docs = [candidates[r["index"]] for r in reranked[: self.top_k]]
        return top_docs


def build_retriever(all_docs: list[Document]) -> SiliconFlowReranker:
    """对外暴露的工厂函数，返回带 Reranker 的完整检索器。"""
    ensemble = build_ensemble_retriever(all_docs)
    return SiliconFlowReranker(base_retriever=ensemble, top_k=RERANK_TOP_K)
```

---

## 6. 对话记忆与问题改写

### 6.1 对话记忆

`rag/memory.py`

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_models import ChatOpenAI
from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL,
    LLM_MODEL, MAX_TOKEN_LIMIT
)

def get_llm() -> ChatOpenAI:
    """获取 SiliconFlow LLM 实例（兼容 OpenAI 接口）。"""
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
        temperature=0.1,    # 法律问答场景，温度偏低，保证确定性
        max_tokens=2048,
    )

def build_memory(llm: ChatOpenAI) -> ConversationSummaryBufferMemory:
    """
    ConversationSummaryBufferMemory：
    - 保留最近若干轮的原始对话
    - 超过 MAX_TOKEN_LIMIT 后，对更早的历史自动做摘要压缩
    - 既保留上下文连贯性，又控制 context 长度
    """
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=MAX_TOKEN_LIMIT,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
```

---

### 6.2 多轮问题改写（指代消解）

多轮对话中，用户往往说「那这种情况下呢？」「他们需要承担什么责任？」
这类问题脱离上下文无法独立检索，需要先改写为完整问题。

`rag/chain.py`（问题改写部分）

```python
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

REWRITE_TEMPLATE = """你是一个法律问答助手。请根据以下对话历史，
将用户的最新问题改写为一个完整、独立、可以直接用于检索的问题。
要求：
- 消解所有代词（他、她、他们、这、那、此等）
- 补充必要的上下文（当事方、法律关系、事件背景）
- 不要回答问题，只输出改写后的问题

对话历史：
{chat_history}

用户最新问题：{question}

改写后的完整问题："""

rewrite_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=REWRITE_TEMPLATE,
)

def build_rewrite_chain(llm):
    """返回问题改写链。"""
    return rewrite_prompt | llm | StrOutputParser()
```

---

## 7. Prompt 设计与引用条文

`rag/prompt.py`

```python
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

SYSTEM_TEMPLATE = """你是一位专业的法律智能问答助手，专门负责解答中国法律相关问题。

## 回答规则
1. **只基于提供的法律条文作答**，不得凭空捏造法条内容。
2. **每一个核心观点必须引用对应条文**，引用格式：【来源】
3. 如果检索到的条文不足以回答问题，明确告知用户，并说明可能需要查阅哪类法律。
4. 回答语言简洁专业，避免歧义。
5. 涉及具体案件时，提示用户咨询专业律师。

## 参考条文
以下是与问题相关的法律条文，请基于这些内容作答：

{context}

## 对话历史
{chat_history}
"""

HUMAN_TEMPLATE = "{question}"

def build_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
    ])


def format_context(docs) -> str:
    """
    将检索到的文档格式化为带引用标记的上下文字符串，
    注入到 Prompt 的 {context} 占位符中。
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        citation = doc.metadata.get("citation", "未知来源")
        text     = doc.page_content.strip()
        parts.append(f"[条文{i}]【{citation}】\n{text}")
    return "\n\n".join(parts)
```

**引用效果示例**：

```
[条文1]【中华人民共和国合同法 · 第三章 合同的效力 · 第四十四条】
依法成立的合同，自成立时生效。法律、行政法规规定应当办理批准、登记等手续生效的，
依照其规定。

[条文2]【中华人民共和国合同法 · 第五章 合同的变更和转让 · 第七十七条】
当事人协商一致，可以变更合同...
```

---

## 8. LangChain 对话链组装

`rag/chain.py`

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from rag.memory  import get_llm, build_memory
from rag.prompt  import build_qa_prompt, format_context
from rag.retriever import build_retriever

class LegalQAChain:
    """
    法律智能问答链，封装：
    - 多轮记忆
    - 问题改写（指代消解）
    - 混合检索 + Reranker
    - 带引用的结构化回答
    """

    def __init__(self, all_docs: list[Document]):
        self.llm       = get_llm()
        self.memory    = build_memory(self.llm)
        self.retriever = build_retriever(all_docs)
        self.prompt    = build_qa_prompt()
        self.rewrite_chain = build_rewrite_chain(self.llm)

    def _rewrite_question(self, question: str) -> str:
        """结合对话历史改写问题。"""
        history = self.memory.load_memory_variables({}).get("chat_history", [])
        if not history:
            return question   # 首轮无需改写
        history_str = "\n".join(
            f"{'用户' if m.type == 'human' else '助手'}: {m.content}"
            for m in history
        )
        return self.rewrite_chain.invoke({
            "chat_history": history_str,
            "question": question,
        })

    def ask(self, question: str) -> dict:
        """
        核心问答方法。
        返回 {"answer": str, "sources": list[dict]}
        """
        # Step 1：问题改写
        rewritten_q = self._rewrite_question(question)

        # Step 2：混合检索 + Reranker
        docs = self.retriever.get_relevant_documents(rewritten_q)

        # Step 3：格式化上下文
        context = format_context(docs)

        # Step 4：获取对话历史
        chat_history = self.memory.load_memory_variables({}).get("chat_history", [])

        # Step 5：调用 LLM 生成回答
        messages = self.prompt.format_messages(
            context=context,
            chat_history=chat_history,
            question=question,          # Prompt 中展示原始问题，改写后的用于检索
        )
        response = self.llm(messages)
        answer = response.content

        # Step 6：保存本轮对话到记忆
        self.memory.save_context(
            {"input": question},
            {"answer": answer},
        )

        # Step 7：提取引用来源
        sources = [
            {
                "citation": doc.metadata.get("citation", ""),
                "source_file": doc.metadata.get("source_file", ""),
                "snippet": doc.page_content[:120] + "...",
            }
            for doc in docs
        ]

        return {"answer": answer, "sources": sources, "rewritten_query": rewritten_q}
```

---

## 9. 主程序入口

`main.py`

```python
"""
法律智能问答系统 — 命令行交互入口
用法：python main.py
"""
import os
from scripts.ingest import load_and_split
from rag.chain import LegalQAChain
from config import MARKDOWN_DIR, CHROMA_PERSIST_DIR

def check_chroma_exists() -> bool:
    """检查 ChromaDB 是否已完成摄入。"""
    return os.path.exists(CHROMA_PERSIST_DIR) and \
           len(os.listdir(CHROMA_PERSIST_DIR)) > 0

def main():
    print("=" * 60)
    print("  法律智能问答系统")
    print("=" * 60)

    if not check_chroma_exists():
        print("[错误] 未检测到 ChromaDB 数据，请先运行摄入脚本：")
        print("  python scripts/pdf_to_md.py")
        print("  python scripts/ingest.py")
        return

    print("正在加载文档索引...")
    # 加载文档用于 BM25 初始化（ChromaDB 已持久化，向量无需重新计算）
    all_docs = load_and_split(MARKDOWN_DIR)

    print("正在初始化问答链...")
    qa_chain = LegalQAChain(all_docs)

    print("\n系统就绪！输入 'quit' 或 'exit' 退出。\n")
    print("-" * 60)

    while True:
        try:
            question = input("您的问题：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "退出"):
            print("再见！")
            break

        result = qa_chain.ask(question)

        print("\n【回答】")
        print(result["answer"])

        if result["sources"]:
            print("\n【引用条文】")
            for i, src in enumerate(result["sources"], 1):
                print(f"  [{i}] {src['citation']}")
                print(f"      {src['snippet']}")

        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
```

---

## 10. 关键注意事项与调优建议

### 10.1 分块策略调优

| 场景 | 建议 |
|------|------|
| 条文内容极短（< 50 字） | 合并相邻条文为一个 chunk |
| 条文内容过长（> 800 字） | 在"款"级别再次切分，保留"第X条第X款"编号 |
| 表格类条文（如罚款标准） | MinerU 会转为 Markdown 表格，整表作为一个 chunk |
| 条文有附件/附录 | 附录单独建 collection，检索时可选择是否跨 collection |

### 10.2 混合检索权重调整

```python
# 以下场景建议调整 ENSEMBLE_WEIGHTS：
# 用户倾向于输入条文编号（如"第44条"）→ 提高 BM25 权重
ENSEMBLE_WEIGHTS = [0.6, 0.4]   # BM25 : 向量

# 用户倾向于语义描述（如"合同解除的条件"）→ 提高向量权重
ENSEMBLE_WEIGHTS = [0.3, 0.7]   # BM25 : 向量
```

### 10.3 Reranker 阈值过滤

```python
# 可在 SiliconFlowReranker._get_relevant_documents 中增加分数过滤
MIN_RELEVANCE_SCORE = 0.3
reranked = [r for r in results if r["relevance_score"] >= MIN_RELEVANCE_SCORE]
```

### 10.4 ChromaDB 元数据过滤

用户明确指定法规时，可加入元数据过滤缩小检索范围：

```python
# 示例：只检索《合同法》相关条文
vector_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": VECTOR_SEARCH_K,
        "filter": {"law_name": "中华人民共和国合同法"}
    }
)
```

### 10.5 SiliconFlow API 限流应对

```python
import time

# 摄入时分批请求，批次间加延迟
for i in range(0, total, batch_size):
    batch = docs[i : i + batch_size]
    vectorstore.add_documents(batch)
    time.sleep(1)   # 避免触发 QPS 限制
```

### 10.6 生产部署建议

- **BM25 索引持久化**：`BM25Retriever` 默认内存存储，重启后需重新从文档构建。可用 `pickle` 序列化保存到磁盘，加速冷启动。
- **ChromaDB 备份**：定期备份 `./data/chroma_db/` 目录，防止数据丢失。
- **日志记录**：记录每次检索的改写问题、召回文档、Reranker 分数，便于后续优化。
- **新文档增量摄入**：新 PDF 只需经过 MinerU → 分块 → `vectorstore.add_documents()` 即可增量写入，不影响已有数据。

---

*文档版本：v1.0 | 基于 LangChain 0.2.x · ChromaDB 0.5.x · SiliconFlow API*
