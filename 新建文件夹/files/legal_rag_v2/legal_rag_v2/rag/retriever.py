"""
检索层：混合检索（向量 + BM25）+ SiliconFlow Reranker 精排。
BM25 索引持久化到磁盘，启动时直接加载，无需重新读取原始文档。
"""
import os
import sys
import pickle
import requests
import chromadb
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from rag.embeddings import get_embedding_model
from config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    VECTOR_SEARCH_K, BM25_SEARCH_K,
    ENSEMBLE_WEIGHTS, RERANK_TOP_K,
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, RERANKER_MODEL,
)

# BM25 索引持久化路径（与 ChromaDB 放在同一目录）
BM25_INDEX_PATH = os.path.join(CHROMA_PERSIST_DIR, "bm25_index.pkl")


def save_bm25_index(docs: list) -> BM25Retriever:
    """
    从文档列表构建 BM25 索引并持久化到磁盘。
    在数据摄入阶段（ingest 脚本）调用一次即可，之后启动无需重建。
    """
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = BM25_SEARCH_K
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 索引已保存：{BM25_INDEX_PATH}")
    return bm25_retriever


def load_bm25_index() -> BM25Retriever:
    """
    从磁盘加载 BM25 索引。
    若索引文件不存在则抛出异常，提示先运行摄入脚本。
    """
    if not os.path.exists(BM25_INDEX_PATH):
        raise FileNotFoundError(
            f"未找到 BM25 索引：{BM25_INDEX_PATH}\n"
            "请先运行摄入脚本：\n"
            "  python scripts/ingest.py       # Markdown 文档\n"
            "  python scripts/ingest_docx.py  # Word 文档"
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        retriever = pickle.load(f)
    print("BM25 索引加载完成")
    return retriever


def build_ensemble_retriever() -> EnsembleRetriever:
    """
    构建混合检索器。
    向量索引从 ChromaDB 加载，BM25 索引从磁盘加载，均无需原始文档。
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

    # BM25 检索器（从磁盘加载）
    bm25_retriever = load_bm25_index()

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=ENSEMBLE_WEIGHTS,
    )


class SiliconFlowReranker(BaseRetriever):
    """调用 SiliconFlow Reranker API 对混合检索结果精排。"""
    base_retriever: EnsembleRetriever = Field(...)
    top_k: int = Field(default=RERANK_TOP_K)
    min_relevance_score: float = Field(default=0.1)

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        payload = {
            "model": RERANKER_MODEL,
            "query": query,
            "documents": [doc.page_content for doc in candidates],
            "top_n": self.top_k,
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                f"{SILICONFLOW_BASE_URL}/rerank",
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except requests.RequestException as e:
            print(f"[警告] Reranker 调用失败，降级使用混合检索结果：{e}")
            return candidates[: self.top_k]

        reranked = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        reranked = [r for r in reranked if r["relevance_score"] >= self.min_relevance_score]
        return [candidates[r["index"]] for r in reranked[: self.top_k]]


def build_retriever() -> SiliconFlowReranker:
    """
    对外暴露的工厂函数。
    启动时直接从磁盘加载所有索引，不需要传入文档列表。
    """
    ensemble = build_ensemble_retriever()
    return SiliconFlowReranker(base_retriever=ensemble, top_k=RERANK_TOP_K)
