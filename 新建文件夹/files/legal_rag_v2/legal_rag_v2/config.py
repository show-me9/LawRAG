import os
from dotenv import load_dotenv

load_dotenv()

# ── SiliconFlow ──────────────────────────────────────────
SILICONFLOW_API_KEY  = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# 模型名称（按需替换为 SiliconFlow 上实际可用的模型）
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
LLM_MODEL       = "deepseek-ai/DeepSeek-V2.5"

# ── ChromaDB ─────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./data/chroma_db"
CHROMA_COLLECTION  = "legal_docs"

# ── 检索参数 ─────────────────────────────────────────────
VECTOR_SEARCH_K  = 20     # 向量检索召回数
BM25_SEARCH_K    = 20     # BM25 召回数
RERANK_TOP_K     = 5      # Reranker 最终保留数
ENSEMBLE_WEIGHTS = [0.4, 0.6]  # [BM25权重, 向量权重]

# ── 对话记忆 ─────────────────────────────────────────────
MAX_HISTORY_TURNS = 10    # 保留最近 N 轮完整对话，超出后压缩

# ── 路径 ─────────────────────────────────────────────────
RAW_PDF_DIR  = "./data/raw_pdfs"
MARKDOWN_DIR = "./data/markdown"
