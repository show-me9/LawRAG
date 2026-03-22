"""
数据摄入主脚本：Markdown → 分块 → 向量化 → 写入 ChromaDB
本脚本为一次性处理，完成后数据持久化在 chroma_db 目录中。

用法：
    python scripts/ingest.py
"""
import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from tqdm import tqdm

# LangChain 1.2.x: text_splitters 从 langchain_text_splitters 导入
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# LangChain 1.2.x: Chroma 从 langchain_chroma 导入
from langchain_chroma import Chroma

from config import (
    MARKDOWN_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION
)
from rag.embeddings import get_embedding_model
from rag.retriever import save_bm25_index

# ── 按 Markdown 标题层级切分 ──────────────────────────────
HEADERS_TO_SPLIT = [
    ("#",   "law_name"),
    ("##",  "chapter"),
    ("###", "article"),
]


def find_md_files(md_dir: str) -> list:
    """递归查找所有 Markdown 文件。"""
    return glob.glob(os.path.join(md_dir, "**", "*.md"), recursive=True)


def load_and_split(md_dir: str) -> list:
    """加载 Markdown 文件并按法律条文结构切分。"""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False
    )

    all_docs = []
    md_files = find_md_files(md_dir)

    if not md_files:
        print(f"[错误] 未在 {md_dir} 中找到 Markdown 文件")
        print("请先运行：python scripts/pdf_to_md.py")
        return []

    print(f"发现 {len(md_files)} 个 Markdown 文件，开始解析切分...")

    for md_path in tqdm(md_files, desc="解析进度"):
        # MinerU 输出结构：markdown/<pdf_name>/<pdf_name>.md
        source_name = os.path.basename(os.path.dirname(md_path))

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content)

        for chunk in chunks:
            text = chunk.page_content.strip()

            # 过滤过短的 chunk（纯标题行或空行）
            if len(text) < 30:
                continue

            # 补充文件级元数据
            chunk.metadata["source_file"] = source_name
            chunk.metadata["full_path"]   = md_path

            # 构建引用字符串，用于回答时展示来源
            law  = chunk.metadata.get("law_name", "")
            chap = chunk.metadata.get("chapter",  "")
            art  = chunk.metadata.get("article",  "")
            chunk.metadata["citation"] = " · ".join(filter(None, [law, chap, art]))

            all_docs.append(chunk)

    print(f"共切分出 {len(all_docs)} 个有效 chunk")
    return all_docs


def build_vectorstore(docs: list) -> Chroma:
    """将切分好的文档向量化并写入 ChromaDB。"""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    # LangChain 1.2.x + ChromaDB 1.x：使用 chromadb.PersistentClient
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding_model,
    )

    # 分批写入，避免触发 SiliconFlow API 限流
    batch_size = 100
    total = len(docs)
    print(f"\n开始向量化写入（共 {total} 个 chunk，每批 {batch_size} 个）...")

    for i in range(0, total, batch_size):
        batch = docs[i: i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  进度：{min(i + batch_size, total)}/{total}")
        time.sleep(0.5)  # 避免触发限流

    print("ChromaDB 写入完成！")
    return vectorstore


def verify_vectorstore() -> None:
    """验证写入结果。"""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        col = client.get_collection(CHROMA_COLLECTION)
        print(f"验证通过：ChromaDB 中共有 {col.count()} 条向量记录")
    except Exception as e:
        print(f"[错误] 验证失败：{e}")


# ── 主流程 ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  数据摄入脚本")
    print("=" * 50)

    docs = load_and_split(MARKDOWN_DIR)
    if not docs:
        sys.exit(1)

    build_vectorstore(docs)
    save_bm25_index(docs)       # 同步保存 BM25 索引到磁盘
    verify_vectorstore()
    print("\n摄入完成！可以运行 main.py 开始问答。")
