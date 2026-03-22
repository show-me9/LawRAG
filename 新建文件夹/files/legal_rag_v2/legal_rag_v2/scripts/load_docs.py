"""
统一文档加载器。
根据实际存在的文件类型，自动选择加载 Markdown 或 Word 文档，
返回用于 BM25 初始化的 Document 列表。

使用场景：main.py 启动时调用，无需关心底层文档格式。
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from config import MARKDOWN_DIR


def load_all_docs(prefer_docx_dir: str = "./data/raw_docx") -> list:
    """
    优先加载 Word 文档（如果存在），否则加载 Markdown。
    返回 Document 列表，用于 BM25 检索器初始化。

    Args:
        prefer_docx_dir: Word 文档目录路径。
    """
    docx_files = glob.glob(
        os.path.join(prefer_docx_dir, "**", "*.docx"), recursive=True
    )
    md_files = glob.glob(
        os.path.join(MARKDOWN_DIR, "**", "*.md"), recursive=True
    )

    if docx_files:
        print(f"检测到 {len(docx_files)} 个 Word 文档，使用 Word 加载器...")
        from scripts.ingest_docx import ingest_docx_files
        return ingest_docx_files(docx_files)

    elif md_files:
        print(f"检测到 {len(md_files)} 个 Markdown 文件，使用 Markdown 加载器...")
        from scripts.ingest import load_and_split
        return load_and_split(MARKDOWN_DIR)

    else:
        print("[错误] 未找到任何可加载的文档（.docx 或 .md）")
        return []
