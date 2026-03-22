"""
法律智能问答系统 — 命令行交互入口

使用前请确保已完成数据摄入（二选一）：
    python scripts/ingest_docx.py  # Word 文档
    python scripts/ingest.py       # Markdown 文档（需先运行 pdf_to_md.py）

启动问答（直接从磁盘加载索引，无需读取原始文档）：
    python main.py
"""
import os
import sys


def check_environment() -> bool:
    """检查运行环境是否就绪。"""
    ok = True
    from config import SILICONFLOW_API_KEY, CHROMA_PERSIST_DIR

    if not SILICONFLOW_API_KEY:
        print("[错误] 未配置 SILICONFLOW_API_KEY")
        print("  请复制 .env.example 为 .env 并填入 API Key")
        ok = False

    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        print("[错误] 未检测到索引数据，请先运行摄入脚本：")
        print("  python scripts/ingest_docx.py  # Word 文档")
        print("  python scripts/ingest.py       # Markdown 文档")
        ok = False

    from rag.retriever import BM25_INDEX_PATH
    if not os.path.exists(BM25_INDEX_PATH):
        print("[错误] 未检测到 BM25 索引，请先运行摄入脚本")
        ok = False

    return ok


def print_banner():
    print("=" * 60)
    print("  法律智能问答系统")
    print("  基于 LangChain 1.2.x + SiliconFlow + ChromaDB")
    print("=" * 60)
    print()


def print_help():
    print("命令说明：")
    print("  直接输入问题  → 进行问答")
    print("  /clear        → 清空对话记忆，开始新对话")
    print("  /verbose      → 切换详细模式（显示改写查询和条文片段）")
    print("  /help         → 显示此帮助")
    print("  quit / exit   → 退出程序")
    print()


def main():
    print_banner()

    if not check_environment():
        sys.exit(1)

    # 直接加载索引，不读取原始文档
    from rag.chain import LegalQAChain
    qa_chain = LegalQAChain()

    print_help()
    print("-" * 60)

    verbose = False

    while True:
        try:
            user_input = input("您的问题：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "退出"):
            print("再见！")
            break
        if user_input == "/clear":
            qa_chain.clear_memory()
            continue
        if user_input == "/verbose":
            verbose = not verbose
            print(f"详细模式：{'开启' if verbose else '关闭'}\n")
            continue
        if user_input == "/help":
            print_help()
            continue

        try:
            result = qa_chain.ask(user_input)
        except Exception as e:
            print(f"\n[错误] 问答失败：{e}\n")
            continue

        if verbose and result["rewritten_query"] != user_input:
            print(f"\n[改写查询] {result['rewritten_query']}")

        print("\n【回答】")
        print(result["answer"])

        if result["sources"]:
            print("\n【引用条文】")
            for i, src in enumerate(result["sources"], 1):
                print(f"  [{i}] {src['citation']}")
                if verbose:
                    print(f"       {src['snippet']}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
