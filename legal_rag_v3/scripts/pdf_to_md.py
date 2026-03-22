"""
批量将 raw_pdfs/ 下的 PDF 用 MinerU 转换为 Markdown。
本脚本为一次性离线处理。

用法：
    python scripts/pdf_to_md.py
"""
import os
import sys
import subprocess
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_PDF_DIR, MARKDOWN_DIR


def convert_pdf_to_md(pdf_path: str, output_dir: str) -> bool:
    """调用 MinerU CLI 转换单个 PDF，返回是否成功。"""
    cmd = [
        "magic-pdf",
        "-p", pdf_path,
        "-o", output_dir,
        "-m", "auto"   # auto: 自动判断使用 OCR 还是原生文本解析
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n[警告] 转换失败: {os.path.basename(pdf_path)}")
        print(f"  原因: {result.stderr.strip()[:200]}")
        return False
    return True


def batch_convert(pdf_dir: str, md_dir: str) -> None:
    os.makedirs(md_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[错误] 未在 {pdf_dir} 中找到 PDF 文件")
        return

    print(f"共发现 {len(pdf_files)} 个 PDF，开始转换...\n")
    success, failed = 0, 0

    for fname in tqdm(pdf_files, desc="转换进度"):
        pdf_path = os.path.join(pdf_dir, fname)
        if convert_pdf_to_md(pdf_path, md_dir):
            success += 1
        else:
            failed += 1

    print(f"\n转换完成：成功 {success} 个，失败 {failed} 个")
    print(f"输出目录：{md_dir}")


if __name__ == "__main__":
    batch_convert(RAW_PDF_DIR, MARKDOWN_DIR)
