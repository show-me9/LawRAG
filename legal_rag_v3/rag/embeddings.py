"""
SiliconFlow Embedding 封装。
LangChain 1.2.x: OpenAIEmbeddings 从 langchain_openai 导入。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain 1.2.x: 从 langchain_openai 导入
from langchain_openai import OpenAIEmbeddings
from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, EMBEDDING_MODEL


def get_embedding_model() -> OpenAIEmbeddings:
    """
    返回 SiliconFlow Embedding 模型实例。
    SiliconFlow 兼容 OpenAI 接口，通过替换 base_url 调用。
    """
    if not SILICONFLOW_API_KEY:
        raise ValueError(
            "未找到 SILICONFLOW_API_KEY，请在 .env 文件中配置。"
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SILICONFLOW_API_KEY,          # LangChain 1.x 用 api_key
        base_url=SILICONFLOW_BASE_URL,         # LangChain 1.x 用 base_url
    )
