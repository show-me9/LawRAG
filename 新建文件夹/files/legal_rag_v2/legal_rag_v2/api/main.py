"""
FastAPI 后端服务入口。

启动方式：
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

接口文档：
    http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import documents, chat, settings, health

app = FastAPI(
    title="法律智能问答系统 API",
    description="基于 LangChain + SiliconFlow + ChromaDB 的法律 RAG 系统",
    version="1.0.0",
)

# ── CORS（允许 Vue3 开发服务器跨域请求）────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 注册路由 ──────────────────────────────────────────────
app.include_router(health.router,    prefix="/api")
app.include_router(documents.router, prefix="/api/documents")
app.include_router(chat.router,      prefix="/api/chat")
app.include_router(settings.router,  prefix="/api/settings")
