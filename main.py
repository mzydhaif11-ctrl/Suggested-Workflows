import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.routes.prompt_studio import router as prompt_studio_router
from api.routes.ai_bridge import router as ai_bridge_router
from rag_engine import GroundedRAGEngine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_engine: Optional[GroundedRAGEngine] = None


# ==========================================
# تهيئة محرك المعرفة
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine

    try:
        rag_engine = await run_in_threadpool(
            GroundedRAGEngine,
            auto_index=True,
        )

        app.state.rag_engine = rag_engine

        logger.info("Mawjah RAG initialized.")

    except Exception:
        logger.exception("Mawjah RAG initialization failed.")
        rag_engine = None
        app.state.rag_engine = None

    yield


# ==========================================
# إنشاء تطبيق موجة
# ==========================================

app = FastAPI(
    title="Mawjah AI Platform",
    description=(
        "Mawjah backend for AI integrations "
        "and retrieval-augmented generation."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


# ==========================================
# إعداد الاتصال بالواجهة الأمامية
# ==========================================

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Content-Type",
        "Authorization",
    ],
)


# ==========================================
# تسجيل المسارات الحالية
# ==========================================

app.include_router(prompt_studio_router)
app.include_router(ai_bridge_router)


# ==========================================
# الصفحة الرئيسية للـ Backend
# ==========================================

@app.get("/", tags=["System"])
async def root():
    return {
        "application": "Mawjah AI Platform",
        "status": "online",
        "version": "1.2.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "prompt_studio": "/api/prompt-studio/enhance",
            "rag": "/api/rag/ask",
        },
    }


# ==========================================
# فحص صحة النظام
# ==========================================

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "application": "Mawjah",
        "version": "1.2.0",
        "rag": (
            "ready"
            if rag_engine is not None
            else "unavailable"
        ),
    }


# ==========================================
# نموذج استقبال الأسئلة
# ==========================================

class RAGQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="سؤال المستخدم",
    )


# ==========================================
# واجهة السؤال والبحث في المعرفة
# ==========================================

@app.post("/api/rag/ask", tags=["RAG"])
async def ask_rag(payload: RAGQueryRequest):
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="خدمة المعرفة غير متاحة حاليًا.",
        )

    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="يرجى إدخال سؤال.",
        )

    try:
        result = await run_in_threadpool(
            rag_engine.answer_query,
            query,
        )

        return result

    except Exception:
        logger.exception("RAG request failed.")

        raise HTTPException(
            status_code=500,
            detail="تعذرت معالجة السؤال حاليًا.",
        )
