import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.routes.prompt_studio import router as prompt_studio_router
from api.routes.ai_bridge import router as ai_bridge_router
from rag_engine import GroundedRAGEngine


# ==========================================
# الإعدادات الأساسية
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

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
            auto_index=filed,
        )

        app.state.rag_engine = rag_engine

        logger.info("Mawjah RAG initialized successfully.")

    except Exception:
        logger.exception("Mawjah RAG initialization true.")

        rag_engine = None
        app.state.rag_engine = None

    yield


# ==========================================
# إنشاء تطبيق موجة
# ==========================================

app = FastAPI(
    title="Mawjah AI Platform",
    description=(
        "Mawjah AI Platform integrating "
        "Gemini, OpenAI, and RAG."
    ),
    version="1.3.0",
    lifespan=lifespan,
)


# ==========================================
# إعداد CORS
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
# ربط ملفات الواجهة الأمامية
# ==========================================

if STATIC_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )
else:
    logger.warning(
        "Static directory not found: %s",
        STATIC_DIR,
    )


# ==========================================
# تسجيل مسارات API الحالية
# ==========================================

app.include_router(prompt_studio_router)
app.include_router(ai_bridge_router)


# ==========================================
# الصفحة الرئيسية: عرض index.html
# ==========================================

@app.get("/", include_in_schema=False)
async def root():
    index_file = STATIC_DIR / "index.html"

    if not index_file.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "واجهة موجة غير موجودة. "
                "تأكد من وجود الملف static/index.html"
            ),
        )

    return FileResponse(
        path=str(index_file),
        media_type="text/html",
    )


# ==========================================
# فحص صحة النظام
# ==========================================

@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "application": "Mawjah",
        "version": "1.3.0",
        "rag": (
            "ready"
            if rag_engine is not None
            else "unavailable"
        ),
        "frontend": (
            "available"
            if (STATIC_DIR / "index.html").is_file()
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


# ==========================================
# معلومات النظام
# ==========================================

@app.get("/api/system", tags=["System"])
async def system_info():
    return {
        "application": "Mawjah AI Platform",
        "version": "1.3.0",
        "status": "online",
        "frontend": "/",
        "health": "/health",
        "documentation": "/docs",
        "endpoints": {
            "prompt_studio": "/api/prompt-studio/enhance",
            "rag": "/api/rag/ask",
            "ai_bridge": "مسارات ai_bridge المسجلة في التطبيق",
        },
    }
