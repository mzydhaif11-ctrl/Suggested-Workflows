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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine

    try:
        rag_engine = await run_in_threadpool(
            GroundedRAGEngine,
            auto_index=True,
        )
        logger.info("Mawjah RAG initialized successfully.")

    except Exception:
        logger.exception("Failed to initialize Mawjah RAG.")
        rag_engine = None

    yield


app = FastAPI(
    title="Mawjah AI Platform",
    description="Mawjah integration layer for Gemini, OpenAI, and RAG",
    version="1.1.0",
    lifespan=lifespan,
)


allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# Existing integrations
app.include_router(prompt_studio_router)
app.include_router(ai_bridge_router)


# RAG request model
class RAGQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="سؤال المستخدم",
    )


# System health check
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "application": "Mawjah",
        "version": "1.1.0",
        "rag": "ready" if rag_engine is not None else "unavailable",
    }


# RAG question endpoint
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
            detail="تعذرت معالجة السؤال. يرجى المحاولة لاحقًا.",
        )
