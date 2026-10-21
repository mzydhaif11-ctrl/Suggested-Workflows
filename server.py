"""
موجة البيان | Mowjn AI-Bayan
خادم FastAPI — يربط الواجهة الأمامية بالمحرك المحلي + علّام
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys

# استيراد المكونات المحلية
from adaptive_memory_engine import AdaptiveMemoryEngine
from rag_engine import RAGEngine
from llm_client import AllamClient
from telemetry import TelemetryLogger

app = FastAPI(title="موجة البيان", description="نظام ذاكرة سياقية تكيفية ومحرك RAG محلي")

# تحميل الإعدادات
TOP_K_MEMORIES = int(os.getenv("TOP_K_MEMORIES", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "mowjn_knowledge")
DOCS_DIR = os.getenv("MAWJAH_DOCS_DIR", "docs/")


class QueryRequest(BaseModel):
    message: str


class QueryResponse(BaseModel):
    response: str
    memories_used: int
    rag_results: int


# تهيئة المكونات عند بدء التشغيل
memory_engine = AdaptiveMemoryEngine(collection_name=CHROMA_COLLECTION)
rag_engine = RAGEngine(docs_dir=DOCS_DIR)
llm_client = AllamClient()
telemetry = TelemetryLogger()


@app.get("/")
async def serve_index():
    """تقديم الصفحة الرئيسية"""
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """معالجة سؤال المستخدم وإرجاع الرد"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="الرسالة فارغة")

    try:
        # 1. استرجاع الذكريات
        memories = memory_engine.retrieve(
            query=request.message,
            top_k=TOP_K_MEMORIES,
            threshold=SIMILARITY_THRESHOLD,
        )

        # 2. بحث RAG
        rag_results = rag_engine.search(query=request.message)

        # 3. بناء السياق المثرى
        parts = []
        if memories:
            mem_text = "\n".join([f"- {m['text']}" for m in memories])
            parts.append(f"سياق سابق:\n{mem_text}")
        if rag_results:
            rag_text = "\n".join([f"- {r['text']}" for r in rag_results])
            parts.append(f"\nمعلومات مرجعية:\n{rag_text}")

        enriched_prompt = "\n\n".join(parts) + f"\n\nالسؤال: {request.message}"

        # 4. إرسال لعلّام
        system_msg = (
            "أنت مساعد ذكي عربي يُعرف بموجة البيان. "
            "تتحدث بالعربية الفصحى السليمة وتفهّم اللهجات المختلفة. "
            "استخدم السياق المقدم للإجابة بدقة ووضوح."
        )
        response = llm_client.generate(system_msg, enriched_prompt)

        # 5. حفظ + Telemetry
        memory_engine.store(request.message, response)
        telemetry.log_interaction(len(memories), len(rag_results), len(response))

        return QueryResponse(
            response=response,
            memories_used=len(memories),
            rag_results=len(rag_results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """حالة النظام"""
    return {
        "status": "running",
        "memory_count": memory_engine.stats()["total"],
        "model": os.getenv("ALLAM_MODEL", "allam-34b"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
