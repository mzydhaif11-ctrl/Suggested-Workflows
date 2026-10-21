"""
موجة البيان | Mowjn AI-Bayan
خادم FastAPI — يدعم وضعين:
1. مع نموذج لغوي (علّام عبر HUMAIN Node)
2. بدون نموذج — بحث محلي فقط من قاعدة المعرفة 🔒
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

# استيراد المكونات المحلية
from adaptive_memory_engine import AdaptiveMemoryEngine
from rag_engine import RAGEngine
from telemetry import TelemetryLogger

app = FastAPI(title="موجة البيان", description="نظام ذاكرة سياقية تكيفية ومحرك RAG محلي")

# تحميل الإعدادات
TOP_K_MEMORIES = int(os.getenv("TOP_K_MEMORIES", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "mowjn_knowledge")
DOCS_DIR = os.getenv("MAWJAH_DOCS_DIR", "docs/")
LLM_MODE = os.getenv("LLM_MODE", "none").lower()  # "none" = بدون نموذج


class QueryRequest(BaseModel):
    message: str


class QueryResponse(BaseModel):
    response: str
    sources: list
    memories_used: int
    rag_results: int


# تهيئة المكونات عند بدء التشغيل
memory_engine = AdaptiveMemoryEngine(collection_name=CHROMA_COLLECTION)
rag_engine = RAGEngine(docs_dir=DOCS_DIR)
telemetry = TelemetryLogger()


@app.get("/")
async def serve_index():
    """تقديم الصفحة الرئيسية"""
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """معالجة سؤال المستخدم"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="الرسالة فارغة")

    try:
        # 1. استرجاع الذكريات السابقة
        memories = memory_engine.retrieve(
            query=request.message,
            top_k=TOP_K_MEMORIES,
            threshold=SIMILARITY_THRESHOLD,
        )

        # 2. بحث في قاعدة المعرفة الثابتة (RAG)
        rag_results = rag_engine.search(query=request.message)

        # بناء الرد حسب الوضع
        if LLM_MODE == "none":
            # === وضع البحث المحلي فقط ===
            response_text, sources = build_local_response(
                request.message, memories, rag_results
            )
        else:
            # === وضع النموذج اللغوي (يتطلب API Key) ===
            from llm_client import get_llm_client
            llm_client = get_llm_client()

            parts = []
            if memories:
                mem_text = "\n".join([f"- {m['text']}" for m in memories])
                parts.append(f"سياق سابق:\n{mem_text}")
            if rag_results:
                rag_text = "\n".join([f"- {r['text']}" for r in rag_results])
                parts.append(f"\nمعلومات مرجعية:\n{rag_text}")

            enriched_prompt = "\n\n".join(parts) + f"\n\nالسؤال: {request.message}"

            system_msg = (
                "أنت مساعد ذكي عربي يُعرف بموجة البيان. "
                "استخدم السياق المقدم للإجابة بدقة ووضوح."
            )
            response_text = llm_client.generate(system_msg, enriched_prompt)
            sources = [r.get('source', '') for r in rag_results]

        # حفظ المحادثة
        memory_engine.store(request.message, response_text)
        telemetry.log_interaction(len(memories), len(rag_results), len(response_text))

        return QueryResponse(
            response=response_text,
            sources=sources,
            memories_used=len(memories),
            rag_results=len(rag_results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def build_local_response(query: str, memories: list, rag_results: list) -> tuple:
    """بناء رد من نتائج البحث المحلي فقط — بدون نموذج لغوي"""
    lines = []
    sources = []

    # عرض نتائج RAG
    if rag_results:
        lines.append("📚 نتائج البحث من قاعدة المعرفة:")
        lines.append("")
        for i, result in enumerate(rag_results, 1):
            text = result.get('text', '')[:500]  # أول 500 حرف
            source = result.get('source', 'غير معروف')
            score = result.get('score', 0)
            lines.append(f"**{i}. {source}** (التشابه: {score:.2%})")
            lines.append(text)
            lines.append("---")
            sources.append(source)
    else:
        lines.append("⚠️ لم أجد نتائج مطابقة في قاعدة المعرفة.")
        lines.append("")
        lines.append("💡 نصيحة: أضف ملفات `.md` أو `.txt` في مجلد `docs/` لتوسيع قاعدة المعرفة.")

    # عرض الذكريات السابقة
    if memories:
        lines.append("")
        lines.append("🧠 سياق المحادثة السابق:")
        for m in memories:
            lines.append(f"• {m['text'][:300]}")

    return "\n\n".join(lines), sources


@app.get("/api/status")
async def get_status():
    """حالة النظام"""
    return {
        "status": "running",
        "mode": LLM_MODE,
        "memory_count": memory_engine.stats()["total"],
        "docs_dir": DOCS_DIR,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
