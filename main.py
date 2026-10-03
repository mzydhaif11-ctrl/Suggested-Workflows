import os
from typing import List, Optional
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from rag_engine import GroundedRAGEngine

# ✅ جديد: عميل OpenAI للاتصال بـ DeepSeek
from openai import OpenAI

app = FastAPI(
    title="منصة موجة البيان للدعم الفني",
    description="نظام دعم فني يعتمد على وثائق Gemini API الرسمية ومحرك RAG مع دعم ChromaDB Cloud",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# استدعاء الإعدادات من سيرفر Render بأمان
# ==========================================
chroma_url = os.environ.get("CHROMA_URL")
chroma_api_key = os.environ.get("CHROMA_API_KEY")
chroma_tenant = os.environ.get("CHROMA_TENANT")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
gemini_api_key_env = os.environ.get("GEMINI_API_KEY")

chroma_client = None
chroma_collection = None
deepseek_client = None

# ✅ تهيئة عميل DeepSeek إن وُجد المفتاح
if deepseek_api_key:
    try:
        deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        print("تم تهيئة عميل DeepSeek بنجاح.")
    except Exception as e:
        print(f"تعذر تهيئة DeepSeek: {e}")

# اتصال ChromaDB Cloud
if chroma_url and chroma_api_key:
    try:
        connect_kwargs = {
            "host": chroma_url,
            "headers": {"Authorization": f"Bearer {chroma_api_key}"}
        }
        if chroma_tenant:
            connect_kwargs["tenant"] = chroma_tenant
            connect_kwargs["database"] = "default_database"

        chroma_client = chromadb.HttpClient(**connect_kwargs)
        chroma_collection = chroma_client.get_or_create_collection(name="tech_advisor_collection")
        print("تم الاتصال بـ ChromaDB Cloud بنجاح.")
    except Exception as e:
        print(f"تعذر الاتصال بـ ChromaDB: {e}")

# تهيئة محرك RAG
engine = GroundedRAGEngine()

# ==========================================
# نماذج البيانات
# ==========================================
class ChatRequest(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = "deepseek"   # ✅ جديد: اختيار المزود

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# ==========================================
# ✅ دالة مساعدة: بناء سياق RAG من ChromaDB
# ==========================================
def build_rag_context(query: str, top_k: int = 5) -> str:
    try:
        results = engine.search(query, top_k=top_k)
        if not results:
            return ""
        chunks = []
        for r in results:
            text = r.get("text") or r.get("content") or r.get("document") or ""
            if text:
                chunks.append(text)
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        print(f"خطأ في بحث RAG: {e}")
        return ""

# ==========================================
# ✅ دالة جديدة: استدعاء DeepSeek
# ==========================================
def ask_deepseek(query: str, context: str = "") -> str:
    if not deepseek_client:
        raise HTTPException(
            status_code=500,
            detail="مفتاح DEEPSEEK_API_KEY غير موجود في إعدادات Render"
        )

    system_prompt = (
        "أنت مستشار تقني متخصص في Gemini API وDeepSeek ونماذج الذكاء الاصطناعي. "
        "أجب بالعربية الفصحى، بدقة، وبأسلوب منظم. "
        "إذا وُجد سياق مرجعي أدناه فاعتمد عليه في إجابتك، وإذا لم يكن كافيًا فاذكر ذلك."
    )

    user_content = query
    if context:
        user_content = f"السياق المرجعي:\n{context}\n\nالسؤال:\n{query}"

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.4,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ DeepSeek: {str(e)}")

# ==========================================
# مسارات التطبيق
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>منصة موجة البيان للدعم الفني</h1>"

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "indexed_documents": len(getattr(engine, "documents", [])),
        "chroma_connected": chroma_client is not None,
        "deepseek_ready": deepseek_client is not None,   # ✅ جديد
        "gemini_ready": bool(gemini_api_key_env),        # ✅ جديد
    }

@app.get("/api/topics")
async def get_topics():
    docs = getattr(engine, "documents", [])
    return {
        "topics": [
            {"title": d.get("title", ""), "url": d.get("url", ""), "filename": d.get("filename", "")}
            for d in docs[:15]
        ]
    }

@app.post("/api/search")
async def search_docs(req: SearchRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = engine.search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    user_input = req.message or req.query

    if not user_input or not user_input.strip():
        raise HTTPException(status_code=400, detail="النص المرسل فارغ")

    model_choice = (req.model or "deepseek").lower()

    # ✅ مسار DeepSeek (مع سياق RAG)
    if model_choice == "deepseek":
        context = build_rag_context(user_input)
        text = ask_deepseek(user_input, context=context)
        return {"text": text, "model": "deepseek-chat (RAG)"}

    # ✅ مسار Gemini (كما كان سابقًا)
    final_key = req.api_key or gemini_api_key_env
    if not final_key:
        raise HTTPException(
            status_code=500,
            detail="مفتاح GEMINI_API_KEY غير موجود في إعدادات الخادم (Render)",
        )

    try:
        result = engine.answer_query(user_input, gemini_api_key=final_key)
        if isinstance(result, dict):
            pure_text = result.get("answer") or result.get("text") or result.get("response") or str(result)
        elif isinstance(result, str):
            pure_text = result
        else:
            pure_text = str(result)
        return {"text": pure_text, "model": "Gemini RAG Engine"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
