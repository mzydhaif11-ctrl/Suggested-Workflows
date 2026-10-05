import os
import json
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
from qdrant_client import QdrantClient
import chromadb
from chromadb.utils import embedding_functions

# ──────────────────────────────────────────────
# متغيرات البيئة
# ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHROMA_URL = os.getenv("CHROMA_URL")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")

# ──────────────────────────────────────────────
# تهيئة نماذج التوليد
# ──────────────────────────────────────────────
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

# ──────────────────────────────────────────────
# تهيئة قواعد البيانات المتجهة (Vector DBs)
# ──────────────────────────────────────────────
def get_embedding(text: str) -> list:
    """توليد متجه رقمي عبر Gemini text-embedding-004"""
    if not GOOGLE_API_KEY:
        return []
    try:
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return res.get("embedding", [])
    except Exception as e:
        print(f"⚠️ فشل توليد المتجه: {e}")
        return []

def get_qdrant_client():
    try:
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(url=QDRANT_URL)
    except Exception as e:
        print(f"⚠️ خطأ اتصال Qdrant: {e}")
        return None

def get_chroma_collection():
    try:
        gemini_ef = None
        if GOOGLE_API_KEY:
            gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=GOOGLE_API_KEY,
                model_name="models/text-embedding-004"
            )

        if CHROMA_URL and CHROMA_API_KEY:
            connect_kwargs = {
                "host": CHROMA_URL,
                "headers": {"Authorization": f"Bearer {CHROMA_API_KEY}"}
            }
            if CHROMA_TENANT:
                connect_kwargs["tenant"] = CHROMA_TENANT
                connect_kwargs["database"] = "default_database"

            client = chromadb.HttpClient(**connect_kwargs)
        else:
            client = chromadb.Client()

        return client.get_or_create_collection(
            name="mowjat_advisor_memory",
            embedding_function=gemini_ef
        )
    except Exception as e:
        print(f"⚠️ خطأ اتصال ChromaDB: {e}")
        return None

def retrieve_context(query: str, top_k: int = 3) -> list:
    """استرجاع السياق الموحد من Qdrant و ChromaDB"""
    contexts = []

    # 1. البحث في Qdrant
    qdrant = get_qdrant_client()
    query_vector = get_embedding(query)
    if qdrant and query_vector:
        try:
            results = qdrant.search(
                collection_name="documents",
                query_vector=query_vector,
                limit=top_k
            )
            for hit in results:
                if hit.payload and "content" in hit.payload:
                    contexts.append(hit.payload["content"])
        except Exception as e:
            print(f"⚠️ خطأ البحث في Qdrant: {e}")

    # 2. البحث في ChromaDB
    chroma = get_chroma_collection()
    if chroma:
        try:
            res = chroma.query(query_texts=[query], n_results=top_k)
            if res and res.get("documents") and res["documents"][0]:
                for doc in res["documents"][0]:
                    contexts.append(doc)
        except Exception as e:
            print(f"⚠️ خطأ البحث في ChromaDB: {e}")

    return contexts

# ──────────────────────────────────────────────
# التطبيق (FastAPI)
# ──────────────────────────────────────────────
app = FastAPI(title="موجة البيان", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "auto"

class ChatResponse(BaseModel):
    response: str
    model_used: str
    sources: List[str] = []

# حل مشكلة الفحص التلقائي من Render وضمان ظهور النص العربي بترميز سليم
@app.get("/")
@app.head("/")
async def root():
    return JSONResponse(
        content={"status": "ok", "service": "موجة البيان"},
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 1. استرجاع السياق من قواعد البيانات (إن وجد)
    context = retrieve_context(request.message)
    context_str = "\n".join([f"- {c}" for c in context]) if context else "لا توجد مستندات إضافية."
    
    selected_model = request.model or "auto"
    if selected_model == "auto":
        selected_model = "deepseek" if any(w in request.message.lower() for w in ["كود", "code", "bug", "دالة"]) else "gemini"

    # 2. المحاولة عبر DeepSeek
    if selected_model == "deepseek" and deepseek_client:
        try:
            prompt = f"المعلومات المتوفرة:\n{context_str}\n\nالسؤال: {request.message}"
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "أنت مساعد ذكي لمنصة موجة البيان. أجب بالعربية بدقة."},
                    {"role": "user", "content": prompt}
                ]
            )
            return ChatResponse(
                response=resp.choices[0].message.content,
                model_used="deepseek-chat",
                sources=context[:3]
            )
        except Exception as e:
            print(f"فشل DeepSeek، التبديل إلى Gemini: {e}")

    # 3. المعالجة عبر Gemini
    if gemini_model:
        try:
            full_prompt = (
                f"أنت مستشار الدعم الفني لمنصة موجة البيان.\n"
                f"استند إلى السياق التالي في إجابتك إن كان مفيداً:\n{context_str}\n\n"
                f"سؤال المستخدم: {request.message}"
            )
            resp = gemini_model.generate_content(full_prompt)
            
            # حفظ المحادثة في ChromaDB كذاكرة للمستقبل
            chroma = get_chroma_collection()
            if chroma:
                try:
                    mem_id = f"mem_{int(os.times().elapsed * 1000)}"
                    chroma.add(
                        ids=[mem_id],
                        documents=[f"سؤال: {request.message} | إجابة: {resp.text}"]
                    )
                except Exception:
                    pass

            return ChatResponse(
                response=resp.text,
                model_used="gemini-1.5-flash",
                sources=context[:3]
            )
        except Exception as e:
            return ChatResponse(
                response=f"تعذر الحصول على رد من النموذج: {e}",
                model_used="error",
                sources=[]
            )

    # في حال عدم ضبط المفاتيح
    return ChatResponse(
        response="الخدمة تعمل بنجاح، يرجى التأكد من إضافة GOOGLE_API_KEY في إعدادات البيئة لتفعيل الردود الذكية.",
        model_used="system",
        sources=[]
    )

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "gemini": "configured" if GOOGLE_API_KEY else "missing_key",
        "deepseek": "configured" if DEEPSEEK_API_KEY else "missing_key",
        "qdrant": "connected" if get_qdrant_client() else "disconnected",
        "chromadb": "connected" if get_chroma_collection() else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
