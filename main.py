import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb

# ──────────────────────────────────────────────
# إعدادات البيئة
# ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_ADMIN_TOKEN = os.getenv("CHROMA_ADMIN_TOKEN")

# ──────────────────────────────────────────────
# تهيئة النماذج
# ──────────────────────────────────────────────
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ──────────────────────────────────────────────
# تهيئة قواعد البيانات المتجهة
# ──────────────────────────────────────────────
def get_qdrant_client():
    """الاتصال بـ Qdrant"""
    try:
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(url=QDRANT_URL)
    except Exception as e:
        print(f"❌ خطأ في الاتصال بـ Qdrant: {e}")
        raise

def get_chroma_client():
    """الاتصال بـ ChromaDB"""
    try:
        if CHROMA_API_KEY:
            client = chromadb.HttpClient(
                host="localhost",
                port=8000,
                headers={
                    "X-Chroma-Token": CHROMA_API_KEY,
                    "Authorization": f"Bearer {CHROMA_ADMIN_TOKEN}"
                }
            )
        else:
            client = chromadb.Client()
        return client
    except Exception as e:
        print(f"❌ خطأ في الاتصال بـ ChromaDB: {e}")
        raise

# ──────────────────────────────────────────────
# التطبيق
# ──────────────────────────────────────────────
app = FastAPI(title="موجة البيان", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# موديلات الطلب/الاستجابة
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "auto"  # auto | deepseek | gemini

class ChatResponse(BaseModel):
    response: str
    model_used: str
    sources: list = []

# ──────────────────────────────────────────────
# دوال مساعدة
# ──────────────────────────────────────────────
def classify_query(message: str) -> str:
    """تصنيف نوع السؤال لتحديد النموذج المناسب"""
    technical_keywords = ["كود", "code", "api", "endpoint", "function", "bug", "خطأ", "debug"]
    creative_keywords = ["اكتب", "compose", "write", "صمم", "design", "اقتراح", "suggest"]
    
    msg_lower = message.lower()
    tech_count = sum(1 for kw in technical_keywords if kw in msg_lower)
    creative_count = sum(1 for kw in creative_keywords if kw in msg_lower)
    
    if tech_count > creative_count:
        return "deepseek"
    elif creative_count > tech_count:
        return "gemini"
    return "gemini"  # الافتراضي

def retrieve_context(query: str, top_k: int = 3) -> list:
    """استرجاع السياق من Qdrant + ChromaDB"""
    contexts = []
    
    try:
        qdrant = get_qdrant_client()
        # استرجاع من Qdrant
        results = qdrant.search(
            collection_name="documents",
            query_text=query,
            limit=top_k
        )
        for hit in results:
            contexts.append(hit.payload.get("content", ""))
    except Exception as e:
        print(f"⚠️ تحذير: فشل الاسترجاع من Qdrant: {e}")
    
    try:
        chroma = get_chroma_client()
        collection = chroma.get_or_create_collection("documents")
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        if results["documents"]:
            for doc in results["documents"][0]:
                contexts.append(doc)
    except Exception as e:
        print(f"⚠️ تحذير: فشل الاسترجاع من ChromaDB: {e}")
    
    return contexts

def generate_response(message: str, context: list, model: str) -> str:
    """توليد الرد باستخدام النموذج المختار"""
    system_prompt = f"""أنت مساعد ذكي لمنصة موجة البيان. استخدم المعلومات التالية للإجابة:

{chr(10).join([f'- {c}' for c in context])}

الإجابة يجب أن تكون بالعربية الفصحى، واضحة ومختصرة."""

    if model == "deepseek":
        try:
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"❌ خطأ في DeepSeek: {e}")
            raise HTTPException(status_code=503, detail="DeepSeek غير متاح")
    
    else:  # gemini
        try:
            prompt = f"{system_prompt}\n\nسؤال المستخدم: {message}"
            resp = gemini_model.generate_content(prompt)
            return resp.text
        except Exception as e:
            print(f"❌ خطأ في Gemini: {e}")
            raise HTTPException(status_code=503, detail="Gemini غير متاح")

# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "موجة البيان"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """نقطة المحادثة الرئيسية"""
    try:
        # تحديد النموذج
        selected_model = request.model if request.model != "auto" else classify_query(request.message)
        
        # استرجاع السياق
        context = retrieve_context(request.message)
        
        # توليد الرد
        response = generate_response(request.message, context, selected_model)
        
        return ChatResponse(
            response=response,
            model_used=selected_model,
            sources=context[:3]
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ خطأ عام: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """فحص صحة الخدمة"""
    checks = {}
    
    try:
        get_qdrant_client()
        checks["qdrant"] = "connected"
    except:
        checks["qdrant"] = "disconnected"
    
    try:
        get_chroma_client()
        checks["chromadb"] = "connected"
    except:
        checks["chromadb"] = "disconnected"
    
    checks["deepseek"] = "configured" if DEEPSEEK_API_KEY else "missing key"
    checks["gemini"] = "configured" if GOOGLE_API_KEY else "missing key"
    
    return {"status": "ok", "checks": checks}

@app.get("/docs/export")
async def export_documents():
    """تصدير المستندات المخزنة"""
    try:
        documents = []
        
        try:
            chroma = get_chroma_client()
            collection = chroma.get_or_create_collection("documents")
            data = collection.get(include=["documents", "metadatas"])
            for i, doc in enumerate(data["documents"] or []):
                documents.append({
                    "id": i,
                    "content": doc,
                    "metadata": data["metadatas"][i] if data["metadatas"] else {}
                })
        except Exception as e:
            print(f"⚠️ خطأ في تصدير ChromaDB: {e}")
        
        return documents if documents else [{"note": "لا توجد مستندات مخزنة بعد"}]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
