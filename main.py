import os
from typing import List, Optional
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from rag_engine import GroundedRAGEngine

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
# استدعاء الإعدادات من سيرفر Render بأمان (بدون كتابة مفاتيح)
# ==========================================
chroma_url = os.environ.get("CHROMA_URL")
chroma_api_key = os.environ.get("CHROMA_API_KEY")
chroma_tenant = os.environ.get("CHROMA_TENANT")

chroma_client = None
chroma_collection = None

# إنشاء الاتصال بالسحابة فقط في حال تم ضبط المتغيرات في Render
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

# تهيئة محرك RAG الأصلي
engine = GroundedRAGEngine()

# ==========================================
# نماذج البيانات (Models)
# ==========================================
class ChatRequest(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None
    api_key: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# ==========================================
# مسارات التطبيق (Endpoints)
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

    final_key = req.api_key or os.environ.get("GEMINI_API_KEY")

    if not final_key:
        raise HTTPException(
            status_code=500,
            detail="مفتاح GEMINI_API_KEY غير موجود في إعدادات الخادم (Render)",
        )

    try:
        result = engine.answer_query(user_input, gemini_api_key=final_key)
        
        # استخراج النص الصافي ومنع ظهور مفاتيح القاموس البرمجي
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
