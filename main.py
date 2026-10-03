import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from rag_engine import GroundedRAGEngine

app = FastAPI(
    title="منصة موجة البيان للدعم الفني",
    description="نظام دعم فني يعتمد على وثائق Gemini API الرسمية ومحرك RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GroundedRAGEngine()

class ChatRequest(BaseModel):
    # تم إضافة message ليتطابق مع ما ترسله واجهة HTML
    message: Optional[str] = None
    query: Optional[str] = None
    api_key: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>منصة موجة البيان للدعم الفني</h1>"

@app.get("/api/health")
async def health():
    return {"status": "healthy", "indexed_documents": len(engine.documents)}

@app.get("/api/topics")
async def get_topics():
    return {
        "topics": [
            {"title": d["title"], "url": d["url"], "filename": d["filename"]}
            for d in engine.documents[:15]
        ]
    }

@app.post("/api/search")
async def search_docs(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = engine.search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # نأخذ النص سواء أرسلته الواجهة كـ message أو query
    user_input = req.message or req.query
    
    if not user_input or not user_input.strip():
        raise HTTPException(status_code=400, detail="النص المرسل فارغ")

    # الأمان أولاً: استخدام مفتاح المستخدم إن وجد، أو جلب المفتاح المحمي من خادم ريندر
    final_key = req.api_key or os.environ.get("GEMINI_API_KEY")

    if not final_key:
        raise HTTPException(
            status_code=500,
            detail="مفتاح GEMINI_API_KEY غير موجود في إعدادات الخادم (Render)",
        )

    try:
        # جلب الرد من محرك RAG
        result = engine.answer_query(user_input, gemini_api_key=final_key)
        
        # --- تهيئة الرد ليتوافق مع واجهة HTML ---
        if isinstance(result, str):
            # إذا كان الرد مجرد نص، نحوله إلى JSON
            return {"text": result, "model": "Gemini RAG Engine"}
        elif isinstance(result, dict):
            # إذا كان قاموساً، نضمن وجود مفتاح text
            if "text" not in result and "response" not in result:
                return {"text": str(result), "model": "Gemini RAG Engine"}
            return result
        else:
            return {"text": str(result), "model": "Gemini RAG Engine"}
            
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
