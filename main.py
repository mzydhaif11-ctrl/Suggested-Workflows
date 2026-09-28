import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from rag_engine import GroundedRAGEngine

app = FastAPI(
    title="منصة موجة البيان للدعم الفني",
    description="نظام دعم فني يعتمد على وثائق Gemini API الرسمية ومحرك RAG",
    version="1.0.0"
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
    query: str
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
    return {
        "status": "healthy",
        "indexed_documents": len(engine.documents)
    }

@app.get("/api/topics")
async def get_topics():
    return {
        "topics": [{"title": d["title"], "url": d["url"], "filename": d["filename"]} for d in engine.documents[:15]]
    }

@app.post("/api/search")
async def search_docs(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = engine.search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return engine.answer_query(req.query, gemini_api_key=req.api_key)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
