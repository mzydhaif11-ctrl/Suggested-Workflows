import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from rag_engine import GroundedRAGEngine
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

app = FastAPI(
    title="منصة موجة البيان للدعم الفني",
    description="نظام دعم فني يعتمد على وثائق Gemini API الرسمية ومحرك RAG مع دعم Qdrant Cloud",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# الإعدادات من متغيرات البيئة
# ==========================================
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
gemini_api_key_env = os.environ.get("GEMINI_API_KEY")

COLLECTION_NAME = "tech_advisor_collection"
VECTOR_SIZE = 384   # مطابق لنموذج BAAI/bge-small-en-v1.5

qdrant_client = None
embedding_model = None
deepseek_client = None

# تهيئة عميل DeepSeek
if deepseek_api_key:
    try:
        deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
        print("✅ تم تهيئة عميل DeepSeek.")
    except Exception as e:
        print(f"❌ DeepSeek: {e}")

# تهيئة Qdrant + Embeddings
if qdrant_url and qdrant_api_key:
    try:
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        existing = [c.name for c in qdrant_client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            print("✅ تم إنشاء Collection جديد.")
        else:
            print("✅ تم الاتصال بـ Qdrant.")
        embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("✅ نموذج التضمين جاهز.")
    except Exception as e:
        print(f"❌ Qdrant: {e}")

engine = GroundedRAGEngine()

# ==========================================
# نماذج البيانات
# ==========================================
class ChatRequest(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = "deepseek"

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# ==========================================
# البحث في Qdrant: السياق + المصادر
# ==========================================
def build_rag_context(query: str, top_k: int = 5):
    if not qdrant_client or not embedding_model:
        return "", []

    try:
        query_vector = list(embedding_model.embed([query]))[0].tolist()
        hits = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        sources = []
        seen_titles = set()

        for i, hit in enumerate(hits, 1):
            payload = hit.payload or {}
            text = payload.get("text", "")
            if not text:
                continue
            chunks.append(f"[{i}] {text}")

            title = payload.get("title") or f"وثيقة {i}"
            if title not in seen_titles:
                seen_titles.add(title)
                sources.append({
                    "index": i,
                    "title": title,
                    "url": payload.get("url", ""),
                    "score": float(hit.score) if hit.score is not None else None,
                })

        return "\n\n---\n\n".join(chunks), sources
    except Exception as e:
        print(f"❌ بحث RAG: {e}")
        return "", []

# ==========================================
# استدعاء DeepSeek
# ==========================================
def ask_deepseek(query: str, context: str = "") -> str:
    if not deepseek_client:
        raise HTTPException(status_code=500, detail="مفتاح DEEPSEEK_API_KEY غير موجود")

    system_prompt = (
        "أنت مستشار تقني متخصص في Gemini API وDeepSeek ونماذج الذكاء الاصطناعي. "
        "أجب بالعربية الفصحى، بدقة، وبأسلوب منظم. "
        "إذا وُجد سياق مرجعي أدناه، اعتمد عليه في إجابتك، "
        "واذكر أرقام المصادر التي استخدمتها بين قوسين مربعين مثل [1] و [2]. "
        "إذا لم يكن السياق كافيًا، قل ذلك صراحة."
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
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ DeepSeek: {str(e)}")

# ==========================================
# إضافة وثائق إلى Qdrant
# ==========================================
def ingest_documents(docs: List[dict]) -> int:
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=500, detail="Qdrant غير متصل")
    if not docs:
        return 0

    all_texts = []
    all_payloads = []

    for doc_idx, doc in enumerate(docs):
        text = doc.get("text", "").strip()
        if not text:
            continue
        title = doc.get("title", f"وثيقة {doc_idx}")
        url = doc.get("url", "")

        chunk_size, overlap = 800, 100
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                all_texts.append(chunk)
                all_payloads.append({
                    "text": chunk,
                    "title": title,
                    "url": url,
                    "doc_index": doc_idx,
                })
            start = end - overlap

    if not all_texts:
        return 0

    vectors = list(embedding_model.embed(all_texts))
    import time
    base_id = int(time.time() * 1000)

    points = []
    for i, (vec, payload) in enumerate(zip(vectors, all_payloads)):
        points.append(PointStruct(
            id=base_id + i,
            vector=vec.tolist(),
            payload=payload,
        ))

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)

# ==========================================
# المسارات
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>منصة موجة البيان للدعم الفني</h1>"

@app.get("/api/health")
async def health():
    count = 0
    if qdrant_client:
        try:
            count = qdrant_client.count(collection_name=COLLECTION_NAME).count
        except Exception:
            pass
    return {
        "status": "healthy",
        "qdrant_connected": qdrant_client is not None,
        "embedding_ready": embedding_model is not None,
        "deepseek_ready": deepseek_client is not None,
        "gemini_ready": bool(gemini_api_key_env),
        "vectors_count": count,
    }

@app.get("/api/topics")
async def get_topics():
    if not qdrant_client:
        return {"topics": []}
    try:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            with_payload=True,
        )
        points = result[0]
        seen = set()
        topics = []
        for p in points:
            payload = p.payload or {}
            title = payload.get("title", "")
            if title and title not in seen:
                seen.add(title)
                topics.append({
                    "title": title,
                    "url": payload.get("url", ""),
                })
        return {"topics": topics}
    except Exception as e:
        return {"topics": [], "error": str(e)}

@app.post("/api/search")
async def search_docs(req: SearchRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    _, sources = build_rag_context(req.query, top_k=req.top_k)
    return {"query": req.query, "results": sources}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    user_input = req.message or req.query

    if not user_input or not user_input.strip():
        raise HTTPException(status_code=400, detail="النص المرسل فارغ")

    model_choice = (req.model or "deepseek").lower()

    if model_choice == "deepseek":
        context, sources = build_rag_context(user_input)
        text = ask_deepseek(user_input, context=context)
        return {
            "text": text,
            "model": "deepseek-chat (Qdrant RAG)",
            "sources": sources,
        }

    final_key = req.api_key or gemini_api_key_env
    if not final_key:
        raise HTTPException(status_code=500, detail="مفتاح GEMINI_API_KEY غير موجود")

    try:
        result = engine.answer_query(user_input, gemini_api_key=final_key)
        if isinstance(result, dict):
            pure_text = result.get("answer") or result.get("text") or result.get("response") or str(result)
        elif isinstance(result, str):
            pure_text = result
        else:
            pure_text = str(result)

        _, sources = build_rag_context(user_input)
        return {
            "text": pure_text,
            "model": "Gemini RAG Engine (Qdrant)",
            "sources": sources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
