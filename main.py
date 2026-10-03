import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
from qdrant_client import QdrantClient

# قراءة المتغيرات البيئية
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "gemini_docs"

# إعداد Gemini
genai.configure(api_key=GEMINI_API_KEY)

# إعداد Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import ingest
        print("بدء تهيئة وفهرسة البيانات...")
        ingest.main()
    except Exception as e:
        print(f"تنبيه التهيئة: {e}")
    yield

app = FastAPI(title="Mowjh Al-Bayan API", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً.")

    try:
        # استخدام Embedding
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=request.question,
            task_type="retrieval_query"
        )
        query_vector = result['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في Embedding: {e}")

    try:
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="تعذر الاتصال بقاعدة بيانات Qdrant.")

    if not search_results:
        return QueryResponse(answer="لم يتم العثور على سياق مطابق.", sources=[])

    context_chunks = [hit.payload.get("text", "") for hit in search_results]
    sources = list({hit.payload.get("source", "") for hit in search_results})
    full_context = "\n---\n".join(context_chunks)

    prompt = f"أجب بناءً على السياق فقط.\nالسياق:\n{full_context}\nالسؤال:\n{request.question}"

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        gen_response = model.generate_content(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail="خطأ أثناء صياغة الإجابة.")

    return QueryResponse(
        answer=gen_response.text,
        sources=sources
    )
