import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

app = FastAPI(title="Mowjh Al-Bayan Support API")

# قراءة المتغيرات البيئية
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "gemini_docs"

# تشغيل العملاء
genai_client = genai.Client(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Mowjh Al-Bayan API"}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً.")

    # 1. تحويل سؤال المستخدم إلى تضمين متجهي
    embed_response = genai_client.models.embed_content(
        model="text-embedding-004",
        contents=request.question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )
    query_vector = embed_response.embedding.values

    # 2. البحث الدلالي في Qdrant
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3,
    )

    if not search_results:
        return QueryResponse(
            answer="لم يتم العثور على سياق مطابق في الوثائق.",
            sources=[]
        )

    # جمع السياق والمصادر
    context_chunks = [hit.payload.get("text", "") for hit in search_results]
    sources = list({hit.payload.get("source", "") for hit in search_results})
    full_context = "\n---\n".join(context_chunks)

    # 3. صياغة الإجابة باستخدام Gemini 2.5 Flash
    prompt = f"""أنت المساعد التقني الذكي لمنصة موج البيان. أجب عن سؤال المستخدم بدقة استناداً إلى السياق المرفق فقط.

السياق المسترجع من الوثائق:
{full_context}

سؤال المستخدم:
{request.question}
"""

    gen_response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return QueryResponse(
        answer=gen_response.text or "تعذر توليد رد.",
        sources=sources
    )
