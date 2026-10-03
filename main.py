import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from google import genai
from qdrant_client import QdrantClient

# 1. قراءة المتغيرات البيئية
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "gemini_docs"

# 2. تشغيل عملاء الاتصال (Gemini & Qdrant)
genai_client = genai.Client(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 3. إعداد دورة حياة التطبيق (لتشغيل قاعدة البيانات تلقائياً)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import ingest
        print("بدء تهيئة وفهرسة البيانات...")
        ingest.main()
        print("اكتملت التهيئة بنجاح.")
    except Exception as e:
        print(f"تنبيه التهيئة: {e}")
    yield

# 4. تهيئة تطبيق FastAPI
app = FastAPI(title="Mowjh Al-Bayan API", lifespan=lifespan)

# 5. إعداد مجلد القوالب (لعرض واجهات HTML)
templates = Jinja2Templates(directory="templates")

# 6. نماذج البيانات (Pydantic Models)
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

# 7. المسارات (Routes)
@app.get("/")
def health_check():
    """مسار فحص حالة السيرفر"""
    return {"status": "ok", "service": "Mowjh Al-Bayan API"}

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """مسار عرض واجهة تسجيل الدخول"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """مسار استقبال الأسئلة والرد عليها بالذكاء الاصطناعي"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً.")

    # الخطوة أ: تحويل سؤال المستخدم إلى متجه رقمي
    try:
        embed_response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=request.question
        )
        
        # استخراج المتجه بشكل آمن
        if hasattr(embed_response, 'embeddings') and embed_response.embeddings:
            query_vector = embed_response.embeddings[0].values
        else:
            query_vector = embed_response.embedding.values
            
    except Exception as e:
        print(f"Embedding Error: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء تحويل السؤال في نماذج Gemini.")

    # الخطوة ب: البحث الدلالي في قاعدة بيانات Qdrant
    try:
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
        )
    except Exception as e:
        print(f"Qdrant Search Error: {e}")
        raise HTTPException(status_code=500, detail="تعذر الاتصال بقاعدة بيانات Qdrant.")

    if not search_results:
        return QueryResponse(
            answer="لم يتم العثور على سياق مطابق في الوثائق.",
            sources=[]
        )

    # استخراج النصوص المسترجعة والمصادر
    context_chunks = [hit.payload.get("text", "") for hit in search_results]
    sources = list({hit.payload.get("source", "") for hit in search_results})
    full_context = "\n---\n".join(context_chunks)

    # الخطوة ج: صياغة الرد الذكي باستخدام Gemini 2.5 Flash
    prompt = f"""أنت المساعد التقني الذكي لمنصة Mowjh Al-Bayan. أجب عن سؤال المستخدم بدقة استناداً إلى السياق المرفق فقط.

السياق المسترجع من الوثائق:
{full_context}

سؤال المستخدم:
{request.question}
"""

    try:
        gen_response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    except Exception as e:
        print(f"Generation Error: {e}")
        raise HTTPException(status_code=500, detail="حدث خطأ أثناء صياغة الإجابة.")

    return QueryResponse(
        answer=gen_response.text or "تعذر توليد رد.",
        sources=sources
    )
