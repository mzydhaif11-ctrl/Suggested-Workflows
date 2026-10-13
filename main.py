import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# استدعاء محرك الـ RAG الخاص بك من ملف rag_engine.py
from rag_engine import GroundedRAGEngine

# قراءة مفاتيح الـ API مباشرة من بيئة Render
gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

# تهيئة محرك الـ RAG ليكون جاهزاً للبحث في مجلد docs
rag_engine = GroundedRAGEngine()

# إعداد عميل DeepSeek
deepseek_client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
) if deepseek_api_key else None

app = FastAPI()

# السماح بالاتصالات (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# هيكل البيانات المستلمة من الواجهة
class ChatRequest(BaseModel):
    message: str
    model_name: str = "rag"  # الخيار الافتراضي صار محرك التوثيق RAG

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        chosen_model = request.model_name.lower()

        # 1. مسار DeepSeek (شات مباشر عام)
        if chosen_model == "deepseek":
            if not deepseek_client:
                raise HTTPException(status_code=500, detail="مفتاح DEEPSEEK_API_KEY غير متوفر في الخادم.")
            
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": request.message}]
            )
            return {"reply": response.choices[0].message.content}

        # 2. مسار RAG (البحث في التوثيق عبر مجلد docs والإجابة بواسطة Gemini)
        else:
            # يستدعي محرك البحث الخاص بك مباشرة
            rag_result = rag_engine.answer_query(query=request.message)
            return {
                "reply": rag_result["answer"],
                "sources": rag_result.get("sources", [])
            }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ربط واجهة المستخدم
app.mount("/", StaticFiles(directory="static", html=True), name="static")
