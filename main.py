import os  # تم تصحيح الحرف الأول ليكون صغيراً
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# استدعاء مكتبات الذكاء الاصطناعي
from google import genai
from openai import OpenAI

load_dotenv()

# جلب المفاتيح من بيئة Render
gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

# إعداد عميل Gemini
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

# إعداد عميل DeepSeek (يعمل عبر مكتبة openai)
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

# تعريف شكل البيانات (أضفنا خيار اختيار النموذج)
class ChatRequest(BaseModel):
    message: str
    model_name: str = "gemini"  # افتراضياً يستخدم جيميناي

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. إذا كان المطلوب تشغيل DeepSeek
        if request.model_name.lower() == "deepseek":
            if not deepseek_client:
                raise HTTPException(status_code=500, detail="مفتاح DEEPSEEK_API_KEY غير متوفر في الخادم.")
            
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": request.message}]
            )
            return {"reply": response.choices[0].message.content}
        
        # 2. الافتراضي تشغيل Gemini
        else:
            if not gemini_client:
                raise HTTPException(status_code=500, detail="مفتاح GEMINI_API_KEY غير متوفر في الخادم.")
            
            response = gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=request.message,
            )
            return {"reply": response.text}

    except Exception as e:
        # طباعة الخطأ في السجلات للتمكن من قراءته في Render
        print(f"Error occurred: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

# ربط واجهة المستخدم
app.mount("/", StaticFiles(directory="static", html=True), name="static")
