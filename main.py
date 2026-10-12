import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 1. استدعاء المكتبة الجديدة الخاصة بـ Gemini بناءً على ملف متطلباتك
from google import genai

# 2. تحميل المتغيرات من ملف .env (مفيد عند التشغيل على جهازك المحلي)
load_dotenv()

# 3. جلب المفتاح (النسخة الجديدة تعتمد بشكل أساسي على GEMINI_API_KEY)
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("تنبيه مهم: لم يتم العثور على مفتاح API في متغيرات النظام.")

# 4. إعداد الـ Client الخاص بالمكتبة الجديدة (google-genai)
# سيتم إنشاء العميل فقط إذا كان المفتاح موجوداً
client = genai.Client(api_key=api_key) if api_key else None

app = FastAPI()

# السماح بالاتصالات (CORS) لمنع أي مشاكل في الواجهة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تعريف شكل البيانات المستقبلة من الواجهة
class ChatRequest(BaseModel):
    message: str

# مسار المحادثة الذي يتصل به ملف index.html
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="مفتاح API غير متوفر في الخادم.")
    
    try:
        # 5. إرسال الطلب بالطريقة المتوافقة مع الإصدار 2.0.0 من google-genai
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=request.message,
        )
        return {"reply": response.text}
    
    except Exception as e:
        # إرجاع أي خطأ يواجه السيرفر ليظهر في الـ Logs
        raise HTTPException(status_code=500, detail=str(e))

# 6. ربط المجلد الثابت (static) لعرض ملف index.html كواجهة رئيسية
# ملاحظة هامة: يجب أن يبقى هذا السطر في نهاية الملف دائماً!
app.mount("/", StaticFiles(directory="static", html=True), name="static")
