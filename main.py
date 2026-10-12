import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# 1. جلب المفتاح من متغيرات النظام (Render Environment)
api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("تنبيه مهم: لم يتم العثور على GOOGLE_API_KEY في متغيرات النظام.")

# إعداد نموذج الذكاء الاصطناعي
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()

# السماح بالاتصالات (CORS) لمنع أي مشاكل في الواجهة
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. تعريف شكل البيانات المستقبلة من الواجهة
class ChatRequest(BaseModel):
    message: str

# 3. إنشاء مسار المحادثة الذي يتصل به ملف index.html
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="مفتاح API غير متوفر في الخادم.")
    
    try:
        # إرسال رسالة المستخدم إلى Gemini
        response = model.generate_content(request.message)
        return {"reply": response.text}
    
    except Exception as e:
        # إرجاع أي خطأ يواجه السيرفر ليظهر في الـ Logs
        raise HTTPException(status_code=500, detail=str(e))

# 4. ربط المجلد الثابت (static) لعرض ملف index.html كواجهة رئيسية
# ملاحظة هامة: يجب أن يبقى هذا السطر في نهاية الملف دائماً!
app.mount("/", StaticFiles(directory="static", html=True), name="static")
