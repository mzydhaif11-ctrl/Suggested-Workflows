import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from openai import OpenAI

# قراءة المفاتيح مباشرة من خادم Render (بدون استخدام load_dotenv)
gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

deepseek_client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
) if deepseek_api_key else None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model_name: str = "gemini"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        if request.model_name.lower() == "deepseek":
            if not deepseek_client:
                raise HTTPException(status_code=500, detail="مفتاح DEEPSEEK_API_KEY غير متوفر في الخادم.")
            
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": request.message}]
            )
            return {"reply": response.choices[0].message.content}
        
        else:
            if not gemini_client:
                raise HTTPException(status_code=500, detail="مفتاح GEMINI_API_KEY غير متوفر في الخادم.")
            
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=request.message,
            )
            return {"reply": response.text}

    except Exception as e:
        print(f"Error occurred: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/", StaticFiles(directory="static", html=True), name="static")
