import os
import json
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
from qdrant_client import QdrantClient
import chromadb
from chromadb.utils import embedding_functions

# ──────────────────────────────────────────────
# متغيرات البيئة
# ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHROMA_URL = os.getenv("CHROMA_URL")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

# ──────────────────────────────────────────────
# التطبيق (FastAPI)
# ──────────────────────────────────────────────
app = FastAPI(title="موجة البيان", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# قواعد البيانات
# ──────────────────────────────────────────────
def get_embedding(text: str) -> list:
    if not GOOGLE_API_KEY:
        return []
    try:
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return res.get("embedding", [])
    except Exception:
        return []

def get_qdrant_client():
    try:
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(url=QDRANT_URL)
    except Exception:
        return None

def get_chroma_collection():
    try:
        gemini_ef = None
        if GOOGLE_API_KEY:
            gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=GOOGLE_API_KEY,
                model_name="models/text-embedding-004"
            )

        if CHROMA_URL and CHROMA_API_KEY:
            connect_kwargs = {
                "host": CHROMA_URL,
                "headers": {"Authorization": f"Bearer {CHROMA_API_KEY}"}
            }
            if CHROMA_TENANT:
                connect_kwargs["tenant"] = CHROMA_TENANT
                connect_kwargs["database"] = "default_database"
            client = chromadb.HttpClient(**connect_kwargs)
        else:
            client = chromadb.Client()

        return client.get_or_create_collection(
            name="mowjat_advisor_memory",
            embedding_function=gemini_ef
        )
    except Exception:
        return None

def retrieve_context(query: str, top_k: int = 3) -> list:
    contexts = []
    qdrant = get_qdrant_client()
    query_vector = get_embedding(query)
    if qdrant and query_vector:
        try:
            results = qdrant.search(
                collection_name="documents",
                query_vector=query_vector,
                limit=top_k
            )
            for hit in results:
                if hit.payload and "content" in hit.payload:
                    contexts.append(hit.payload["content"])
        except Exception:
            pass

    chroma = get_chroma_collection()
    if chroma:
        try:
            res = chroma.query(query_texts=[query], n_results=top_k)
            if res and res.get("documents") and res["documents"][0]:
                for doc in res["documents"][0]:
                    contexts.append(doc)
        except Exception:
            pass

    return contexts

# ──────────────────────────────────────────────
# موديلات الطلب والرد
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "auto"

class ChatResponse(BaseModel):
    response: str
    model_used: str
    sources: List[str] = []

# ──────────────────────────────────────────────
# الصفحة الرئيسية (تشغيل index.html مباشرة عند فتح الرابط)
# ──────────────────────────────────────────────
@app.get("/")
@app.head("/")
async def serve_index():
    index_file = Path("index.html")
    # إذا كان ملف index.html موجوداً في المستودع يتم عرضه مباشرة
    if index_file.is_file():
        return FileResponse(index_file)
    
    # واجهة افتراضية احتياطية في حال لم ترفع index.html بعد
    return HTMLResponse("""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>منصة موجة البيان</title>
        <style>
            * { box-sizing: border-box; }
            body { font-family: system-ui, sans-serif; background: #0f172a; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; padding: 12px; }
            .chat-card { width: 100%; max-width: 480px; height: 90vh; background: #1e293b; border-radius: 16px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
            .header { background: #2563eb; padding: 18px; text-align: center; font-weight: bold; font-size: 1.2rem; }
            .messages { flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
            .msg { padding: 12px 16px; border-radius: 12px; max-width: 85%; line-height: 1.5; font-size: 0.95rem; word-break: break-word; }
            .user { background: #3b82f6; align-self: flex-start; }
            .bot { background: #334155; align-self: flex-end; }
            .input-box { display: flex; padding: 12px; background: #0f172a; gap: 8px; }
            input { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #334155; background: #1e293b; color: white; outline: none; font-size: 1rem; }
            button { background: #2563eb; color: white; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; font-size: 1rem; }
        </style>
    </head>
    <body>
        <div class="chat-card">
            <div class="header">مستشار موجة البيان</div>
            <div class="messages" id="chat">
                <div class="msg bot">مرحباً بك في منصة موجة البيان! كيف يمكنني مساعدتك؟</div>
            </div>
            <div class="input-box">
                <input type="text" id="userInput" placeholder="اكتب سؤالك هنا..." onkeydown="if(event.key==='Enter') send()">
                <button onclick="send()">إرسال</button>
            </div>
        </div>
        <script>
            async function send() {
                const input = document.getElementById('userInput');
                const chat = document.getElementById('chat');
                const text = input.value.trim();
                if (!text) return;
                chat.innerHTML += `<div class="msg user">${text}</div>`;
                input.value = '';
                chat.scrollTop = chat.scrollHeight;

                const botMsg = document.createElement('div');
                botMsg.className = 'msg bot';
                botMsg.innerText = 'جاري التفكير...';
                chat.appendChild(botMsg);
                chat.scrollTop = chat.scrollHeight;

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, model: 'auto' })
                    });
                    const data = await res.json();
                    botMsg.innerText = data.response;
                } catch(e) {
                    botMsg.innerText = 'تعذر الاتصال بالخادم.';
                }
                chat.scrollTop = chat.scrollHeight;
            }
        </script>
    </body>
    </html>
    """)

# ──────────────────────────────────────────────
# نقطة نهاية المحادثة (API)
# ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    context = retrieve_context(request.message)
    context_str = "\n".join([f"- {c}" for c in context]) if context else "لا توجد مستندات إضافية."

    selected_model = request.model or "auto"
    if selected_model == "auto":
        selected_model = "deepseek" if any(w in request.message.lower() for w in ["كود", "code", "bug", "دالة"]) else "gemini"

    # 1. التشغيل عبر DeepSeek
    if selected_model == "deepseek" and deepseek_client:
        try:
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "أنت مساعد ذكي لمنصة موجة البيان. أجب بالعربية بدقة."},
                    {"role": "user", "content": f"السياق:\n{context_str}\n\nالسؤال: {request.message}"}
                ]
            )
            return ChatResponse(
                response=resp.choices[0].message.content,
                model_used="deepseek-chat",
                sources=context[:3]
            )
        except Exception:
            pass

    # 2. التشغيل عبر Gemini
    if GOOGLE_API_KEY:
        models_to_try = ["gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-pro"]
        for m_name in models_to_try:
            try:
                model_instance = genai.GenerativeModel(m_name)
                full_prompt = (
                    f"أنت مستشار الدعم الفني لمنصة موجة البيان.\n"
                    f"السياق المتاح:\n{context_str}\n\n"
                    f"سؤال المستخدم: {request.message}"
                )
                resp = model_instance.generate_content(full_prompt)

                # حفظ المحادثة في ChromaDB
                chroma = get_chroma_collection()
                if chroma:
                    try:
                        mem_id = f"mem_{int(os.times().elapsed * 1000)}"
                        chroma.add(
                            ids=[mem_id],
                            documents=[f"سؤال: {request.message} | إجابة: {resp.text}"]
                        )
                    except Exception:
                        pass

                return ChatResponse(
                    response=resp.text,
                    model_used=m_name,
                    sources=context[:3]
                )
            except Exception:
                continue

    return ChatResponse(
        response="الخدمة تعمل بنجاح، يرجى التأكد من ضبط GOOGLE_API_KEY في إعدادات البيئة لتفعيل الذكاء الاصطناعي.",
        model_used="system",
        sources=[]
    )

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "gemini": "configured" if GOOGLE_API_KEY else "missing_key",
        "deepseek": "configured" if DEEPSEEK_API_KEY else "missing_key",
        "qdrant": "connected" if get_qdrant_client() else "disconnected",
        "chromadb": "connected" if get_chroma_collection() else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
