import os
import json
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from openai import OpenAI
from qdrant_client import QdrantClient
import chromadb
from chromadb.utils import embedding_functions

# إعدادات البيئة
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHROMA_URL = os.getenv("CHROMA_URL")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None

deepseek_client = None
if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

def get_qdrant_client():
    try:
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(url=QDRANT_URL)
    except Exception as e:
        print(f"خطأ Qdrant: {e}")
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
    except Exception as e:
        print(f"خطأ ChromaDB: {e}")
        return None

# ──────────────────────────────────────────────
# تعريف التطبيق باسم app (هذا السطر هو سبب الخطأ)
# ──────────────────────────────────────────────
app = FastAPI(title="موجة البيان", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "auto"

class ChatResponse(BaseModel):
    response: str
    model_used: str
    sources: List[str] = []

@app.get("/")
@app.head("/")
async def root():
    return {"status": "ok", "service": "موجة البيان"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return ChatResponse(
        response=f"تم استلام رسالتك: {request.message}",
        model_used=request.model or "gemini",
        sources=[]
    )

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "qdrant": "connected" if get_qdrant_client() else "disconnected",
        "chromadb": "connected" if get_chroma_collection() else "disconnected"
    }
