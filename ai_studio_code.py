# ============================================================
# منصة موجة البيان - خادم FastAPI المتكامل مع قاعدة بيانات SQLite
# مستودع: mzydhaif11-ctrl/Suggested-Workflows
# ============================================================

import os
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime

# 1. إعداد قاعدة البيانات SQLite و SQLAlchemy
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./mowjat_bayan.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    email = Column(String(100), unique=True, index=True)
    role = Column(String(50), default="مطور")
    token_quota = Column(Integer, default=100000)
    tokens_used = Column(Integer, default=0)

class WorkflowDB(Base):
    __tablename__ = "workflows"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100))
    title = Column(String(150))
    description = Column(Text)
    model = Column(String(50), default="gemini-3.8-flash")
    category = Column(String(50), default="دعم فني")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ChatLogDB(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text)
    response = Column(Text)
    model = Column(String(50))
    latency_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. إنشاء تطبيق FastAPI
app = FastAPI(
    title="منصة موجة البيان - FastAPI Engine",
    description="خادم الدعم الفني الذكي مع قاعدة بيانات مدمجة ونماذج Google Gemini & DeepSeek",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = "gemini-3.8-flash"
    temperature: Optional[float] = 0.7
    api_key: Optional[str] = None

class WorkflowCreate(BaseModel):
    title: str
    description: str
    model: Optional[str] = "gemini-3.8-flash"
    category: Optional[str] = "دعم فني"
    user_id: Optional[str] = "usr_guest"

# 3. المسارات الرئيسية
@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return "<h1>منصة موجة البيان للدعم الفني - خادم FastAPI وقاعدة البيانات نشطان!</h1>"

@app.get("/api/health")
async def health(db: Session = Depends(get_db)):
    return {
        "status": "healthy",
        "database": "SQLite Connected",
        "users_count": db.query(UserDB).count(),
        "workflows_count": db.query(WorkflowDB).count(),
    }

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="الاستفسار مطلوب")
    
    start_time = time.time()
    api_key = req.api_key or os.environ.get("GEMINI_API_KEY")
    
    answer_text = ""
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=req.model,
                contents=req.query,
            )
            answer_text = response.text
        except Exception as e:
            answer_text = f"تمت الإجابة عبر موجة البيان: {str(e)}"
    else:
        answer_text = f"إجابة منصة موجة البيان: النموذج الموصى به لطلبك هو {req.model} مع سرعة استجابة فائقة."

    latency = round((time.time() - start_time) * 1000, 2)

    # حفظ في سجل المحادثات بقاعدة البيانات
    log = ChatLogDB(
        query=req.query,
        response=answer_text,
        model=req.model,
        latency_ms=latency
    )
    db.add(log)
    db.commit()

    return {
        "answer": answer_text,
        "model": req.model,
        "latency_ms": latency,
        "database_logged": True
    }

@app.get("/api/workflows")
async def list_workflows(db: Session = Depends(get_db)):
    workflows = db.query(WorkflowDB).all()
    return {"workflows": workflows}

@app.post("/api/workflows")
async def create_workflow(item: WorkflowCreate, db: Session = Depends(get_db)):
    wf = WorkflowDB(
        title=item.title,
        description=item.description,
        model=item.model,
        category=item.category,
        user_id=item.user_id
    )
    db.add(wf)
    db.commit()
    db.refresh(wf)
    return {"success": True, "workflow": wf}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)