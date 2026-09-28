import os from typing import Optional, List from fastapi import FastAPI, HTTPException from
fastapi.middleware.cors import CORSMiddleware from fastapi.responses import HTMLResponse, FileResponse from
pydantic import BaseModel from rag_
engine import GroundedRAGEngine app = FastAPI( title=
منصة موجة البيان للدعم الفني"
"الذكي
, description=
سمية Gemini API خادم دعم العمﻼء المؤصل بوثائق"
"الر
, version=
"1.0.0" ) app.add
_
middleware( CORSMiddleware,
allow
_
origins=["*"], allow
credentials=True, allow
methods=["*"], allow
_
_
_
headers=["*"], ) engine =
GroundedRAGEngine() class ChatRequest(BaseModel): query: str api
_
key: Optional[str] = None class
SearchRequest(BaseModel): query: str top_
k: Optional[int] = 5 @app.get("/"
, response
_
class=HTMLResponse) async
def read
_
index(): index
_
file = os.path.join(os.path.dirname(__
file
,)__
"static"
,
"index.html") if
os.path.exists(index
_
file): return FileResponse(index
_
file) return "<h1 style=
'text-align:center;
منصة موجة البيان>'
"h1>" @app.get("/api/health") async def health(): return { "status/<تعمل بنجاح
:
"healthy"
,
"platform"
:
"Mowjh Al-
Bayan"
,
"indexed
documents"
_
: len(engine.documents) } @app.post("/api/search") async def search
_
docs(req:
SearchRequest): if not req.query.strip(): raise HTTPException(status
code=400, detail=
_
)"ﻻ يمكن إرسال استعﻼم فارغ"
return {"query"
: req.query,
"results"
: engine.search(req.query, top_
k=req.top_
k)} @app.post("/api/chat") async
def chat(req: ChatRequest): if not req.query.strip(): raise HTTPException(status
code=400, detail=
ﻻ يمكن إرسال سؤال"
_
return engine.answer )"فارغ
_query(req.query, gemini
_
api
_
key=req.api
_
key) if
"
name
==
main
"
__
__
__
__
: import uvicorn
port = int(os.environ.get("PORT"
, 8080)) uvicorn.run("main:app"
, host=
"0.0.0.0"
, port=port, reload=True)
