import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.prompt_studio import router as prompt_studio_router
from api.routes.ai_bridge import router as ai_bridge_router

app = FastAPI(
    title="Mawjah AI Platform",
    description="Mawjah integration layer for Gemini and OpenAI",
    version="1.0.0",
)

# ضع عناوين الواجهة المسموح بها في إعدادات البيئة.
# لا تستخدم السماح لجميع المصادر في بيئة الإنتاج.
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# الإضافات
app.include_router(prompt_studio_router)
app.include_router(ai_bridge_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "application": "Mawjah",
        "version": "1.0.0",
    }
