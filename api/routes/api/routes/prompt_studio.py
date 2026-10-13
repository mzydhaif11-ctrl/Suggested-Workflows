# api/routes/prompt_studio.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/prompt-studio", tags=["Prompt Studio"])

class EnhancePromptRequest(BaseModel):
    rawIdea: str
    temperature: float = 0.7

class EnhancePromptResponse(BaseModel):
    enhancedPrompt: str

@router.post("/enhance", response_model=EnhancePromptResponse)
async def enhance_prompt_endpoint(payload: EnhancePromptRequest):
    idea = payload.rawIdea.strip()
    if not idea:
        raise HTTPException(status_code=400, detail="الفكرة المدخلة فارغة")

    try:
        # صياغة الموجه وفق معايير هندسة الأوامر (Prompt Engineering)
        structured_prompt = (
            f"[الدور والسياق]: بصفتك خبيراً متخصصاً في تنفيذ المهام بدقة:\n"
            f"[المهمة المطلوبة]: {idea}\n\n"
            f"[إرشادات التنفيذ]:\n"
            f"1. التفكير المنهجي والتحليل المباشر للهدف.\n"
            f"2. تقديم النتائج في شكل خطوات واضحة وقابلة للتطبيق.\n"
            f"3. تجنب الحشو والمقدمات العامة وتقديم إجابة موجهة للهدف.\n"
            f"[شكل المخرجات]: مخرجات مهيكلة ومباشرة."
        )

        return EnhancePromptResponse(enhancedPrompt=structured_prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
