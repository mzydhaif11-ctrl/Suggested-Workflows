# api/routes/prompt_studio.py

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

router = APIRouter(prefix="/api/prompt-studio", tags=["Prompt Studio"])

# تهيئة عميل OpenAI عبر مفتاح البيئة
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EnhancePromptRequest(BaseModel):
    rawIdea: str = Field(..., description="الفكرة الأولية المدخلة من المستخدم")
    model: str = Field(default="gpt-4o-mini", description="نموذج OpenAI المختار")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class EnhancePromptResponse(BaseModel):
    enhancedPrompt: str
    modelUsed: str

SYSTEM_PROMPT_ENGINEER_INSTRUCTION = """
أنت خبير محترف في هندسة الأوامر (Senior Prompt Engineer).
مهمتك: تحويل الفكرة الأولية البسيطة التي يقدمها المستخدم إلى موجه (Prompt) احترافي، دقيق، وشديد الفعالية.

القواعد الإرشادية لصياغة الموجه:
1. حدد دور وسياق النموذج بوضوح (Role & Context).
2. صغ التعليمات في خطوات متسلسلة وقابلة للتنفيذ (Clear Instructions).
3. حدد القيود، الاستثناءات، وما يجب تجنبه (Constraints).
4. حدد هيكل المخرجات المطلوبة بدقة (Output Format).
5. اكتب الموجه النهائي باللغة التي تناسب سياق الفكرة، وركّز على جودة الصياغة دون أي مقدمات أو شروحات جانبية.
"""

@router.post("/enhance", response_model=EnhancePromptResponse)
async def enhance_prompt_with_openai(payload: EnhancePromptRequest):
    idea = payload.rawIdea.strip()
    if not idea:
        raise HTTPException(status_code=400, detail="الفكرة الأولية فارغة.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="لم يتم العثور على OPENAI_API_KEY في المتغيرات البيئية."
        )

    try:
        response = client.chat.completions.create(
            model=payload.model,
            temperature=payload.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ENGINEER_INSTRUCTION},
                {"role": "user", "content": f"حوّل هذه الفكرة إلى موجه احترافي متكامل:\n{idea}"}
            ]
        )
        
        enhanced_text = response.choices[0].message.content or ""
        return EnhancePromptResponse(
            enhancedPrompt=enhanced_text.strip(),
            modelUsed=payload.model
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"خطأ أثناء استدعاء OpenAI: {str(exc)}")
