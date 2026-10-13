import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

router = APIRouter(
    prefix="/api/prompt-studio",
    tags=["Prompt Studio"],
)

logger = logging.getLogger(__name__)


class EnhancePromptRequest(BaseModel):
    rawIdea: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="الفكرة الأولية",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="نموذج OpenAI",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
    )


class EnhancePromptResponse(BaseModel):
    enhancedPrompt: str
    modelUsed: str


SYSTEM_PROMPT_ENGINEER_INSTRUCTION = """
أنت خبير محترف في هندسة الأوامر.

مهمتك تحويل الفكرة الأولية إلى موجه احترافي ودقيق.

اتبع التعليمات التالية:
1. حدد دور النموذج والسياق بوضوح.
2. اكتب تعليمات واضحة ومتسلسلة.
3. حدد القيود والاستثناءات.
4. حدد شكل المخرجات المطلوبة.
5. استخدم اللغة المناسبة للفكرة.
6. أخرج الموجه النهائي فقط دون مقدمات.
"""


@router.post(
    "/enhance",
    response_model=EnhancePromptResponse,
)
async def enhance_prompt_with_openai(
    payload: EnhancePromptRequest,
):
    idea = payload.rawIdea.strip()

    if not idea:
        raise HTTPException(
            status_code=400,
            detail="الفكرة الأولية فارغة.",
        )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="خدمة OpenAI غير مهيأة.",
        )

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=payload.model,
            temperature=payload.temperature,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_ENGINEER_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": (
                        "حوّل هذه الفكرة إلى موجه احترافي متكامل:\n"
                        f"{idea}"
                    ),
                },
            ],
        )

        enhanced_text = (
            response.choices[0].message.content or ""
        ).strip()

        if not enhanced_text:
            raise HTTPException(
                status_code=502,
                detail="لم يُرجع النموذج موجهًا.",
            )

        return EnhancePromptResponse(
            enhancedPrompt=enhanced_text,
            modelUsed=response.model,
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "OpenAI prompt enhancement failed."
        )

        raise HTTPException(
            status_code=502,
            detail="تعذر تحسين الفكرة حاليًا.",
        )
