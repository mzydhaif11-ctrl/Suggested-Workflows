from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.services.ai_providers import (
    generate_with_gemini,
    generate_with_openai,
)

router = APIRouter(
    prefix="/api/ai",
    tags=["AI Integration"],
)


class AIRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)

    provider: Literal["gemini", "openai", "both"] = "both"


@router.post("/generate")
def generate(request: AIRequest):
    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=422,
            detail="Prompt cannot be empty",
        )

    results = {}

    providers = {
        "gemini": generate_with_gemini,
        "openai": generate_with_openai,
    }

    selected = (
        ["gemini", "openai"]
        if request.provider == "both"
        else [request.provider]
    )

    for name in selected:
        try:
            output = providers[name](prompt)

            results[name] = {
                "status": "success",
                "output": output,
            }

        except Exception:
            # لا نعرض تفاصيل الاستثناء أو بيانات الاتصال للعميل.
            results[name] = {
                "status": "error",
                "message": f"{name} request failed",
            }

    success_count = sum(
        result["status"] == "success"
        for result in results.values()
    )

    if success_count == 0:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "All selected AI providers failed",
                "results": results,
            },
        )

    return {
        "status": (
            "success"
            if success_count == len(results)
            else "partial_success"
        ),
        "providers": results,
    }
