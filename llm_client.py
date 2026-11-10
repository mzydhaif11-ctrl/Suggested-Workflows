"""Small synchronous client for the local Ollama chat API."""

import os
from typing import Optional

import requests


class OllamaClientError(RuntimeError):
    """Raised when Ollama cannot produce a response."""


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "iKhalid/ALLaM:7b")
        self.timeout = timeout or float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

    def generate(self, system_message: str, user_message: str) -> str:
        """Generate a non-streaming response using Ollama's /api/chat endpoint."""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    "options": {"temperature": 0.2},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise OllamaClientError(
                "تعذر الاتصال بـ Ollama. تحقق من تشغيله ومن عنوان الخدمة."
            ) from exc
        except ValueError as exc:
            raise OllamaClientError("أعاد Ollama استجابة غير صالحة.") from exc

        message = payload.get("message", {}).get("content", "").strip()
        if not message:
            raise OllamaClientError(
                "لم يُنتج النموذج إجابة. تحقق من تثبيت النموذج واسمه في إعدادات Ollama."
            )
        return message


def get_llm_client() -> OllamaClient:
    """Build a client from environment configuration."""
    return OllamaClient()
