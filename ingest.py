import os
import re
from typing import Optional
from google import genai

# المسارات المتوافقة مع مجلد docs المستخدم في rag_engine
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# جلب المفتاح مباشرة من بيئة Render
gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None


def _slugify(text: str) -> str:
    """تحويل العنوان إلى اسم ملف إنجليزي أو لاتيني صالح"""
    text = re.sub(r'[^\w\s-]', '', text).strip().lower()
    return re.sub(r'[-\s]+', '_', text) or "document"


def save_markdown_doc(title: str, content: str, original_url: str = "https://ai.google.dev/gemini-api/docs", filename: Optional[str] = None) -> str:
    """
    حفظ المستند بصيغة Markdown متوافقة بدقة مع فلاتر GroundedRAGEngine
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    # تحديد اسم الملف
    if not filename:
        filename = f"{_slugify(title)}.md"
    elif not filename.endswith(".md"):
        filename = f"{filename}.md"
        
    filepath = os.path.join(DOCS_DIR, filename)

    # التنسيق الذي يبحث عنه كود rag_engine بالضبط
    doc_template = f"""---
original_url: {original_url}
---

# {title}

{content.strip()}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(doc_template)

    return filepath


def smart_ingest(raw_text: str, title: str, original_url: str = "https://ai.google.dev/gemini-api/docs", optimize_with_ai: bool = False) -> str:
    """
    معالجة النص وحفظه، مع خيار إعادة هيكلته بذكاء اصطناعي لرفع دقة محرك الـ RAG
    """
    final_content = raw_text

    # إذا تم تفعيل الذكاء الاصطناعي وكان المفتاح موجوداً، يقوم بصياغة المحتوى لتسهيل البحث بـ BM25
    if optimize_with_ai and client:
        try:
            prompt = (
                "You are a technical documentation engineer. "
                "Structure and clean the following raw content into clear, well-formatted technical markdown paragraphs. "
                "Retain all technical facts, terms, and steps. Return only the cleaned content:\n\n"
                f"{raw_text}"
            )
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            if response and response.text:
                final_content = response.text.strip()
        except Exception:
            # في حال فشل الاتصال، يتم حفظ النص الأصلي تلقائياً كـ Fallback بدون توقف
            final_content = raw_text

    return save_markdown_doc(title=title, content=final_content, original_url=original_url)


if __name__ == "__main__":
    # تجربة معالجة ملف مستند تجريبي داخل مجلد docs
    sample_title = "Authentication and Keys"
    sample_text = "To use Gemini models, generate an API key via Google AI Studio and configure GEMINI_API_KEY in Render."
    save_markdown_doc(title=sample_title, content=sample_text)
