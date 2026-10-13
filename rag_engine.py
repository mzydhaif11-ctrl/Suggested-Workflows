from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai

from vector_store import (
    DOCS_DIR,
    get_api_key,
    ensure_index,
    vector_search,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

WORKSPACE_PAGES_DIR = Path(
    os.getenv(
        "MAWJAH_DOCS_DIR",
        str(DOCS_DIR),
    )
)

GENERATION_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)


class GroundedRAGEngine:
    def __init__(
        self,
        pages_dir: Optional[str] = None,
        auto_index: bool = True,
    ):
        self.pages_dir = (
            Path(pages_dir)
            if pages_dir
            else WORKSPACE_PAGES_DIR
        )

        self.documents: List[Dict[str, Any]] = []

        self._load_lexical_documents()

        if auto_index and get_api_key():
            try:
                result = ensure_index(
                    self.pages_dir
                )

                logger.info(
                    "RAG vector index status: %s",
                    result,
                )

            except Exception:
                logger.exception(
                    "Could not initialize Chroma/Gemini index; "
                    "using keyword fallback."
                )

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        if not text:
            return []

        return re.findall(
            r"[a-zA-Z0-9_\u0600-\u06FF]+",
            str(text).lower(),
        )

    def _load_lexical_documents(self) -> None:
        self.documents = []

        if not self.pages_dir.is_dir():
            logger.warning(
                "Knowledge directory does not exist: %s",
                self.pages_dir,
            )
            return

        paths = sorted(
            [
                *self.pages_dir.glob("*.md"),
                *self.pages_dir.glob("*.txt"),
            ],
            key=lambda path: path.name.lower(),
        )

        for path in paths:
            if path.name.lower() == "index.md":
                continue

            try:
                content = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                if not content.strip():
                    continue

                url_match = re.search(
                    r"original_url\s*:\s*[\"']?"
                    r"(https?://[^\s\"'`]+)",
                    content,
                    flags=re.IGNORECASE,
                )

                source_url = (
                    url_match.group(1).rstrip(".,)")
                    if url_match
                    else (
                        "https://github.com/"
                        "mzydhaif11-ctrl/"
                        "Suggested-Workflows/blob/main/docs/"
                        + path.name
                    )
                )

                heading_match = re.search(
                    r"^#\s+(.+)$",
                    content,
                    flags=re.MULTILINE,
                )

                title = (
                    heading_match.group(1).strip().strip("# ")
                    if heading_match
                    else (
                        path.stem
                        .replace("_", " ")
                        .replace("-", " ")
                        .title()
                    )
                )

                tokens = self._tokenize(
                    title + " " + content
                )

                if not tokens:
                    continue

                self.documents.append({
                    "filename": path.name,
                    "filepath": str(path),
                    "title": title,
                    "url": source_url,
                    "content": content,
                    "tokens": Counter(tokens),
                    "token_count": len(tokens),
                })

            except (OSError, UnicodeError, ValueError):
                logger.exception(
                    "Could not read knowledge file %s",
                    path,
                )

    def index_documents(
        self,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        from Ingrst import index_documents

        result = index_documents(
            force_rebuild=force_rebuild
        )

        self._load_lexical_documents()

        return result

    def _keyword_search(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)

        if not query_tokens or not self.documents:
            return []

        query_lower = query.lower().strip()
        total_docs = len(self.documents)

        doc_freq = {
            token: sum(
                1
                for doc in self.documents
                if token in doc["tokens"]
            )
            for token in set(query_tokens)
        }

        average_length = (
            sum(
                doc["token_count"]
                for doc in self.documents
            )
            / max(total_docs, 1)
        )

        results: List[Dict[str, Any]] = []

        for doc in self.documents:
            score = 0.0

            lower_title = doc["title"].lower()
            doc_len = doc["token_count"]

            if query_lower and query_lower in lower_title:
                score += 20.0

            for token in query_tokens:
                if token in lower_title:
                    score += 4.0

                term_frequency = (
                    doc["tokens"].get(token, 0)
                )

                if term_frequency:
                    frequency = doc_freq.get(token, 1)

                    inverse_document_frequency = math.log(
                        1.0
                        + (
                            total_docs - frequency + 0.5
                        )
                        / (frequency + 0.5)
                    )

                    k1 = 1.2
                    b = 0.75

                    denominator = term_frequency + k1 * (
                        1 - b
                        + b * (
                            doc_len
                            / max(average_length, 1.0)
                        )
                    )

                    score += (
                        inverse_document_frequency
                        * (
                            term_frequency
                            * (k1 + 1)
                            / denominator
                        )
                    )

            if score > 0:
                excerpt = self._extract_relevant_excerpt(
                    doc["content"],
                    query_tokens,
                )

                results.append({
                    "title": doc["title"],
                    "url": doc["url"],
                    "filename": doc["filename"],
                    "score": round(score, 4),
                    "excerpt": excerpt,
                })

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:max(1, int(top_k))]

    def search(
        self,
        query: str,
        top_k: int = 3,
        api_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = str(query or "").strip()

        if not query:
            return []

        if api_key or get_api_key():
            try:
                results = vector_search(
                    query,
                    top_k=top_k,
                    api_key=api_key,
                )

                if results:
                    return results

            except Exception:
                logger.exception(
                    "Vector search failed; using keyword search."
                )

        return self._keyword_search(
            query,
            top_k=top_k,
        )

    def _extract_relevant_excerpt(
        self,
        content: str,
        query_tokens: List[str],
        max_chars: int = 450,
    ) -> str:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                content,
            )
            if len(paragraph.strip()) > 35
        ]

        if not paragraphs:
            paragraphs = (
                [content.strip()]
                if content.strip()
                else []
            )

        def paragraph_score(
            paragraph: str,
        ) -> int:
            lower = paragraph.lower()

            return sum(
                1
                for token in query_tokens
                if token in lower
            )

        best_paragraph = max(
            paragraphs,
            key=paragraph_score,
            default="",
        )

        clean_text = re.sub(
            r"^---\s*\n.*?\n---\s*\n",
            "",
            best_paragraph,
            flags=re.DOTALL,
        )

        clean_text = re.sub(
            r"#+\s*",
            "",
            clean_text,
        )

        clean_text = re.sub(
            r"\[([^\]]+)\]\([^\)]+\)",
            r"\1",
            clean_text,
        )

        clean_text = re.sub(
            r"\s+",
            " ",
            clean_text,
        ).strip()

        if len(clean_text) > max_chars:
            clean_text = (
                clean_text[:max_chars].rsplit(" ", 1)[0]
                + "..."
            )

        return clean_text

    def answer_query(
        self,
        query: str,
        provided_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = str(query or "").strip()

        if not query:
            return {
                "query": query,
                "answer": "اكتب سؤالك أولًا.",
                "sources": [],
            }

        api_key = provided_api_key or get_api_key()

        retrieved_docs = self.search(
            query,
            top_k=3,
            api_key=api_key,
        )

        if retrieved_docs and api_key:
            context_text = "\n\n".join(
                f"المستند: {doc['title']}\n"
                f"الرابط: {doc['url']}\n"
                f"المقتطف: {doc['excerpt']}"
                for doc in retrieved_docs
            )

            prompt = (
                "أنت مساعد دعم تقني لمنصة موجة. "
                "أجب باللغة العربية بوضوح. "
                "اعتمد على المقاطع التالية عند الإجابة، "
                "ولا تدّعِ أن معلومة واردة في المصادر "
                "إذا لم تكن موجودة فيها. "
                "إذا كانت المصادر غير كافية، صرّح بذلك. "
                "اختم بذكر المستندات وروابطها عند توفرها.\n\n"
                f"المصادر المسترجعة:\n{context_text}\n\n"
                f"سؤال المستخدم: {query}\n\n"
                "الإجابة:"
            )

            try:
                client = genai.Client(
                    api_key=api_key
                )

                response = client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt,
                )

                answer = (
                    getattr(response, "text", None) or ""
                ).strip()

                if not answer:
                    raise RuntimeError(
                        "Gemini returned an empty answer."
                    )

                return {
                    "query": query,
                    "answer": answer,
                    "sources": retrieved_docs,
                }

            except Exception:
                logger.exception(
                    "Gemini answer generation failed."
                )

                first = retrieved_docs[0]

                return {
                    "query": query,
                    "answer": (
                        "تعذّر توليد إجابة بالذكاء الاصطناعي حاليًا. "
                        "هذا أقرب مقتطف وجدته في قاعدة المعرفة:\n\n"
                        f"{first['excerpt']}\n\n"
                        f"المصدر: {first['url']}"
                    ),
                    "sources": retrieved_docs,
                }

        if retrieved_docs:
            first = retrieved_docs[0]

            return {
                "query": query,
                "answer": (
                    f"وجدت معلومات في مستند «{first['title']}»:\n\n"
                    f"{first['excerpt']}\n\n"
                    f"المصدر: {first['url']}"
                ),
                "sources": retrieved_docs,
            }

        if api_key:
            try:
                client = genai.Client(
                    api_key=api_key
                )

                response = client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=(
                        "أجب باللغة العربية. "
                        "لا تدّعِ أنك تستند إلى توثيق منصة موجة؛ "
                        "فلم يتم العثور على مصدر مناسب في قاعدة المعرفة.\n\n"
                        f"السؤال: {query}"
                    ),
                )

                answer = (
                    getattr(response, "text", None) or ""
                ).strip()

                if answer:
                    return {
                        "query": query,
                        "answer": answer,
                        "sources": [],
                        "grounded": False,
                    }

            except Exception:
                logger.exception(
                    "General Gemini fallback failed."
                )

        return {
            "query": query,
            "answer": (
                "لم أجد مقطعًا مناسبًا في ملفات المعرفة، "
                "ولم أتمكن من استخدام Gemini. "
                "تحقق من ملفات مجلد docs "
                "ومن إعداد GEMINI_API_KEY في Render."
            ),
            "sources": [],
        }
