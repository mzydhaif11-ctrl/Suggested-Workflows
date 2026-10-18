
"""
Grounded RAG Engine for Mowjn AI-Bayan
=======================================

محرك استرجاع المعلومات المعزز — محلي بالكامل.
لا يرسل أي بيانات لسيرفرات خارجية.

يستخدم:
  - Vector Search (sentence-transformers + ChromaDB) للبحث الدلالي
  - Keyword Search (BM25-like) كخيار احتياطي
  - توليد إجابات محلية من المقاطع المسترجعة
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from vector_store import (
    DOCS_DIR,
    ensure_index,
    vector_search,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_PAGES_DIR = Path(
    os.getenv("MAWJAH_DOCS_DIR", str(DOCS_DIR))
)


class GroundedRAGEngine:
    """محرك RAG محلي يعتمد على البحث الدلالي والكلمات المفتاحية."""

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

        if auto_index:
            try:
                result = ensure_index(self.pages_dir)
                logger.info("RAG vector index status: %s", result)
            except Exception:
                logger.exception(
                    "Could not initialize vector index; "
                    "using keyword fallback."
                )

    # ─── Tokenization ──────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(
            r"[a-zA-Z0-9_\u0600-\u06FF]+",
            str(text).lower(),
        )

    # ─── Lexical Document Loading ──────────────────────────

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
                    encoding="utf-8", errors="replace"
                )
                if not content.strip():
                    continue

                url_match = re.search(
                    r"original_url\s*:\s*[\"']?"
                    r"(https?://[^\s\"'\`]+)",
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

                tokens = self._tokenize(title + " " + content)
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
                    "Could not read knowledge file %s", path
                )

    # ─── Indexing ──────────────────────────────────────────

    def index_documents(self, force_rebuild: bool = False) -> Dict[str, Any]:
        from ingest import index_documents as ingest_index

        result = ingest_index(force_rebuild=force_rebuild)
        self._load_lexical_documents()
        return result

    # ─── Keyword Search (BM25-like) ────────────────────────

    def _keyword_search(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []

        query_lower = query.lower().strip()
        total_docs = len(self.documents)
        doc_freq = {
            token: sum(
                1 for doc in self.documents if token in doc["tokens"]
            )
            for token in set(query_tokens)
        }
        average_length = (
            sum(doc["token_count"] for doc in self.documents)
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

                term_frequency = doc["tokens"].get(token, 0)
                if term_frequency:
                    frequency = doc_freq.get(token, 1)
                    idf = math.log(
                        1.0 + (total_docs - frequency + 0.5) / (frequency + 0.5)
                    )
                    k1 = 1.2
                    b = 0.75
                    denominator = term_frequency + k1 * (
                        1 - b + b * (doc_len / max(average_length, 1.0))
                    )
                    score += idf * (term_frequency * (k1 + 1) / denominator)

            if score > 0:
                excerpt = self._extract_relevant_excerpt(
                    doc["content"], query_tokens
                )
                results.append({
                    "title": doc["title"],
                    "url": doc["url"],
                    "filename": doc["filename"],
                    "score": round(score, 4),
                    "excerpt": excerpt,
                })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:max(1, int(top_k))]

    # ─── Unified Search ────────────────────────────────────

    def search(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        بحث موحّد: يحاول البحث الدلالي أولاً، ثم يعود للكلمات المفتاحية.
        لا يحتاج مفتاح API — كل شيء محلي.
        """
        query = str(query or "").strip()
        if not query:
            return []

        # محاولة البحث الدلالي أولاً (محلي بالكامل)
        try:
            results = vector_search(query, top_k=top_k)
            if results:
                return results
        except Exception:
            logger.exception("Vector search failed; using keyword search.")

        # العودة للبحث بالكلمات المفتاحية
        return self._keyword_search(query, top_k=top_k)

    # ─── Excerpt Extraction ────────────────────────────────

    def _extract_relevant_excerpt(
        self,
        content: str,
        query_tokens: List[str],
        max_chars: int = 450,
    ) -> str:
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", content)
            if len(p.strip()) > 35
        ]
        if not paragraphs:
            paragraphs = ([content.strip()] if content.strip() else [])

        def paragraph_score(paragraph: str) -> int:
            lower = paragraph.lower()
            return sum(1 for t in query_tokens if t in lower)

        best_paragraph = max(paragraphs, key=paragraph_score, default="")

        clean_text = re.sub(
            r"^---\s*\n.*?\n---\s*\n", "", best_paragraph, flags=re.DOTALL
        )
        clean_text = re.sub(r"#+\s*", "", clean_text)
        clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars].rsplit(" ", 1)[0] + "..."

        return clean_text

    # ─── Local Answer Generation ───────────────────────────

    def _build_local_answer(
        self, query: str, retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """بناء إجابة محلية من المقاطع المسترجعة بدون إرسال لـ Google."""
        if not retrieved_docs:
            return (
                "لم أجد معلومات كافية في قاعدة المعرفة للإجابة على سؤالك.\n"
                "حاول صياغة السؤال بطريقة مختلفة أو أضف وثائق جديدة إلى مجلد docs/."
            )

        parts = []
        parts.append(f"بناءً على الوثائق المتاحة، إليك الإجابة:")
        parts.append("")

        for i, doc in enumerate(retrieved_docs, 1):
            parts.append(f"📄 المصدر {i}: {doc['title']}")
            parts.append(f"   {doc['excerpt']}")
            parts.append("")

        # إضافة ملخص مبني على الكلمات المفتاحية المشتركة
        query_tokens = set(self._tokenize(query))
        relevant_excerpts = []
        for doc in retrieved_docs:
            excerpt = doc.get("excerpt", "")
            if any(t in excerpt.lower() for t in query_tokens):
                relevant_excerpts.append(excerpt)

        if relevant_excerpts:
            parts.append("📝 الملخص:")
            parts.append(relevant_excerpts[0][:300])
            parts.append("")

        return "\n".join(parts)

    # ─── Main Query Interface ──────────────────────────────

    def answer_query(
        self, query: str
    ) -> Dict[str, Any]:
        """
        واجهة الاستعلام الرئيسية — محلي بالكامل.

        :param query: سؤال المستخدم
        :returns: dict مع الإجابة والمصادر
        """
        query = str(query or "").strip()
        if not query:
            return {
                "query": query,
                "answer": "اكتب سؤالك أولًا.",
                "sources": [],
            }

        # 1. استرجاع الوثائق ذات الصلة
        retrieved_docs = self.search(query, top_k=3)

        # 2. بناء إجابة محلية
        answer = self._build_local_answer(query, retrieved_docs)

        return {
            "query": query,
            "answer": answer,
            "sources": retrieved_docs,
            "grounded": len(retrieved_docs) > 0,
        }


# ─── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    engine = GroundedRAGEngine(auto_index=True)

    print("\n🔍 محرك RAG المحلي — موجة البيان")
    print("=" * 50)
    print("اكتب سؤالك (أو 'exit' للخروج)\n")

    while True:
        try:
            q = input("❓ سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 وداعاً!")
            break

        if q.lower() in ("exit", "quit", "خروج"):
            print("👋 وداعاً!")
            break

        if not q:
            continue

        result = engine.answer_query(q)
        print(f"\n{result['answer']}\n")

        if result["sources"]:
            print("المصادر:")
            for s in result["sources"]:
                print(f"  • {s['title']} (درجة: {s['score']})")
            print()