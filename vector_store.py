
"""
Vector Store Module for Mowjn AI-Bayan
=======================================

مخزن متجهات دلالية لقاعدة المعرفة الثابتة (ملفات docs/).
يستخدم sentence-transformers محلياً + ChromaDB للبحث الدلالي السريع.

لا يحتاج أي مفتاح API — يشتغل بالكامل على الجهاز المحلي.

Exports:
    DOCS_DIR        - مسار مجلد الوثائق
    get_api_key     - جلب مفتاح API (موجود للتوافق مع rag_engine.py)
    ensure_index    - بناء فهرس المتجهات (أو التحقق من وجوده)
    rebuild_index   - إعادة بناء الفهرس من الصفر
    vector_search   - بحث دلالي في قاعدة المعرفة
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"

# ✅ نموذج محلي يدعم العربي ضمن 100+ لغة
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "mowjn_knowledge")
PERSIST_DIR = str(BASE_DIR / ".chroma_knowledge")

# كاش للنموذج — يُحمَّل مرة وحدة فقط
_embedder = None


# ─── Helpers ────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:
    """جلب مفتاح API (موجود للتوافق مع rag_engine.py)."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _load_embedder():
    """تحميل نموذج التضمين (lazy loading)."""
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"🔄 جاري تحميل نموذج التضمين: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("✅ تم تحميل نموذج التضمين بنجاح")
    except ImportError:
        raise RuntimeError(
            "⚠️ sentence-transformers غير مثبّت.\n"
            "ثبّته: pip install sentence-transformers"
        )

    return _embedder


def _embed_text(text: str) -> list[float]:
    """تحويل نص إلى متجه دلالي باستخدام النموذج المحلي."""
    embedder = _load_embedder()
    vector = embedder.encode(text, show_progress_bar=False)
    return vector.tolist()


def _chunk_document(content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """تقسيم وثيقة طويلة إلى مقاطع متداخلة."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def _get_client():
    """إنشاء عميل ChromaDB دائم."""
    import chromadb
    from chromadb.config import Settings

    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client):
    """الحصول على مجموعة ChromaDB أو إنشاؤها."""
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ─── Core Functions ─────────────────────────────────────────────

def ensure_index(pages_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    بناء فهرس المتجهات إذا لم يكن موجوداُ.

    :param pages_dir: مسار مجلد الوثائق (اختياري)
    :returns: حالة الفهرس
    """
    target_dir = Path(pages_dir) if pages_dir else DOCS_DIR

    if not target_dir.is_dir():
        logger.warning(f"⚠️ مجلد الوثائق غير موجود: {target_dir}")
        return {"status": "skipped", "reason": "no_docs_dir"}

    client = _get_client()
    collection = _get_collection(client)
    existing_count = collection.count()

    if existing_count > 0:
        logger.info(f"✅ فهرس موجود بالفعل ({existing_count} مقطع)")
        return {"status": "exists", "count": existing_count}

    return _build_index(target_dir, client, collection)


def rebuild_index(docs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """إعادة بناء الفهرس بالكامل من الصفر."""
    target_dir = Path(docs_dir) if docs_dir else DOCS_DIR

    client = _get_client()

    # حذف القديم
    try:
        client.delete_collection(CHROMA_COLLECTION)
        logger.info("🗑️ تم حذف الفهرس القديم")
    except Exception:
        pass

    collection = _get_collection(client)
    return _build_index(target_dir, client, collection)


def _build_index(
    target_dir: Path,
    client,
    collection,
) -> Dict[str, Any]:
    """بناء الفهرس الداخلي."""
    files = sorted(
        list(target_dir.glob("*.md")) + list(target_dir.glob("*.txt")),
        key=lambda p: p.name.lower(),
    )

    ids, embeddings, documents, metadatas = [], [], [], []
    doc_count = 0
    chunk_id = 0

    for filepath in files:
        if filepath.name.lower() == "index.md":
            continue

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue

            chunks = _chunk_document(content)
            for chunk in chunks:
                chunk_id += 1
                mem_id = f"doc_{filepath.stem}_{chunk_id}"

                try:
                    emb = _embed_text(chunk)
                    ids.append(mem_id)
                    embeddings.append(emb)
                    documents.append(chunk)
                    metadatas.append({
                        "filename": filepath.name,
                        "source_file": str(filepath),
                    })
                    doc_count += 1
                except Exception as e:
                    logger.error(f"❌ فشل تضمين مقطع من {filepath.name}: {e}")

        except (OSError, UnicodeError) as e:
            logger.error(f"❌ فشل قراءة {filepath.name}: {e}")

    if ids:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"✅ تم فهرسة {doc_count} مقطع من {len(files)} ملف")

    return {"status": "success", "chunks_indexed": doc_count, "files_processed": len(files)}


def vector_search(
    query: str,
    top_k: int = 3,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    بحث دلالي في قاعدة المعرفة.

    :param query: نص الاستعلام
    :param top_k: عدد النتائج المطلوبة
    :param api_key: (غير مستخدم — موجود للتوافق مع rag_engine.py)
    :returns: قائمة نتائج البحث مرتبة بالأولوية
    """
    client = _get_client()
    collection = _get_collection(client)

    total = collection.count()
    if total == 0:
        logger.info("ℹ️ الفهرس فارغ — لا توجد نتائج")
        return []

    try:
        query_emb = _embed_text(query)
    except Exception as e:
        logger.error(f"❌ فشل تضمين الاستعلام: {e}")
        return []

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        score = round(1.0 - results["distances"][0][i], 4)
        meta = results["metadatas"][0][i]

        output.append({
            "title": meta.get("filename", "غير معروف"),
            "url": f"https://github.com/mzydhaif11-ctrl/Suggested-Workflows/blob/main/docs/{meta.get('filename', '')}",
            "filename": meta.get("filename", ""),
            "score": score,
            "excerpt": results["documents"][0][i][:450] + "...",
        })

    return output


# ─── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("🔧 Vector Store Management")
    print("=" * 40)
    print("1. ensure_index  - بناء الفهرس إذا لم يكن موجوداُ")
    print("2. rebuild_index - إعادة البناء من الصفر")
    print("3. search        - تجربة بحث")
    print()

    choice = input("اختر (1/2/3): ").strip()

    if choice == "1":
        result = ensure_index()
        print(f"نتيجة: {result}")
    elif choice == "2":
        result = rebuild_index()
        print(f"نتيجة: {result}")
    elif choice == "3":
        q = input("اكتب سؤالك: ")
        results = vector_search(q, top_k=3)
        for r in results:
            print(f"\n📄 {r['title']} (درجة: {r['score']})")
            print(r["excerpt"])