"""
Mowjn AI-Bayan | Adaptive Memory Engine
========================================
نسخة كاملة مع embedding مدمج (Sentence Transformers).

License: MIT
Author: Mowjn AI-Bayan Team
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveMemoryEngine:
    """
    محرك ذاكرة سياقية تكيفية مع embedding مدمج.
    ✅ يعطيه نص → هو يتكفّل بالباقي.
    """

    # ✅ نموذج embedder الافتراضي (يدعم العربية)
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        decay_rate: float = 0.01,
        similarity_threshold: float = 0.5,
        dedup_threshold: float = 0.95,
        max_store_size: int = 10_000,
        decay_function: str = "exponential",
        persist_directory: Optional[str] = None,
        collection_name: str = "mowjn_memories",
        fetch_multiplier: int = 3,
        embedding_model: Optional[str] = None,
    ):
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.max_store_size = max_store_size
        self.decay_function = decay_function
        self.fetch_multiplier = fetch_multiplier

        self._write_lock = threading.Lock()

        # ✅ تحميل نموذج الـ embedding (مرة وحدة)
        model_name = embedding_model or self.DEFAULT_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._embedder = None

        # ✅ ChromaDB setup
        if persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                Settings(anonymized_telemetry=False)
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ─── Context Manager ─────────────────────────────────────

    @contextmanager
    def session(self) -> Iterator["AdaptiveMemoryEngine"]:
        try:
            yield self
        finally:
            self.close()

    # ─── Core Helpers ────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """✅ تحويل نصوص إلى embeddings."""
        if self._embedder is None:
            raise RuntimeError(
                "No embedding model available. "
                "Install sentence-transformers or pass embeddings manually."
            )
        embeddings = self._embedder.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def _calculate_time_decay(self, timestamp: datetime) -> tuple[float, float]:
        now = datetime.now(timezone.utc)
        days_old = max(0, (now - timestamp).total_seconds() / 86400)

        if self.decay_function == "exponential":
            factor = math.exp(-self.decay_rate * days_old)
        elif self.decay_function == "linear":
            factor = max(0, 1 - self.decay_rate * days_old)
        elif self.decay_function == "step":
            step_days = 7
            factor = max(0, 1 - self.decay_rate * (days_old // step_days))
        else:
            factor = math.exp(-self.decay_rate * days_old)

        return float(factor), float(days_old)

    @staticmethod
    def _safe_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    # ─── Core Methods ────────────────────────────────────────

    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: Optional[list[float]] = None,  # ✅ صار اختياري
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        إضافة سجل جديد. 
        ✅ لو ما مرّرت embedding، ينحسب تلقائياً من content.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # ✅ احسب embedding لو ما موجود
        if embedding is None:
            embedding = self._embed([content])[0]

        # فحص التكرار
        existing = self.retrieve(embedding, top_k=1)
        if existing and existing[0]["semantic_score"] > self.dedup_threshold:
            logger.info(f"Duplicate detected for {memory_id}, skipping.")
            return False

        try:
            with self._write_lock:
                self._collection.upsert(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{"timestamp": self._safe_iso(timestamp)}],
                )
            logger.info(f"Memory added: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add memory {memory_id}: {e}")
            return False

    def retrieve(
        self,
        query: str,  # ✅ صار يستقبل نص بدل embedding
        query_embedding: Optional[list[float]] = None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        استرجاع أفضل النتائج.
        ✅ تقدر تمرّ نص أو embedding.
        """
        # ✅ احسب embedding لو ما موجود
        if query_embedding is None:
            query_embedding = self._embed([query])[0]

        total = self._collection.count()
        if total == 0:
            return []

        fetch_k = min(top_k * self.fetch_multiplier, total)

        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for i in range(len(raw["ids"][0])):
            semantic_score = 1.0 - raw["distances"][0][i]

            if semantic_score < self.similarity_threshold:
                continue

            ts_dt = datetime.fromisoformat(raw["metadatas"][0][i]["timestamp"])
            time_decay, days_old = self._calculate_time_decay(ts_dt)
            final_score = semantic_score * time_decay

            results.append({
                "id": raw["ids"][0][i],
                "content": raw["documents"][0][i],
                "semantic_score": round(semantic_score, 4),
                "time_decay_factor": round(time_decay, 4),
                "final_score": round(final_score, 4),
                "days_old": round(days_old, 2),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    # ─── Batch Operations ────────────────────────────────────

    def add_memories_batch(
        self, memories: list[dict]
    ) -> dict[str, bool]:
        """
        ✅ إضافة دفعة (مبسّطة). مرّر {'id', 'content'} فقط — بدون embedding.
        """
        # ✅ فصل اللي محتاج احتساب embedding
        needs_embedding = [
            (i, mem) for i, mem in enumerate(memories)
            if "embedding" not in mem
        ]

        if needs_embedding:
            texts = [mem["content"] for _, mem in needs_embedding]
            computed = self._embed(texts)
            for (i, mem), emb in zip(needs_embedding, computed):
                memories[i]["embedding"] = emb

        ids, embeddings, documents, metadatas = [], [], [], []

        for mem in memories:
            ids.append(mem["id"])
            embeddings.append(mem["embedding"])
            documents.append(mem["content"])
            ts = mem.get("timestamp") or datetime.now(timezone.utc)
            metadatas.append({"timestamp": self._safe_iso(ts)})

        try:
            with self._write_lock:
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            return {mid: True for mid in ids}
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            return {
                mem["id"]: self.add_memory(
                    memory_id=mem["id"],
                    content=mem["content"],
                    embedding=mem.get("embedding"),
                    timestamp=mem.get("timestamp"),
                )
                for mem in memories
            }

    # ─── Maintenance ─────────────────────────────────────────

    def cleanup_low_score(self, min_final_score: float = 0.1) -> list[str]:
        total = self._collection.count()
        if total == 0:
            return []

        raw = self._collection.get(include=["metadatas"], limit=total)

        ids_to_remove = []
        for i in range(len(raw["ids"])):
            ts_str = raw["metadatas"][i].get("timestamp")
            if not ts_str:
                continue
            ts_dt = datetime.fromisoformat(ts_str)
            time_decay, _ = self._calculate_time_decay(ts_dt)
            if time_decay < min_final_score:
                ids_to_remove.append(raw["ids"][i])

        if ids_to_remove:
            with self._write_lock:
                self._collection.delete(ids=ids_to_remove)
            logger.info(f"Cleaned up {len(ids_to_remove)} low-score memories.")

        return ids_to_remove

    def enforce_size_limit(self) -> list[str]:
        total = self._collection.count()
        if total <= self.max_store_size:
            return []

        excess = total - self.max_store_size
        raw = self._collection.get(include=["metadatas"], limit=total)

        indexed = [
            (i, raw["metadatas"][i].get("timestamp", ""))
            for i in range(len(raw["ids"]))
        ]
        indexed.sort(key=lambda x: x[1])
        ids_to_remove = [raw["ids"][i] for i, _ in indexed[:excess]]

        if ids_to_remove:
            with self._write_lock:
                self._collection.delete(ids=ids_to_remove)
            logger.info(
                f"Removed {len(ids_to_remove)} oldest memories (size limit)."
            )

        return ids_to_remove

    def get_stats(self) -> dict:
        count = self._collection.count()
        return {
            "total_memories": count,
            "max_store_size": self.max_store_size,
            "decay_function": self.decay_function,
            "utilization": f"{count / self.max_store_size:.1%}"
            if self.max_store_size else "N/A",
        }

    def close(self) -> None:
        logger.info("AdaptiveMemoryEngine closed.")


# ==========================================
# تجربة سريعة
# ==========================================
if __name__ == "__main__":
    tmp_dir = Path(tempfile.gettempdir()) / "test_mowjn_final"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # ✅ لاحظ: ما احتجنا نجيب embeddings يدوياً!
    with AdaptiveMemoryEngine(
        decay_rate=0.05,
        similarity_threshold=0.4,
        persist_directory=str(tmp_dir),
    ) as engine:

        current_time = datetime.now(timezone.utc)

        # ✅ إضافة نص فقط — بدون embedding
        engine.add_memory(
            memory_id="MEM_001",
            content="قرار سابق بخصوص معايير الحوكمة والسياسات العامة",
            timestamp=current_time - timedelta(days=30),
        )

        engine.add_memory(
            memory_id="MEM_002",
            content="تحديث أخير بخصوص معايير الحوكمة والسياسات العامة",
            timestamp=current_time - timedelta(days=1),
        )

        # ✅ استرجاع بنص فقط — النظام يحسب embedding تلقائياً
        top = engine.retrieve("معايير الحوكمة والسياسات", top_k=2)

        print("--- نتائج الاسترجاع ---")
        for m in top:
            print(
                f"ID: {m['id']} | النهاية: {m['final_score']} | "
                f"التشابه: {m['semantic_score']} | "
                f"عمرها {m['days_old']} يوم"
            )
            print(f"المحتوى: {m['content']}\n")

        print(f"الإجمالي: {engine.get_stats()['total_memories']}")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print("✅ تم")