"""
Mowjn AI-Bayan | Adaptive Memory Engine
========================================
محرك ذاكرة سياقية تكيفية مع تضاؤل زمني وتخزين دائم على ChromaDB.

License: MIT
Author: Mowjn AI-Bayan Team
"""

from __future__ import annotations

import logging
import math
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveMemoryEngine:
    """
    محرك ذاكرة سياقية تكيفية:
      - تضاؤل زمني (exponential / linear / step)
      - بحث دلالي عبر فهرس HNSW في ChromaDB
      - تخزين دائم على القرص
      - thread-safe للكتابة
      - embedding مدمج (يدعم العربي)
    """

    # ✅ نموذج افتراضي مختار لقوة دعمه للعربي ضمن 100+ لغة
    DEFAULT_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

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
        """
        :param decay_rate: معدل التضاؤل الزمني
        :param similarity_threshold: الحد الأدنى للتشابه القبول
        :param dedup_threshold: حد اكتشاف التكرار
        :param max_store_size: الحد الأقصى لعدد الذكريات
        :param decay_function: دالة التضاؤل
        :param persist_directory: مسار مجلد التخزين الدائم (None = مؤقت)
        :param collection_name: اسم مجموعة ChromaDB
        :param fetch_multiplier: مضاعف جلب النتائج قبل الفلترة الزمنية
        :param embedding_model: اسم نموذج sentence-transformers
        """
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.max_store_size = max_store_size
        self.decay_function = decay_function
        self.fetch_multiplier = fetch_multiplier

        self._write_lock = threading.Lock()

        # ✅ تحميل نموذج الـ embedding
        model_name = embedding_model or self.DEFAULT_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning(
                "⚠️ sentence-transformers غير مثبّت. "
                "ثبّته: pip install sentence-transformers"
            )
            self._embedder = None

        # ✅ ChromaDB: دائم أو مؤقت
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

    # ─── Helpers ─────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """تحويل نصوص إلى embeddings."""
        if self._embedder is None:
            raise RuntimeError(
                "لا يوجد نموذج embedding. ثبّت sentence-transformers "
                "أو مرّر embedding يدوي."
            )
        vectors = self._embedder.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def _calculate_time_decay(self, timestamp: datetime) -> tuple[float, float]:
        now = datetime.now(timezone.utc)
        days_old = max(0, (now - timestamp).total_seconds() / 86400)

        if self.decay_function == "exponential":
            factor = math.exp(-self.decay_rate * days_old)
        elif self.decay_function == "linear":
            factor = max(0, 1 - self.decay_rate * days_old)
        elif self.decay_function == "step":
            factor = max(0, 1 - self.decay_rate * (days_old // 7))
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
        embedding: Optional[list[float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """إضافة سجل. لو ما مرّرت embedding يُحسب تلقائياً من content."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if embedding is None:
            embedding = self._embed([content])[0]

        # فحص التكرار
        existing = self.retrieve_by_embedding(embedding, top_k=1)
        if existing and existing[0]["semantic_score"] > self.dedup_threshold:
            logger.info(f"🔁 Duplicate: {memory_id} — تم التخطّي")
            return False

        try:
            with self._write_lock:
                self._collection.upsert(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{"timestamp": self._safe_iso(timestamp)}],
                )
            logger.info(f"➕ Memory added: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add {memory_id}: {e}")
            return False

    def retrieve_by_text(
        self, query: str, top_k: int = 3
    ) -> list[dict]:
        """استرجاع باستعلام نصّي."""
        embedding = self._embed([query])[0]
        return self.retrieve_by_embedding(embedding, top_k)

    def retrieve_by_embedding(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[dict]:
        """استرجاع باستعلام متجهي."""
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

    # alias للبحث السريع
    retrieve = retrieve_by_text

    # ─── Batch Operations ────────────────────────────────────

    def add_memories_batch(self, memories: list[dict]) -> dict[str, bool]:
        """إضافة دفعة. مرّر {'id', 'content'} بدون embedding — يتكفّل بها."""
        needs_emb = [
            (i, m) for i, m in enumerate(memories) if "embedding" not in m
        ]

        if needs_emb:
            texts = [m["content"] for _, m in needs_emb]
            computed = self._embed(texts)
            for (i, _), emb in zip(needs_emb, computed):
                memories[i]["embedding"] = emb

        ids, embeddings, documents, metadatas = [], [], [], []
        for m in memories:
            ids.append(m["id"])
            embeddings.append(m["embedding"])
            documents.append(m["content"])
            ts = m.get("timestamp") or datetime.now(timezone.utc)
            metadatas.append({"timestamp": self._safe_iso(ts)})

        try:
            with self._write_lock:
                self._collection.upsert(
                    ids=ids, embeddings=embeddings,
                    documents=documents, metadatas=metadatas,
                )
            return {mid: True for mid in ids}
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            return {
                m["id"]: self.add_memory(
                    memory_id=m["id"], content=m["content"],
                    embedding=m.get("embedding"), timestamp=m.get("timestamp"),
                )
                for m in memories
            }

    # ─── Maintenance ─────────────────────────────────────────

    def cleanup_low_score(self, min_final_score: float = 0.1) -> list[str]:
        """حذف الذكريات المنخفضة (تقييم زمني عادل)."""
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
            logger.info(f"🧹 Cleaned {len(ids_to_remove)} low-score memories")

        return ids_to_remove

    def enforce_size_limit(self) -> list[str]:
        """إنفاذ الحد الأقصى: يحذف الأقدم عند التجاوز."""
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
            logger.info(f"📦 Removed {len(ids_to_remove)} oldest memories")

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
        logger.info("🔒 AdaptiveMemoryEngine closed")