
"""
Mowjn AI-Bayan | Adaptive Memory Engine
========================================
نظام ذاكرة سياقية تكيفية لمنصة موجة البيان.
يدعم التشابه الدلالي مع تضاؤل زمني، تخزين دائم، واكتشاف التكرار.

License: MIT
Author: Mowjn AI-Bayan Team
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveMemoryEngine:
    """
    محرك ذاكرة سياقية تكيفية يجمع بين التشابه الدلالي والوزن الزمني.

    يدعم:
      - تخزين دائم عبر SQLite
      - اكتشاف التكرار (deduplication)
      - عمليات دفعات (batch operations)
      - دوال تضاؤل قابلة للتبديل
      - تنظيف تلقائي للذكريات المنخفضة الدرجة
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        similarity_threshold: float = 0.5,
        dedup_threshold: float = 0.95,
        max_store_size: int = 10_000,
        decay_function: str = "exponential",
        db_path: Optional[str] = None,
    ):
        """
        :param decay_rate: معدل التضاؤل الزمني
        :param similarity_threshold: الحد الأدنى للتشابه القبول
        :param dedup_threshold: حد اكتشاف التكرار
        :param max_store_size: الحد الأقصى لعدد الذكريات
        :param decay_function: نوع دالة التضاؤل ('exponential' | 'linear' | 'step')
        :param db_path: مسار ملف قاعدة البيانات (افتراضي: mowjn_memory.db)
        """
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.max_store_size = max_store_size
        self.decay_function = decay_function
        self.db_path = db_path or "mowjn_memory.db"
        self._db: Optional[sqlite3.Connection] = None
        self._ensure_db()

    # ─── Database Layer ──────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _ensure_db(self) -> None:
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp REAL NOT NULL,
                created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_timestamp ON memories(timestamp)")
        conn.commit()

    def _serialize_embedding(self, emb: list[float]) -> bytes:
        return np.array(emb, dtype=np.float32).tobytes()

    def _deserialize_embedding(self, blob: bytes) -> list[float]:
        return np.frombuffer(blob, dtype=np.float32).tolist()

    # ─── Core Methods ────────────────────────────────────────

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """حساب التشابه الجتاهي بين متجهين."""
        a = np.array(vec_a, dtype=np.float64)
        b = np.array(vec_b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _calculate_time_decay(self, timestamp: datetime) -> tuple[float, float]:
        """حساب معامل التضاؤل الزمني بناءً على الفارق بالأيام."""
        now = datetime.now(timezone.utc)
        days_old = (now - timestamp).total_seconds() / (3600 * 24)
        days_old = max(0, days_old)

        if self.decay_function == "exponential":
            factor = math.exp(-self.decay_rate * days_old)
        elif self.decay_function == "linear":
            factor = max(0, 1 - self.decay_rate * days_old)
        elif self.decay_function == "step":
            step_days = 7  # كل أسبوع تنخفض الدرجة
            factor = max(0, 1 - self.decay_rate * (days_old // step_days))
        else:
            factor = math.exp(-self.decay_rate * days_old)

        return float(factor), float(days_old)

    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        إضافة سجل جديد إلى الذاكرة مع فحص التكرار.

        :return: True إذا أُضيف بنجاح، False إذا كان مكرراُ
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Deduplication check
        existing = self.retrieve(embedding, top_k=1)
        if existing and existing[0]["semantic_score"] > self.dedup_threshold:
            logger.info(f"Duplicate detected for {memory_id}, skipping.")
            return False

        ts_float = timestamp.timestamp()
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO memories (id, content, embedding, timestamp) VALUES (?, ?, ?, ?)",
                (memory_id, content, self._serialize_embedding(embedding), ts_float),
            )
            conn.commit()
            logger.info(f"Memory added: {memory_id}")
            return True
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to add memory {memory_id}: {e}")
            return False

    def retrieve(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[dict]:
        """استرجاع أفضل النتائج بالدمج بين التشابه الدلالي والوزن الزمني."""
        conn = self._get_connection()
        rows = conn.execute("SELECT id, content, embedding, timestamp FROM memories").fetchall()

        results = []
        for row in rows:
            item_emb = self._deserialize_embedding(row["embedding"])
            semantic_score = self._cosine_similarity(query_embedding, item_emb)

            if semantic_score < self.similarity_threshold:
                continue

            ts_dt = datetime.fromtimestamp(row["timestamp"], tz=timezone.utc)
            time_decay, days_old = self._calculate_time_decay(ts_dt)
            final_score = semantic_score * time_decay

            results.append({
                "id": row["id"],
                "content": row["content"],
                "semantic_score": round(float(semantic_score), 4),
                "time_decay_factor": round(float(time_decay), 4),
                "final_score": round(float(final_score), 4),
                "days_old": round(days_old, 2),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    # ─── Batch Operations ────────────────────────────────────

    def add_memories_batch(self, memories: list[dict]) -> dict[str, bool]:
        """
        إضافة دفعة من الذكريات.

        :param memories: قائمة بـ {'id', 'content', 'embedding', 'timestamp'}
        :return: قاموس بالنتائج لكل ID
        """
        results = {}
        for mem in memories:
            success = self.add_memory(
                memory_id=mem["id"],
                content=mem["content"],
                embedding=mem["embedding"],
                timestamp=mem.get("timestamp"),
            )
            results[mem["id"]] = success
        return results

    # ─── Maintenance ─────────────────────────────────────────

    def cleanup_low_score(
        self, query_embedding: list[float], min_final_score: float = 0.1
    ) -> list[str]:
        """حذف الذكريات ذات الدرجة النهائية المنخفضة."""
        results = self.retrieve(query_embedding, top_k=self.max_store_size)
        ids_to_remove = [r["id"] for r in results if r["final_score"] < min_final_score]

        conn = self._get_connection()
        removed = []
        for mid in ids_to_remove:
            conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
            removed.append(mid)

        conn.commit()
        logger.info(f"Cleaned up {len(removed)} low-score memories.")
        return removed

    def get_stats(self) -> dict:
        """إحصائيات عامة عن الذاكرة."""
        conn = self._get_connection()
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        oldest = conn.execute(
            "SELECT MIN(timestamp) FROM memories"
        ).fetchone()[0]
        newest = conn.execute(
            "SELECT MAX(timestamp) FROM memories"
        ).fetchone()[0]

        return {
            "total_memories": count,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest,
            "max_store_size": self.max_store_size,
            "decay_function": self.decay_function,
        }

    def close(self) -> None:
        """إغلاق الاتصال بقاعدة البيانات."""
        if self._db:
            self._db.close()
            self._db = None


# ==========================================
# تجربة سريعة (Example Usage)
# ==========================================
if __name__ == "__main__":
    import tempfile
    import os

    # استخدام قاعدة بيانات مؤقتة للاختبار
    tmp_db = os.path.join(tempfile.gettempdir(), "test_mowjn.db")

    engine = AdaptiveMemoryEngine(
        decay_rate=0.05,
        similarity_threshold=0.4,
        db_path=tmp_db,
    )

    current_time = datetime.now(timezone.utc)
    base_vector = [0.12, 0.85, 0.44, 0.21, 0.90]

    # إضافة سياق قديم
    engine.add_memory(
        memory_id="MEM_001",
        content="قرار سابق بخصوص معايير الحوكمة والسياسات العامة",
        embedding=[0.11, 0.84, 0.43, 0.20, 0.89],
        timestamp=current_time - timedelta(days=30),
    )

    # إضافة سياق حديث
    engine.add_memory(
        memory_id="MEM_002",
        content="تحديث أخير بخصوص معايير الحوكمة والسياسات العامة",
        embedding=[0.10, 0.82, 0.45, 0.22, 0.88],
        timestamp=current_time - timedelta(days=1),
    )

    # اختبار الاسترجاع
    top_memories = engine.retrieve(base_vector, top_k=2)
    print("--- نتائج الاسترجاع التكيفي ---")
    for memory in top_memories:
        print(
            f"ID: {memory['id']} | النهاية: {memory['final_score']} | "
            f"التشابه: {memory['semantic_score']} | "
            f"التضاؤل: {memory['time_decay_factor']} "
            f"(عمرها {memory['days_old']} يوم)"
        )
        print(f"المحتوى: {memory['content']}\n")

    # إحصائيات
    stats = engine.get_stats()
    print(f"إجمالي الذكريات: {stats['total_memories']}")

    # تنظيف
    engine.close()
    os.remove(tmp_db)
    print("تم التنظيف ✅")
