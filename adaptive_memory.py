
"""Adaptive Memory Engine module for intelligent context retrieval."""
from datetime import datetime, timedelta, timezone
import math

import numpy as np


class BasicMemoryEngine:
    """
    Engine for adaptive memory management with semantic similarity and time decay.

    Uses sentence-transformers for automatic Arabic-aware embedding generation.
    (basic/educational version)
    """

    # ✅ نموذج يدعم العربي ضمن 100+ لغة
    DEFAULT_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

    def __init__(self, decay_rate=0.01, similarity_threshold=0.5, embedding_model=None):
        """
        Initialize the Basic Memory Engine.

        :param decay_rate: معدل التضاؤل الزمني (كلما زاد، قل وزن الذكريات القديمة بسرعة)
        :param similarity_threshold: الحد الأدنى لمستوى التشابه القبول
        :param embedding_model: اسم نموذج sentence-transformers (اختياري)
        """
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold
        self.memory_store = []

        # ✅ تحميل نموذج الـ embedding
        model_name = embedding_model or self.DEFAULT_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(model_name)
            print(f"✅ تم تحميل نموذج التضمين: {model_name}")
        except ImportError:
            print(
                "⚠️ sentence-transformers غير مثبّت.\n"
                "ثبّته: pip install sentence-transformers"
            )
            self._embedder = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """تحويل نصوص إلى embeddings."""
        if self._embedder is None:
            raise RuntimeError(
                "لا يوجد نموذج embedding. ثبّت sentence-transformers."
            )
        vectors = self._embedder.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def _cosine_similarity(self, vec_a, vec_b):
        """حساب التشابه الدلالي بين متجهين"""
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def _calculate_time_decay(self, timestamp):
        """حساب معامل التضاؤل الزمني بناءً على الفارق بالأيام"""
        current_time = datetime.now(timezone.utc)
        time_diff = (current_time - timestamp).total_seconds() / (3600 * 24)
        # دالة الأس للتضاؤل التدريجي: e^(-lambda * t)
        decay_factor = math.exp(-self.decay_rate * max(0, time_diff))
        return decay_factor, time_diff

    def add_memory(self, memory_id, content, embedding=None, timestamp=None):
        """
        إضافة سجل جديد إلى الذاكرة.

        :param memory_id: معرف فريد للذكرى
        :param content: نص المحتوى
        :param embedding: متجه دلالي (اختياري — يُحسب تلقائياُ من content)
        :param timestamp: الطابع الزمني (اختياري — الوقت الحالي إذا لم يُمرَّر)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # ✅ حساب الـ embedding تلقائياً إذا لم يُمرَّر
        if embedding is None:
            embedding = self._embed([content])[0]

        self.memory_store.append({
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "timestamp": timestamp
        })

    def retrieve(self, query_embedding=None, query_text=None, top_k=3):
        """
        استرجاع أفضل النتائج بالدمج بين التشابه الدلالي والوزن الزمني.

        :param query_embedding: متجه الاستعلام (اختياري)
        :param query_text: نص الاستعلام (اختياري — يُحوَّل لمتجه تلقائياً)
        :param top_k: عدد النتائج المطلوبة
        :returns: قائمة نتائج مرتبة بالأولوية
        """
        # ✅ دعم الاستعلام بالنص مباشرة
        if query_text and not query_embedding:
            query_embedding = self._embed([query_text])[0]

        if query_embedding is None:
            raise ValueError("مرّر query_embedding أو query_text")

        results = []

        for item in self.memory_store:
            # 1. حساب التشابه الدلالي
            semantic_score = self._cosine_similarity(
                query_embedding, item["embedding"]
            )

            if semantic_score < self.similarity_threshold:
                continue

            # 2. حساب الوزن الزمني
            time_decay, days_old = self._calculate_time_decay(item["timestamp"])

            # 3. الدرجة النهائية = التشابه الدلالي * معامل التضاؤل الزمني
            final_score = semantic_score * time_decay

            results.append({
                "id": item["id"],
                "content": item["content"],
                "semantic_score": round(float(semantic_score), 4),
                "time_decay_factor": round(float(time_decay), 4),
                "final_score": round(float(final_score), 4),
                "days_old": round(days_old, 2)
            })

        # ترتيب النتائج من الأعلى إلى الأدنى بناءً على الدرجة النهائية
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]


# ==========================================
# تجربة سريعة للوظيفة (Example Usage)
# ==========================================
if __name__ == "__main__":
    engine = BasicMemoryEngine(decay_rate=0.05, similarity_threshold=0.4)
    current_time = datetime.now(timezone.utc)

    # ═══════════════════════════════════════
    # الآن نمرر النصوص فقط — الـ embedding يُحسب تلقائياً!
    # ═══════════════════════════════════════

    # إضافة سياق قديم (قبل 30 يوم)
    engine.add_memory(
        memory_id="MEM_001",
        content="قرار سابق بخصوص معايير الحوكمة والسياسات العامة",
        timestamp=current_time - timedelta(days=30)
    )

    # إضافة سياق حديث (قبل يوم واحد)
    engine.add_memory(
        memory_id="MEM_002",
        content="تحديث أخير بخصوص معايير الحوكمة والسياسات العامة",
        timestamp=current_time - timedelta(days=1)
    )

    # إضافة سياق مختلف تماماً
    engine.add_memory(
        memory_id="MEM_003",
        content="تقرير مالي عن أرباح الربع الثالث",
        timestamp=current_time - timedelta(days=5)
    )

    # ═══════════════════════════════════════
    # استعلام بالنص مباشرة — بدون متجه يدوي!
    # ═══════════════════════════════════════
    print("\n--- استرجاع: 'الحوكمة والسياسات' ---\n")
    top_memories = engine.retrieve(query_text="الحوكمة والسياسات", top_k=2)

    for memory in top_memories:
        print(
            f"ID: {memory['id']} | النهاية: {memory['final_score']} | "
            f"التشابه: {memory['semantic_score']} | "
            f"التضاؤل: {memory['time_decay_factor']} "
            f"(عمرها {memory['days_old']} يوم)"
        )
        print(f"المحتوى: {memory['content']}\n")

    # ═══════════════════════════════════════
    # استعلام ثاني بصياغة مختلفة
    # ═══════════════════════════════════════
    print("--- استرجاع: 'معايير الإدارة والامتثال' ---\n")
    top_memories = engine.retrieve(query_text="معايير الإدارة والامتثال", top_k=2)

    for memory in top_memories:
        print(
            f"ID: {memory['id']} | النهاية: {memory['final_score']} | "
            f"التشابه: {memory['semantic_score']} | "
            f"التضاؤل: {memory['time_decay_factor']} "
            f"(عمرها {memory['days_old']} يوم)"
        )
        print(f"المحتوى: {memory['content']}\n")