import numpy as np
from datetime import datetime, timezone
import math

class AdaptiveMemoryEngine:
    def __init__(self, decay_rate=0.01, similarity_threshold=0.5):
        """
        :param decay_rate: معدل التضاؤل الزمني (كلما زاد، قل وزن الذكريات القديمة بسرعة)
        :param similarity_threshold: الحد الأدنى لمستوى التشابه القبول
        """
        self.decay_rate = decay_rate
        self.similarity_threshold = similarity_threshold
        self.memory_store = []

    def _cosine_similarity(self, vec_a, vec_b):
        """حساب التشابه الجتاهي بين متجهين"""
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _calculate_time_decay(self, timestamp):
        """حساب معامل التضاؤل الزمني بناءً على الفارق الأيام"""
        now = datetime.now(timezone.utc)
        time_diff = (now - timestamp).total_seconds() / (3600 * 24) # تحويل إلى أيام
        # دالة الأس للتضاؤل التدريجي: e^(-lambda * t)
        decay_factor = math.exp(-self.decay_rate * max(0, time_diff))
        return decay_factor, time_diff

    def add_memory(self, memory_id, content, embedding, timestamp=None):
        """إضافة سجل جديد إلى الذاكرة"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        self.memory_store.append({
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "timestamp": timestamp
        })

    def retrieve(self, query_embedding, top_k=3):
        """استرجاع أفضل النتائج بالدمج بين التشابه الدلالي والوزن الزمني"""
        results = []

        for item in self.memory_store:
            # 1. حساب التشابه الدلالي
            semantic_score = self._cosine_similarity(query_embedding, item["embedding"])
            
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
    from datetime import timedelta

    engine = AdaptiveMemoryEngine(decay_rate=0.05, similarity_threshold=0.4)
    now = datetime.now(timezone.utc)

    # نموذج لمتجهات وهمية للعرض (شبه مطابقة)
    base_vector = [0.12, 0.85, 0.44, 0.21, 0.90]

    # إضافة سياق قديم (قبل 30 يوم)
    engine.add_memory(
        memory_id="MEM_001",
        content="قرار سابق بخصوص معايير الحوكمة والسياسات العامة",
        embedding=[0.11, 0.84, 0.43, 0.20, 0.89],
        timestamp=now - timedelta(days=30)
    )

    # إضافة سياق حديث (قبل يوم واحد)
    engine.add_memory(
        memory_id="MEM_002",
        content="تحديث أخير بخصوص معايير الحوكمة والسياسات العامة",
        embedding=[0.10, 0.82, 0.45, 0.22, 0.88],
        timestamp=now - timedelta(days=1)
    )

    # استعلام جديد
    query_vec = [0.12, 0.85, 0.44, 0.21, 0.90]
    top_memories = engine.retrieve(query_vec, top_k=2)

    print("--- نتائج الاسترجاع التكيفي ---")
    for m in top_memories:
        print(f"ID: {m['id']} | النهاية: {m['final_score']} | التشابه: {m['semantic_score']} | التضاؤل الزمني: {m['time_decay_factor']} (عمرها {m['days_old']} يوم)")
        print(f"المحتوى: {m['content']}\n")
