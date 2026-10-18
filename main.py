
"""
Mowjn AI-Bayan | Demo Script
=============================

عرض شامل لميزات AdaptiveMemoryEngine مع تكامل Telemetry مجهول.

تشغيل: python main.py
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from adaptive_memory_engine import AdaptiveMemoryEngine
from telemetry import AnonymousTelemetry


# ─── إعداد الـ logging ─────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """طباعة عنوان قسم بشكل واضح."""
    print(f"\n{'═' * 50}")
    print(f" {title}")
    print(f"{'═' * 50}\n")


def print_results(results: list[dict]) -> None:
    """طباعة نتائج الاسترجاع بشكل منسّق."""
    if not results:
        print(" (لا توجد نتائج)")
        return

    for i, m in enumerate(results, 1):
        print(f" [{i}] ID: {m['id']}")
        print(
            f" التشابه: {m['semantic_score']} | "
            f"التضاؤل: {m['time_decay_factor']} | "
            f"النهائية: {m['final_score']}"
        )
        print(f" العمر: {m['days_old']} يوم")
        print(f" المحتوى: {m['content']}")
        print()


def main():
    # ─── إعداد مجلد مؤقت ────────────────────────────────────
    tmp_dir = Path(tempfile.gettempdir()) / "mowjn_demo"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ─── تهيئة Telemetry ────────────────────────────────────
    tel = AnonymousTelemetry(
        storage_path=str(tmp_dir / ".telemetry"),
        opt_in_required=False,
        sample_rate=0.1,
    )

    # ─── إنشاء المحرك ───────────────────────────────────────
    with AdaptiveMemoryEngine(
        decay_rate=0.05,
        similarity_threshold=0.4,
        dedup_threshold=0.92,
        max_store_size=100,
        decay_function="exponential",
        persist_directory=str(tmp_dir),
        collection_name="demo_memories",
    ) as engine:
        current_time = datetime.now(timezone.utc)

        # ═══════════════════════════════════════
        # ربط Telemetry بالمحرك
        # ═══════════════════════════════════════
        original_retrieve = engine.retrieve_by_text

        def tracked_retrieve(query: str, top_k: int = 3):
            start = time.time()
            result = original_retrieve(query, top_k=top_k)
            elapsed_ms = (time.time() - start) * 1000

            avg_score = (
                sum(r.get("semantic_score", 0) for r in result) / len(result)
                if result else 0
            )

            tel.record_search_event(
                query_length=len(query),
                results_count=len(result),
                avg_score=avg_score,
                search_time_ms=elapsed_ms,
                has_results=len(result) > 0,
            )

            return result

        engine.retrieve_by_text = tracked_retrieve

        # ═══════════════════════════════════════
        # 1) إضافة ذكريات فردية
        # ═══════════════════════════════════════
        print_section("🚀 تشغيل AdaptiveMemoryEngine")
        print_section("1️⃣ إضافة ذكريات فردية")

        engine.add_memory(
            memory_id="POL_001",
            content="قرار سابق بخصوص معايير الحوكمة والسياسات العامة",
            timestamp=current_time - timedelta(days=30),
        )

        engine.add_memory(
            memory_id="POL_002",
            content="تحديث أخير بخصوص معايير الحوكمة والسياسات العامة",
            timestamp=current_time - timedelta(days=1),
        )

        engine.add_memory(
            memory_id="RISK_001",
            content="تقييم المخاطر المالية للربع الثالث",
            timestamp=current_time - timedelta(days=5),
        )

        engine.add_memory(
            memory_id="COMP_001",
            content="نتائج المراجعة الداخلية لقسم الامتثال",
            timestamp=current_time - timedelta(days=10),
        )

        # ═══════════════════════════════════════
        # 2) إضافة دفعة (batch)
        # ═══════════════════════════════════════
        print_section("2️⃣ إضافة دفعة ذكريات")

        batch = [
            {"id": "BATCH_01", "content": "سياسة الخصوصية وحماية البيانات الجديدة"},
            {"id": "BATCH_02", "content": "إجراءات التعيين وإدارة المواهب"},
            {"id": "BATCH_03", "content": "خطة استمرارية الأعمال والطوارئ"},
        ]

        results = engine.add_memories_batch(batch)
        print(f" ✅ تم إضافة: {[k for k, v in results.items() if v]}")

        # ═══════════════════════════════════════
        # 3) استرجاع باستعلام نصّي
        # ═══════════════════════════════════════
        print_section("3️⃣ استرجاع بنص: 'الحوكمة والسياسات'")

        results = engine.retrieve_by_text("الحوكمة والسياسات", top_k=3)
        print_results(results)

        # ═══════════════════════════════════════
        # 4) استرجاع باستعلام نصّي ثاني
        # ═══════════════════════════════════════
        print_section("4️⃣ استرجاع بنص: 'المخاطر المالية'")

        results = engine.retrieve_by_text("المخاطر المالية", top_k=3)
        print_results(results)

        # ═══════════════════════════════════════
        # 5) اختبار اكتشاف التكرار
        # ═══════════════════════════════════════
        print_section("5️⃣ اختبار اكتشاف التكرار")

        print(" محاولة إضافة ذاكرة مكررة...\n")
        added = engine.add_memory(
            memory_id="POL_DUP",
            content="قرار سابق بخصوص معايير الحوكمة والسياسات العامة",
            timestamp=current_time,
        )
        print(f" نتيجة: {'✅ أُضيفت' if added else '❌ رُفضت (مكررة)'}")

        # ═══════════════════════════════════════
        # 6) إحصائيات
        # ═══════════════════════════════════════
        print_section("6️⃣ إحصائيات الذاكرة")

        stats = engine.get_stats()
        for k, v in stats.items():
            print(f" {k}: {v}")

        # ═══════════════════════════════════════
        # 7) تجربة التضاؤل الخطّي (linear)
        # ═══════════════════════════════════════
        print_section("7️⃣ مقارنة دوال التضاؤل")

        engine.close()

        with AdaptiveMemoryEngine(
            decay_rate=0.1,
            similarity_threshold=0.3,
            decay_function="linear",
            persist_directory=str(tmp_dir),
            collection_name="demo_linear",
        ) as linear_engine:
            linear_engine.add_memory(
                "L_001", "وثيقة قديمة جداً",
                timestamp=current_time - timedelta(days=9),
            )
            linear_engine.add_memory(
                "L_002", "وثيقة حديثة",
                timestamp=current_time - timedelta(days=1),
            )

            print(" استرجاع مع تضاؤل خطّي:\n")
            results = linear_engine.retrieve_by_text("وثيقة", top_k=2)
            print_results(results)

        # ═══════════════════════════════════════
        # 8) رؤى Telemetry المحلية
        # ═══════════════════════════════════════
        print_section("📊 رؤى Telemetry المحلية")

        insights = tel.get_local_insights()
        for key, value in insights.items():
            print(f" {key}: {value}")

        # ═══════════════════════════════════════
        # تنظيف
        # ═══════════════════════════════════════
        print_section("🧹 التنظيف")

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(" ✅ تم حذف المجلد المؤقت")

    print("\n 🎉 انتهت التجربة!\n")


if __name__ == "__main__":
    main()