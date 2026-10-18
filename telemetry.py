"""
Anonymous Telemetry Module for Mowjn AI-Bayan
================================================

نظام تتبع مجهول يحسن النظام تلقائياً بناءً على سلوك المستخدمين.
لا يرسل أي نص أو هوية — فقط إحصائيات رقمية مجهولة.

مبدأ الخصوصية:
  - لا نص استعلام
  - لا هوية مستخدم
  - لا IP address
  - فقط أرقام إحصائية مجمعة
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AnonymousTelemetry:
    """نظام تتبع مجهول لتحسين النظام تلقائياً."""

    def __init__(
        self,
        storage_path: str = ".telemetry_data",
        opt_in_required: bool = True,
        sample_rate: float = 0.1,  # 10% فقط تُرسل للمجتمع
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.opt_in_required = opt_in_required
        self.sample_rate = sample_rate

        # تحقق من موافقة المستخدم
        self._consent_file = self.storage_path / "consent.json"
        self._has_consent = self._check_consent()

    def _check_consent(self) -> bool:
        """التحقق من موافقة المستخدم على المشاركة المجهولة."""
        if not self.opt_in_required:
            return True
        if self._consent_file.exists():
            try:
                data = json.loads(self._consent_file.read_text())
                return data.get("consented", False)
            except Exception:
                pass
        return False

    def request_consent(self) -> bool:
        """طلب موافقة المستخدم للمشاركة المجهولة."""
        print("""
╔══════════════════════════════════════════════════════╗
║  🌊 موجة البيان — مشاركة مجهولة                   ║
║                                                      ║
║  هل تريد المساهمة بتحسين النظام للجميع؟             ║
║  • لن نرسل أي نص أو هوية                            ║
║  • فقط إحصائيات رقمية مجمعة                         ║
║  • يمكنك إلغاء الموافقة في أي وقت                    ║
║                                                      ║
║  [y/n]                                              ║
╚══════════════════════════════════════════════════════╝
        """)
        try:
            response = input("شارك مجهولاً لتحسين النظام؟ (y/n): ").strip().lower()
            consented = response == "y"
            self._consent_file.write_text(json.dumps({"consented": consented}))
            self._has_consent = consented
            return consented
        except (EOFError, KeyboardInterrupt):
            return False

    def record_search_event(
        self,
        query_length: int,
        results_count: int,
        avg_score: float,
        search_time_ms: float,
        has_results: bool,
    ) -> None:
        """
        تسجيل حدث بحث مجهول.

        :param query_length: طول الاستعلام بالرموز (ليس النص نفسه)
        :param results_count: عدد النتائج المسترجعة
        :param avg_score: متوسط درجة التشابه
        :param search_time_ms: وقت البحث بالميلي ثانية
        :param has_results: هل وجد نتائج؟
        """
        if not self._has_consent:
            return

        event = {
            "type": "search",
            "query_length": query_length,
            "results_count": results_count,
            "avg_score": round(avg_score, 4),
            "search_time_ms": round(search_time_ms, 2),
            "has_results": has_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._get_session_id(),
        }

        self._store_event(event)

    def record_recommendation_event(
        self,
        rec_count: int,
        clicked_index: Optional[int],
        session_duration_sec: float,
    ) -> None:
        """تسجيل حدث توصية مجهول."""
        if not self._has_consent:
            return

        event = {
            "type": "recommendation",
            "rec_count": rec_count,
            "clicked_index": clicked_index,
            "session_duration_sec": round(session_duration_sec, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self._get_session_id(),
        }

        self._store_event(event)

    def _store_event(self, event: Dict[str, Any]) -> None:
        """حفظ الحدث محلياً مع احتمالية إرساله للمجتمع."""
        events_file = self.storage_path / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        # إرسال عينة للمجتمع (10% فقط)
        if random.random() < self.sample_rate:
            self._send_to_community(event)

    def _get_session_id(self) -> str:
        """معرف جلسة مجهول (غير مرتبط بهوية المستخدم)."""
        session_file = self.storage_path / "session_id.txt"
        if session_file.exists():
            return session_file.read_text().strip()

        # إنشاء معرف عشوائي جديد
        session_id = hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:16]
        session_file.write_text(session_id)
        return session_id

    def _send_to_community(self, event: Dict[str, Any]) -> None:
        """إرسال الحدث المجهول إلى خادم المجتمع (اختياري)."""
        # هنا تقدر تربطه بخادم مركزي يجمع الإحصائيات
        # مثال: POST إلى API endpoint
        community_url = os.getenv("TELEMETRY_ENDPOINT")
        if not community_url:
            logger.debug("No telemetry endpoint configured; skipping community sync")
            return

        try:
            import requests
            response = requests.post(
                community_url,
                json=event,
                timeout=5,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code != 200:
                logger.warning(f"Community sync failed: {response.status_code}")
        except Exception as e:
            logger.debug(f"Community sync error (non-critical): {e}")

    def get_local_insights(self) -> Dict[str, Any]:
        """تحليل محلي للبيانات المجمعة — يفيد المستخدم مباشرة."""
        insights = {
            "total_searches": 0,
            "avg_results_per_search": 0,
            "avg_search_time_ms": 0,
            "no_result_rate": 0,
        }

        events_files = list(self.storage_path.glob("events_*.jsonl"))
        if not events_files:
            return insights

        all_events = []
        for ef in events_files:
            for line in ef.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    all_events.append(json.loads(line))

        search_events = [e for e in all_events if e["type"] == "search"]
        if not search_events:
            return insights

        insights["total_searches"] = len(search_events)
        insights["avg_results_per_search"] = round(
            sum(e["results_count"] for e in search_events) / len(search_events), 1
        )
        insights["avg_search_time_ms"] = round(
            sum(e["search_time_ms"] for e in search_events) / len(search_events), 1
        )
        insights["no_result_rate"] = round(
            sum(1 for e in search_events if not e["has_results"]) / len(search_events), 3
        )

        return insights


# ─── Integration Example ─────────────────────────────

def integrate_with_rag_engine(rag_engine, telemetry: AnonymousTelemetry):
    """تكامل بسيط مع GroundedRAGEngine."""
    original_answer = rag_engine.answer_query

    def enhanced_answer(query: str):
        start = time.time()
        result = original_answer(query)
        elapsed_ms = (time.time() - start) * 1000

        sources = result.get("sources", [])
        avg_score = (
            sum(s.get("score", 0) for s in sources) / len(sources)
            if sources else 0
        )

        telemetry.record_search_event(
            query_length=len(query),
            results_count=len(sources),
            avg_score=avg_score,
            search_time_ms=elapsed_ms,
            has_results=len(sources) > 0,
        )

        return result

    rag_engine.answer_query = enhanced_answer
    return rag_engine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tel = AnonymousTelemetry(opt_in_required=True)

    # طلب الموافقة
    if tel.request_consent():
        print("✅ شكراً لمساهمتك!")
    else:
        print("👍 لا بأس — النظام يشتغل بدون المشاركة")

    # مثال: تسجيل حدث بحث
    tel.record_search_event(
        query_length=25,
        results_count=3,
        avg_score=0.72,
        search_time_ms=145.3,
        has_results=True,
    )

    # عرض رؤى محلية
    insights = tel.get_local_insights()
    print(f"\n📊 رؤيك المحلية: {insights}")
