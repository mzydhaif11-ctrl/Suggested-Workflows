
"""
Ingest Module for Mowjn AI-Bayan
=================================

تغذية البيانات الأولية للنظام — فهرسة الوثائق محلياً.
لا يرسل أي بيانات لسيرفرات خارجية.

يستخدم:
  - sentence-transformers للتضمين الدلالي المحلي
  - ChromaDB للتخزين والفهرسة
"""

import logging
from pathlib import Path
from typing import Any, Dict

from vector_store import DOCS_DIR, ensure_index, rebuild_index

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def index_documents(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    فهرسة الوثائق في قاعدة المعرفة.

    :param force_rebuild: إذا True، يحذف الفهرس القديم ويبنيه من الصفر
    :returns: حالة عملية الفهرسة
    """
    if force_rebuild:
        logger.info("🔄 إعادة بناء الفهرس من الصفر...")
        result = rebuild_index(DOCS_DIR)
    else:
        logger.info("🔍 التحقق من وجود الفهرس...")
        result = ensure_index(pages_dir=DOCS_DIR)

    logger.info("Knowledge indexing result: %s", result)
    return result


# ─── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # دعم خيار --rebuild من سطر الأوامر
    force = "--rebuild" in sys.argv or "-r" in sys.argv

    print("🔧 Ingest Module — موجة البيان")
    print("=" * 40)
    print(f"مجلد الوثائق: {DOCS_DIR}")
    print(f"إعادة البناء: {'نعم' if force else 'لا'}")
    print()

    result = index_documents(force_rebuild=force)
    print(f"\n✅ النتيجة: {result}")