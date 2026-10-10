<div align="center">

# 🌊 موجة البيان | Mowjn AI-Bayan

### نظام ذاكرة سياقية تكيفية لمنصة ذكاء اصطناعي

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)
[![Gemini API](https://img.shields.io/badge/API-Gemini%203.8-orange.svg)](https://ai.google.dev/)

</div>

---

## 📖 نظرة عامة

**المشكلة:** النماذج اللغوية الكبيرة (LLMs) ما عندها ذاكرة دائمة — كل محادثة تبدأ من الصفر بدون سياق سابق.

**الحل:** نظام ذاكرة تكيفية يحفظ كل محادثة كـ Embedding دلالي، ويسترجع الذكريات المرتبطة تلقائيًا قبل توليد كل رد — مع تضاؤل زمني تدريجي يضمن أولوية المعلومات الحديثة.

---

## 🏗️ البنية المعمارية

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   User      │────▶│  Context         │────▶│  Gemini      │
│  Prompt     │     │  Enrichment      │     │  Model       │
└─────────────┘     │                  │     │              │
                    │  ┌────────────┐  │     └──────────────┘
                    │  │ Adaptive   │◀─┼────────────┼── Save
                    │  │ Memory     │  │            │  Response
                    │  │ Engine     │  │            │
                    │  └────────────┘  │            │
                    │        │         │            │
                    │    ┌──────────┐  │            │
                    └────▶│ SQLite   │◀─┘            │
                         │ Database │               │
                         └──────────┘               │
                                                    │
                         ┌──────────────────────────┘
                         ▼
                    ┌──────────────┐
                    │ Embedding    │
                    │ Model        │
                    └──────────────┘
```

---

## ⚙️ المكونات الرئيسية

| المكون | الوظيفة | الحالة |
|--------|---------|--------|
| `main.py` | الملف الرئيسي — إدارة الطلبات والاستجابة عبر Google Gen AI SDK | ✅ جاهز |
| `adaptive_memory_engine.py` | محرك الذاكرة التكيفية (تشابه + تضاؤل زمني + SQLite) | ✅ جاهز |
| `rag_engine.py` | محرك استرجاع المعلومات المعزز | 🔧 قيد التطوير |
| `ingest.py` | تغذية البيانات الأولية للنظام | 🔧 قيد التطوير |
| `mowjh_engine.py` | المحرك الأساسي للمنصة | 🔧 قيد التطوير |

---

## 🧠 كيف يشتغل؟

```
📥 الاستقبال → 🔄 التضمين → 🔍 الاسترجاع → ⏳ التضاؤل الزمني → 📝 إثراء السياق → ✨ التوليد → 💾 الحفظ
```

### 1. الاستقبال 📥
يصل سؤال المستخدم → يُحوَّل إلى متجه دلالي (Embedding)

### 2. الاسترجاع 🔍
محرك الذاكرة يبحث في قاعدة البيانات عن الذكريات ذات التشابه الأعلى مع السؤال الحالي

### 3. التضاؤل الزمني ⏳
كل ذكرى تُضرب بمعامل تضاؤل (`e^(-λt)`) — كلما مر الوقت، قل وزنها تدريجياً

### 4. إثراء السياق 📝
أفضل النتائج تُضاف فوق السؤال الأصلي كـ Context → النموذج يفهم الصورة الكاملة

### 5. التوليد والحفظ ✨💾
النموذج يولد الرد → المحادثة كاملة تُحفظ كذكرى جديدة مع فحص التكرار (Deduplication)

---

## 🔧 المتطلبات

```bash
pip install numpy google-genai
```

أو تثبيت الكل دفعة واحدة:
```bash
pip install -r requirements.txt
```

---

## 🚀 التشغيل

### عبر Google Colab:

```python
!pip install numpy google-genai

import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
os.environ["MODEL_NAME"] = "gemini-3.8-flash"
os.environ["EMBEDDING_MODEL"] = "gemini-embedding-2"
os.environ["MAX_WORKERS"] = "1"

%run main.py
```

### عبر الجهاز المحلي:

```bash
git clone https://github.com/mzydhaif11-ctrl/Suggested-Workflows.git
cd Suggested-Workflows
pip install -r requirements.txt
export GOOGLE_API_KEY="your_api_key_here"
python main.py
```

---

## ⚙️ الإعدادات (Environment Variables)

| المتغير | الوصف | القيمة الافتراضية |
|---------|-------|-------------------|
| `GOOGLE_API_KEY` | مفتاح Google AI Studio | **مطلوب** 🔑 |
| `MODEL_NAME` | نموذج التوليد | `gemini-3.8-flash` |
| `EMBEDDING_MODEL` | نموذج المتجهات | `gemini-embedding-2` |
| `MAX_TOKENS` | الحد الأقصى للرد | `1024` |
| `TEMPERATURE` | درجة الإبداع | `0.7` |
| `DECAY_RATE` | معدل التضاؤل الزمني | `0.01` |
| `SIMILARITY_THRESHOLD` | الحد الأدنى للتشابه | `0.5` |
| `TOP_K_MEMORIES` | عدد الذكريات المسترجعة | `3` |
| `MEMORY_DB_PATH` | مسار قاعدة البيانات | `mowjn_memory.db` |
| `MAX_WORKERS` | عدد الطلبات المتزامنة | `5` |

---

## 📊 مميزات محرك الذاكرة

| الميزة | الوصف |
|--------|-------|
| ✅ تخزين دائم | SQLite — لا ضياع عند إعادة التشغيل |
| ✅ اكتشاف التكرار | Deduplication — يمنع حفظ معلومات مكررة |
| ✅ دوال تضاؤل متعددة | أسية / خطية / Step |
| ✅ عمليات دفعات | Batch Operations — معالجة متوازية |
| ✅ تنظيف تلقائي | حذف الذكريات منخفضة الدرجة |
| ✅ إحصائيات فورية | معرفة حجم الذاكرة وحالتها |

---

## 📂 التقنيات المستخدمة

```
Python 3.10+  •  Google Gen AI SDK  •  SQLite  •  NumPy  •  ThreadPoolExecutor
```

---

## 👤 المؤلف

> **مزيد الرشيدي** — مطور مهتم بالذكاء الاصطناعي ومعالجة اللغات الطبيعية

---

## 📄 الترخيص

هذا المشروع مرخص بموجب [MIT License](LICENSE)

---

<div align="center">

**صُنع بـ ❤️ بواسطة مزيد الرشيدي**

</div>
