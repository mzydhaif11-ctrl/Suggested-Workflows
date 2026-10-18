
# 🌊 موجة البيان | Mowjn AI-Bayan

### نظام ذاكرة سياقية تكيفية ومحرك استرجاع معلومات محلي بالكامل

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)
[![Local First](https://img.shields.io/badge/Privacy-Local%20First-purple.svg)](#)
[![Arabic NLP](https://img.shields.io/badge/NLP-Arabic%20Support-orange.svg)](#)
[![Made in KSA](https://img.shields.io/badge/Made%20in-KSA%20🇸🇦-006C35?style=for-the-badge)](#)

---

## 🎯 الرؤية

> **نُريد أن تكون التقنية العربية في خدمة الوطن.**

موجة البيان مشروع يهدف لبناء بنية تحتية ذكية محلية بالكامل — لا تعتمد على سيرفرات خارجية، لا تُرسل بيانات المواطنين لأي جهة أجنبية، وتخدم اللغة العربية بأعلى جودة ممكنة.

هذا المشروع هو خطوة نحو **السيادة الرقمية** — لأن البيانات الوطنية يجب أن تبقى وطنية. 🇸🇦

---

## 📖 نظرة عامة

**المشكلة:** النماذج اللغوية الكبيرة (LLMs) ما عندها ذاكرة دائمة — كل محادثة تبدأ من الصفر بدون سياق سابق. والبدائل المتاحة تعتمد على خدمات سحابية أجنبية تنقل البيانات خارج الحدود.

**الحل:** نظام متكامل يجمع بين:
- **ذاكرة سياقية تكيفية** تحفظ المحادثات كمتجهات دلالية مع تضاؤل زمني تدريجي
- **محرك استرجاع معلومات (RAG)** يبحث دلالياً في قاعدة المعرفة الثابتة
- **كل شيء محلي** — لا يرسل أي بيانات لسيرفرات خارجية 🔒

---

## 🏗️ البنية المعمارية

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   User      │────▶│  Context         │────▶│  Response        │
│  Prompt     │     │  Enrichment      │     │  Generation      │
└─────────────┘     │                  │     └──────────────────┘
                    │  ┌────────────┐  │
                    │  │ Adaptive   │◀─┼────────────┼── Save
                    │  │ Memory     │  │            │  Conversation
                    │  │ Engine     │  │            │
                    │  └────────────┘  │            │
                    │       │          │            │
                    │  ┌──────────┐    │            │
                    └──▶│ ChromaDB │◀──┘            │
                        │ (Persistent)              │
                        └──────────┘                │
                                                    │
                         ┌──────────────────────────┘
                         ▼
                    ┌──────────────────┐
                    │ Grounded RAG     │
                    │ Engine           │
                    │                  │
                    │ ┌────────────┐   │
                    │ │ Vector     │   │
                    │ │ Store      │   │
                    │ │ (docs/)    │   │
                    │ └────────────┘   │
                    └──────────────────┘
                         │
                         ▼
                    ┌──────────────────┐
                    │ Sentence-        │
                    │ Transformers     │
                    │ (Local Embedding)│
                    └──────────────────┘
```

---

## ⚙️ المكونات الرئيسية

| المكون | الوظيفة | التقنية | الحالة |
|--------|---------|---------|--------|
| `adaptive_memory_engine.py` | محرك الذاكرة التكيفية المتقدم | ChromaDB + sentence-transformers | ✅ جاهز |
| `adaptive_memory.py` | نسخة تعليمية مبسطة | NumPy + sentence-transformers | ✅ جاهز |
| `vector_store.py` | مخزن متجهات لقاعدة المعرفة | ChromaDB + sentence-transformers | ✅ جاهز |
| `rag_engine.py` | محرك استرجاع المعلومات المعزز | Vector Search + BM25 Keyword | ✅ جاهز |
| `ingest.py` | تغذية وفهرسة الوثائق | vector_store integration | ✅ جاهز |
| `main.py` | الملف الرئيسي — إدارة الطلبات | AdaptiveMemoryEngine | ✅ جاهز |

---

## 🧠 كيف يشتغل؟

```
📥 الاستقبال → 🔄 التضمين → 🔍 الاسترجاع → ⏳ التضاؤل الزمني → 📝 إثراء السياق → ✨ التوليد → 💾 الحفظ
```

### 1. الاستقبال 📥
يصل سؤال المستخدم → يُحوَّل إلى متجه دلالي (Embedding) محلياُ

### 2. الاسترجاع 🔍
محرك الذاكرة يبحث في قاعدة البيانات عن الذكريات ذات التشابه الأعلى مع السؤال الحالي

### 3. التضاؤل الزمني ⏳
كل ذكرى تُضرب بمعامل تضاؤل (`e^(-λt)`) — كلما مر الوقت، قل وزنها تدريجياً

### 4. إثراء السياق 📝
أفضل النتائج تُضاف فوق السؤال الأصلي كـ Context → النموذج يفهم الصورة الكاملة

### 5. التوليد والحفظ ✨💾
النموذج يولد الرد → المحادثة كاملة تُحفظ كذكرى جديدة مع فحص التكرار (Deduplication)

---

## 🔒 الخصوصية والأمان — السيادة الرقمية

هذا النظام مصمم ليكون **محلي بالكامل**:

| الميزة | التفاصيل |
|--------|----------|
| ❌ لا إرسال بيانات | لا يوجد اتصال بـ Google أو أي خدمة خارجية |
| ❌ لا يحتاج API Key | يعمل بدون مفاتيح API |
| ✅ تخزين محلي | ChromaDB يحفظ المتجهات على القرص المحلي |
| ✅ نموذج محلي | sentence-transformers يتعالج بالكامل على الجهاز |
| ✅ تحكم كامل | أنت تتحكم في كل البيانات والوصول إليها |
| ✅ سيادة رقمية | البيانات الوطنية تبقى داخل الحدود الوطنية 🇸🇦 |

---

## 🔧 المتطلبات

```bash
pip install numpy sentence-transformers chromadb
```

أو تثبيت الكل دفعة واحدة:
```bash
pip install -r requirements.txt
```

### المتطلبات الإضافية:
- Python 3.10+
- ~500MB مساحة للتحميل الأولي لنموذج التضمين
- مجلد `docs/` يحتوي ملفات `.md` أو `.txt` لقاعدة المعرفة

---

## 🚀 التشغيل

### عبر Google Colab:

```python
!pip install numpy sentence-transformers chromadb -q

import os
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

%run main.py
```

### عبر الجهاز المحلي:

```bash
git clone https://github.com/mzydhaif11-ctrl/Suggested-Workflows.git
cd Suggested-Workflows
pip install -r requirements.txt
python main.py
```

### تشغيل فهرسة الوثائق:

```bash
# فهرسة عادية (يبني لو ما موجود)
python ingest.py

# إعادة بناء كاملة
python ingest.py --rebuild
```

### تجربة محرك RAG:

```bash
python rag_engine.py
# اكتب أسئلة بالعربي واختبر النتائج
```

---

## ⚙️ الإعدادات (Environment Variables)

| المتغير | الوصف | القيمة الافتراضية |
|---------|-------|-------------------|
| `EMBEDDING_MODEL` | نموذج التضمين الدلالي | `sentence-transformers/paraphrase-xlm-r-multilingual-v1` |
| `DECAY_RATE` | معدل التضاؤل الزمني | `0.01` |
| `SIMILARITY_THRESHOLD` | الحد الأدنى للتشابه | `0.5` |
| `TOP_K_MEMORIES` | عدد الذكريات المسترجعة | `3` |
| `CHROMA_COLLECTION` | اسم مجموعة ChromaDB | `mowjn_knowledge` |
| `MAWJAH_DOCS_DIR` | مسار مجلد الوثائق | `docs/` |

---

## 📊 مميزات محرك الذاكرة

| الميزة | الوصف |
|--------|-------|
| ✅ تخزين دائم | ChromaDB — لا ضياع عند إعادة التشغيل |
| ✅ اكتشاف التكرار | Deduplication — يمنع حفظ معلومات مكررة |
| ✅ دوال تضاؤل متعددة | أسية / خطية / Step |
| ✅ عمليات دفعات | Batch Operations — معالجة متوازية |
| ✅ تنظيف تلقائي | حذف الذكريات منخفضة الدرجة |
| ✅ إحصائيات فورية | معرفة حجم الذاكرة وحالتها |
| ✅ دعم العربية | نموذج متعدد اللغات يدعم العربي بقوة |
| ✅ Thread-Safe | آمن للكتابة المتزامنة |

---

## 📂 التقنيات المستخدمة

```
Python 3.10+  •  sentence-transformers  •  ChromaDB  •  NumPy  •  HNSW Index
```

---

## 📁 هيكل المشروع

```
Suggested-Workflows/
├── adaptive_memory.py          # نسخة تعليمية مبسطة
├── adaptive_memory_engine.py   # محرك الذاكرة المتقدم
├── vector_store.py             # مخزن المتجهات الدلالية
├── rag_engine.py               # محرك RAG المحلي
├── ingest.py                   # فهرسة الوثائق
├── main.py                     # الملف الرئيسي
├── docs/                       # قاعدة المعرفة الثابتة
├── .chroma_knowledge/          # قاعدة بيانات ChromaDB (محفوظ محلياً)
├── requirements.txt            # المتطلبات
├── .gitignore                  # ملفات مستبعدة
└── README.md                   # هذا الملف
```

---

## 👤 المؤلف

> **مزيد الرشيدي** — مطور مهتم بالذكاء الاصطناعي ومعالجة اللغات الطبيعية
>
> *"نبني لأننا نؤمن بأن التقنية العربية يجب أن تخدم العرب أولاً."*

---

## 📄 الترخيص

هذا المشروع مرخص بموجب [MIT License](LICENSE)

---

**صُنع بـ ❤️ في المملكة العربية السعودية 🇸🇦**

*"من أجل وطن أذكى."*
