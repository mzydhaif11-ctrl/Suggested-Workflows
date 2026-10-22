from __future__ import annotations

import json
import logging
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ═══════ إعدادات ═══════

EMBEDDING_MODEL = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
GENERATIVE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
PERSIST_DIR = "/content/mowjn_farahidi_data"
DECAY_RATE = 0.01
SIMILARITY_THRESHOLD = 0.5
TOP_K_MEMORIES = 3
DB_FILE = "arabic_lexicon.db"


_engine_state = {
    "embedder": None,
    "tokenizer": None,
    "model": None,
    "chroma_client": None,
    "collection": None,
}


def _ensure_persist_dir():
    path = Path(PERSIST_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_embedder():
    if _engine_state["embedder"] is not None:
        return _engine_state["embedder"]
    try:
        from sentence_transformers import SentenceTransformer
        print(f"🔄 جاري تحميل نموذج التضمين...")
        _engine_state["embedder"] = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ تم تحميل نموذج التضمين")
    except ImportError:
        raise RuntimeError("pip install sentence-transformers")
    return _engine_state["embedder"]


def _load_generative_model():
    if _engine_state["model"] is not None:
        return _engine_state["tokenizer"], _engine_state["model"]
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"🔄 جاري تحميل {GENERATIVE_MODEL}... (قد يستغرق دقائق)")
        _engine_state["tokenizer"] = AutoTokenizer.from_pretrained(GENERATIVE_MODEL)
        _engine_state["model"] = AutoModelForCausalLM.from_pretrained(GENERATIVE_MODEL)
        print("✅ تم تحميل النموذج بنجاح")
    except ImportError:
        raise RuntimeError("pip install transformers torch accelerate sentencepiece")
    return _engine_state["tokenizer"], _engine_state["model"]


def _get_chroma_collection():
    if _engine_state["collection"] is not None:
        return _engine_state["collection"]
    import chromadb
    from chromadb.config import Settings
    persist_path = _ensure_persist_dir()
    _engine_state["chroma_client"] = chromadb.PersistentClient(
        path=str(persist_path),
        settings=Settings(anonymized_telemetry=False),
    )
    _engine_state["collection"] = _engine_state["chroma_client"].get_or_create_collection(
        name="farahidi_context",
        metadata={"hnsw:space": "cosine"},
    )
    print("✅ تم تهيئة الذاكرة السياقية (ChromaDB)")
    return _engine_state["collection"]


# ═══════ Memory & Context ═══════

def _embed(text: str) -> list:
    vector = _load_embedder().encode(text, show_progress_bar=False)
    return vector.tolist()


def _time_decay(timestamp_str: str) -> float:
    ts = datetime.fromisoformat(timestamp_str)
    now = datetime.now(timezone.utc)
    days_old = max(0, (now - ts).total_seconds() / 86400)
    return float(max(0, 1 - DECAY_RATE * days_old))


def _add_memory(word: str, content: str) -> bool:
    collection = _get_chroma_collection()
    embedding = _embed(content)
    existing = collection.query(query_embeddings=[embedding], n_results=1, include=["distances"])
    if existing["distances"][0][0] < 0.05:
        print(f"🔁 تكرار: '{word}' — تم التخطّي")
        return False
    memory_id = f"WORD_{hashlib.md5(word.encode()).hexdigest()[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()
    collection.upsert(
        ids=[memory_id], embeddings=[embedding], documents=[content],
        metadatas=[{"word": word, "timestamp": timestamp}],
    )
    print(f"➕ تم حفظ: '{word}'")
    return True


def _retrieve_context(query: str, top_k: int = TOP_K_MEMORIES) -> list:
    collection = _get_chroma_collection()
    total = collection.count()
    if total == 0:
        return []
    query_emb = _embed(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k * 3, total),
        include=["documents", "metadatas", "distances"],
    )
    context = []
    for i in range(len(results["ids"][0])):
        semantic_score = round(1.0 - results["distances"][0][i], 4)
        if semantic_score < SIMILARITY_THRESHOLD:
            continue
        meta = results["metadatas"][0][i]
        time_factor = _time_decay(meta["timestamp"])
        final_score = semantic_score * time_factor
        context.append({
            "word": meta.get("word", ""),
            "content": results["documents"][0][i],
            "semantic_score": semantic_score,
            "time_factor": round(time_factor, 4),
            "final_score": round(final_score, 4),
        })
    context.sort(key=lambda x: x["final_score"], reverse=True)
    return context[:top_k]


# ═══════ SQLite Fallback ═══════

def _sqlite_lookup(word: str) -> Optional[dict]:
    import sqlite3
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM dictionary WHERE word = ?", (word,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "word": row[0], "root": row[1], "definition": row[2],
                "synonyms": json.loads(row[3]) if row[3] else [],
                "antonyms": json.loads(row[4]) if row[4] else [],
                "register": row[5], "source": "محلي (موثق)",
            }
    except Exception:
        pass
    return None


def _sqlite_save(word: str, data: dict):
    import sqlite3
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dictionary (
                word TEXT PRIMARY KEY, root TEXT, definition TEXT,
                synonyms TEXT, antonyms TEXT, register TEXT, source TEXT
            )
        """)
        cursor.execute(
            "INSERT OR REPLACE INTO dictionary VALUES (?, ?, ?, ?, ?, ?, ?)",
            (word, data.get("root", ""), data.get("definition", ""),
             json.dumps(data.get("synonyms", []), ensure_ascii=False),
             json.dumps(data.get("antonyms", []), ensure_ascii=False),
             data.get("register", "عام"), "محلي AI"),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ═══════ Local Generation ═══════

def _generate_with_local_model(prompt: str) -> dict:
    tokenizer, model = _load_generative_model()
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.2, top_p=0.9)
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip()
        if "```json" in generated:
            generated = generated.split("```json")[1].split("```")[0].strip()
        elif "```" in generated:
            generated = generated.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(generated)
        except json.JSONDecodeError:
            return {"raw_response": generated}
    except Exception as e:
        return {"error": str(e)}


# ═══════ Public API ═══════

def init_farahidi(api_key=None):
    """تهيئة المحرك — api_key لم تعد مطلوبة."""
    if api_key:
        logger.warning("api_key deprecated — engine runs locally")
    _ensure_persist_dir()
    import sqlite3
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dictionary (
            word TEXT PRIMARY KEY, root TEXT, definition TEXT,
            synonyms TEXT, antonyms TEXT, register TEXT, source TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"\n🌊 Farahidi × Mowjn v0.4.0 | Model: {GENERATIVE_MODEL}")
    print("=" * 50)


def lookup(word: str) -> dict:
    """بحث عن كلمة عربية مع إثراء سياقي."""
    word = word.strip()

    # Semantic search in ChromaDB
    memories = _retrieve_context(word)
    if memories:
        best = memories[0]
        print(f"⚡ Semantic Cache Hit: '{word}' (تشابه: {best['semantic_score']})")
        definition = best["content"]
        try:
            data = json.loads(definition)
        except json.JSONDecodeError:
            data = {"definition": definition}
        return {"word": word, "source": "memory_cache", "similarity": best["semantic_score"], "context_used": len(memories), **data}

    # SQLite fallback
    sqlite_result = _sqlite_lookup(word)
    if sqlite_result:
        print(f"⚡ SQLite Cache Hit: '{word}'")
        return sqlite_result

    # Cache miss — generate with local model
    print(f"🔄 Semantic Cache Miss: '{word}' — جاري التحليل...")
    prompt_parts = [
        'أنت معجم عربي موثق متخصص. حلل الكلمة التالية ورد بصيغة JSON فقط تحتوي على:',
        '- definition: التعريف المفصل',
        '- root: الجذر الثلاثي',
        '- synonyms: قائمة مرادفات',
        '- antonyms: قائمة أضداد',
        '- register: المستوى اللغوي (فصيح/عامي/مختلط)',
        '',
    ]
    related = _retrieve_context(word, top_k=3)
    if related:
        context_text = "\n".join([f"- {m['content']}" for m in related])
        prompt_parts.append(f"سياق مرتبط:\n{context_text}\n")
    prompt_parts.append(f'\nحلل كلمة "{word}":')
    full_prompt = "\n".join(prompt_parts)

    data = _generate_with_local_model(full_prompt)
    if "error" not in data:
        content_to_save = json.dumps(data, ensure_ascii=False)
        _add_memory(word, content_to_save)
        _sqlite_save(word, data)
        data["word"] = word
        data["source"] = "محلي AI"
        data["context_used"] = len(related)
    return data


def enhance_text(text: str) -> dict:
    """تحسين نص عربي بالنموذج المحلي."""
    prompt = f'حسّن النص العربي التالي مع الحفاظ على المعنى. رد بصيغة JSON تحوي: original, enhanced, changes (قائمة), explanation\n\nالنص: "{text}"\n\nرد:'
    return _generate_with_local_model(prompt)


def ask_conversationally(message: str) -> dict:
    """واجهة محادثة طبيعية."""
    memories = _retrieve_context(message, top_k=5)
    prompt_parts = ["أنت مساعد لغوي عربي ذكي ومتخصص. أجب بدقة ووضوح."]
    if memories:
        context_text = "\n".join([f"- {m['content']}" for m in memories[:3]])
        prompt_parts.append(f"\nسياق سابق:\n{context_text}")
    prompt_parts.append(f"\n\nالسؤال: {message}")
    full_prompt = "\n".join(prompt_parts)

    tokenizer, model = _load_generative_model()
    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=800, do_sample=True, temperature=0.7, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(full_prompt):].strip()
    _add_memory(message, f"سؤال: {message}\nرد: {answer[:300]}")
    return {"query": message, "response": answer, "context_memories": len(memories)}


# ═══════ Demo ═══════

print("\n🌊 Farahidi × Mowjn Hybrid Engine v0.4.0")
print("=" * 50)
print("النموذج: Qwen2.5-Coder-7B | الذاكرة: موجة البيان")
print("=" * 50)

# تجربة ١: كلمة جديدة
print("\n📌 تجربة ١: كلمة جديدة (Cache Miss)")
result = lookup("اليعسوب")
print(f"   المصدر: {result.get('source')}")
print(f"   التعريف: {result.get('definition', 'غير متوفر')[:200]}")

# تجربة ٢: نفس الكلمة (Cache Hit)
print("\n📌 تجربة ٢: نفس الكلمة (Semantic Cache Hit)")
result = lookup("اليعسوب")
print(f"   المصدر: {result.get('source')}")
print(f"   التشابه: {result.get('similarity')}")

# تجربة ٣: صياغة مختلفة (بحث دلالي)
print("\n📌 تجربة ٣: صياغة مختلفة (بحث دلالي)")
result = lookup("حشرة طائرة كبيرة لها جناحين")
print(f"   المصدر: {result.get('source')}")
print(f"   التشابه: {result.get('similarity')}")

# تجربة ٤: سؤال مفتوح
print("\n📌 تجربة ٤: سؤال مفتوح")
result = ask_conversationally("وش الفرق بين اليعسوب والذباب؟")
print(f"   السياق المستخدم: {result['context_memories']} ذكرى")
print(f"   الرد: {result['response'][:400]}")

print("\n🎉 انتهت التجربة!")
