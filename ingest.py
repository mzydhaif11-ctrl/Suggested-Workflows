import os
import glob
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb
import google.generativeai as genai

# ──────────────────────────────────────────────
# إعدادات البيئة
# ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHROMA_URL = os.getenv("CHROMA_URL")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("⚠️ تحذير: لم يتم العثور على GOOGLE_API_KEY أو GEMINI_API_KEY")

# حجم القطعة بالرموز
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ──────────────────────────────────────────────
# دوال مساعدة
# ──────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """تقسيم النص إلى قطع متداخلة"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str) -> list:
    """إنشاء متجه 768 بعد باستخدام Gemini text-embedding-004"""
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        print(f"❌ خطأ في إنشاء المتجه: {e}")
        raise

def get_qdrant_client():
    """الاتصال بـ Qdrant Cloud أو محلياً"""
    try:
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(url=QDRANT_URL)
    except Exception as e:
        print(f"❌ تعذر الاتصال بـ Qdrant: {e}")
        return None

def get_chroma_client():
    """الاتصال بـ ChromaDB السحابي أو المحلي"""
    try:
        if CHROMA_URL and CHROMA_API_KEY:
            connect_kwargs = {
                "host": CHROMA_URL,
                "headers": {"Authorization": f"Bearer {CHROMA_API_KEY}"}
            }
            if CHROMA_TENANT:
                connect_kwargs["tenant"] = CHROMA_TENANT
                connect_kwargs["database"] = "default_database"

            return chromadb.HttpClient(**connect_kwargs)
        return chromadb.Client()
    except Exception as e:
        print(f"❌ تعذر الاتصال بـ ChromaDB: {e}")
        return None

# ──────────────────────────────────────────────
# الدالة الرئيسية
# ──────────────────────────────────────────────
def ingest_documents(docs_dir: str = "./docs"):
    """استيراد المستندات وتوليد المتجهات إلى Qdrant و ChromaDB"""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        # فحص بديل لمجلد pages إن وجد
        alt_path = Path("./pages")
        if alt_path.exists():
            docs_path = alt_path
        else:
            print(f"⚠️ المجلد {docs_dir} غير موجود — تم إلغاء الفهرسة")
            return
    
    md_files = list(docs_path.glob("*.md"))
    if not md_files:
        print(f"⚠️ لا توجد ملفات .md داخل {docs_path}")
        return
    
    print(f"📂 جاري استيراد {len(md_files)} ملف من المسار: {docs_path}...")
    
    qdrant = get_qdrant_client()
    chroma = get_chroma_client()
    
    collection_name = "documents"
    
    # تهيئة مجموعة Qdrant بأبعاد 768 المطابقة لـ text-embedding-004
    if qdrant:
        try:
            collections = qdrant.get_collections().collections
            if not any(c.name == collection_name for c in collections):
                qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                print("✅ تم إنشاء مجموعة Qdrant بنجاح (الأبعاد: 768).")
        except Exception as e:
            print(f"⚠️ خطأ أثناء تهيئة مجموعة Qdrant: {e}")
    
    # تهيئة مجموعة ChromaDB
    chroma_collection = None
    if chroma:
        try:
            chroma_collection = chroma.get_or_create_collection(name=collection_name)
            print("✅ تم تجهيز مجموعة ChromaDB بنجاح.")
        except Exception as e:
            print(f"⚠️ خطأ أثناء تهيئة مجموعة ChromaDB: {e}")
    
    point_id = 0
    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8")
            filename = md_file.name
            
            chunks = chunk_text(text)
            print(f"  📄 معالجة {filename}: تم تقسيمه إلى {len(chunks)} مقطع")
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                try:
                    embedding = get_embedding(chunk)
                    
                    # الرفع إلى Qdrant
                    if qdrant:
                        qdrant.upsert(
                            collection_name=collection_name,
                            points=[PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "content": chunk,
                                    "source": filename,
                                    "chunk_index": i
                                }
                            )]
                        )
                    
                    # الرفع إلى ChromaDB
                    if chroma_collection:
                        chroma_collection.add(
                            ids=[f"{filename}_{i}"],
                            documents=[chunk],
                            metadatas=[{"source": filename, "chunk_index": i}],
                            embeddings=[embedding]
                        )
                    
                    point_id += 1
                    
                except Exception as e:
                    print(f"  ⚠️ خطأ في معالجة القطعة {i} من الملف {filename}: {e}")
                    continue
        
        except Exception as e:
            print(f"  ❌ تعذر قراءة الملف {md_file.name}: {e}")
            continue
    
    print(f"✅ تم الانتهاء بنجاح! تم تضمين ورفع {point_id} مقطع إلى قواعد البيانات.")

# ──────────────────────────────────────────────
# التشغيل المباشر
# ──────────────────────────────────────────────
if __name__ == "__main__":
    docs_directory = os.getenv("DOCS_DIR", "./docs")
    ingest_documents(docs_directory)
