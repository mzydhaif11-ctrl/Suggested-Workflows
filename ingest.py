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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_ADMIN_TOKEN = os.getenv("CHROMA_ADMIN_TOKEN")

genai.configure(api_key=GOOGLE_API_KEY)

# حجم القطعة (chunk size) بالرموز
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ──────────────────────────────────────────────
# دوال مساعدة
# ──────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """تقسيم النص إلى قطع صغيرة"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

def get_embedding(text: str) -> list:
    """إنشاء متجه باستخدام Gemini Embedding"""
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        print(f"❌ خطأ في إنشاء المتجه: {e}")
        raise

def get_qdrant_client():
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)

def get_chroma_client():
    if CHROMA_API_KEY:
        client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            headers={
                "X-Chroma-Token": CHROMA_API_KEY,
                "Authorization": f"Bearer {CHROMA_ADMIN_TOKEN}"
            }
        )
    else:
        client = chromadb.Client()
    return client

# ──────────────────────────────────────────────
# الدالة الرئيسية
# ──────────────────────────────────────────────
def ingest_documents(docs_dir: str = "./docs"):
    """استيراد المستندات إلى Qdrant و ChromaDB"""
    
    # التحقق من وجود المجلد
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"⚠️ مجلد {docs_dir} غير موجود — تخطي الاستيراد")
        return
    
    # قراءة ملفات Markdown
    md_files = list(docs_path.glob("*.md"))
    if not md_files:
        print("⚠️ لا توجد ملفات .md للاستيراد")
        return
    
    print(f"📂 جاري استيراد {len(md_files)} ملف...")
    
    # الاتصال بقواعد البيانات
    qdrant = get_qdrant_client()
    chroma = get_chroma_client()
    
    # إنشاء المجموعات
    collection_name = "documents"
    
    # Qdrant
    try:
        collections = qdrant.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            # استخدام متجه افتراضي لحين أول عملية embedding
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            print("✅ تم إنشاء مجموعة Qdrant")
    except Exception as e:
        print(f"⚠️ تحذير Qdrant: {e}")
    
    # ChromaDB
    try:
        chroma_collection = chroma.get_or_create_collection(collection_name)
        print("✅ تم إنشاء مجموعة ChromaDB")
    except Exception as e:
        print(f"⚠️ تحذير ChromaDB: {e}")
        return
    
    # معالجة كل ملف
    point_id = 0
    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8")
            filename = md_file.name
            
            # تقسيم النص
            chunks = chunk_text(text)
            print(f"  📄 {filename}: {len(chunks)} قطعة")
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                try:
                    embedding = get_embedding(chunk)
                    
                    # إضافة إلى Qdrant
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
                    
                    # إضافة إلى ChromaDB
                    chroma_collection.add(
                        ids=[f"{filename}_{i}"],
                        documents=[chunk],
                        metadatas=[{"source": filename, "chunk_index": i}],
                        embeddings=[embedding]
                    )
                    
                    point_id += 1
                    
                except Exception as e:
                    print(f"  ⚠️ خطأ في قطعة {i} من {filename}: {e}")
                    continue
        
        except Exception as e:
            print(f"  ❌ خطأ في قراءة {md_file.name}: {e}")
            continue
    
    print(f"✅ تم استيراد {point_id} قطعة بنجاح!")

# ──────────────────────────────────────────────
# تشغيل مباشر
# ──────────────────────────────────────────────
if __name__ == "__main__":
    docs_directory = os.getenv("DOCS_DIR", "./docs")
    ingest_documents(docs_directory)
