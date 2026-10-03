import os
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# قراءة المتغيرات البيئية
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "gemini_docs"

# إعداد Gemini
genai.configure(api_key=GEMINI_API_KEY)

# إعداد Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

DOCUMENTS = [
    {
        "id": 1,
        "text": "Gemini Live API توفر قدرات تفاعلية متقدمة مثل المحادثات الصوتية بالوقت الفعلي والتجاوب السريع مع المستخدمين وتحليل البيانات.",
        "source": "توثيق Gemini Live"
    },
    {
        "id": 2,
        "text": "منصة موجة البيان (Mowjh Al-Bayan) هي منصة دعم تقني تعتمد على الذكاء الاصطناعي لتحليل الاستفسارات وتقديم حلول سريعة وموثوقة للعملاء.",
        "source": "مقدمة موجة البيان"
    },
    {
        "id": 3,
        "text": "تعتمد عملية الـ RAG على تخزين البيانات في Qdrant كمتجهات رقمية واسترجاعها للرد على استفسارات المستخدمين بدقة وسرعة.",
        "source": "دليل المطورين"
    }
]

def create_collection():
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(col.name == COLLECTION_NAME for col in collections)
        
        if not exists:
            print(f"إنشاء مجموعة جديدة باسم: {COLLECTION_NAME}...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print("تم إنشاء المجموعة بنجاح.")
        else:
            print(f"المجموعة '{COLLECTION_NAME}' موجودة مسبقاً.")
    except Exception as e:
        print(f"خطأ أثناء التحقق/إنشاء المجموعة: {e}")
        raise e

def generate_embedding(text: str):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return response['embedding']

def main():
    print("بدء عملية تهيئة وفهرسة البيانات...")
    create_collection()
    
    points = []
    for doc in DOCUMENTS:
        print(f"جاري تحويل المستند {doc['id']}...")
        try:
            vector = generate_embedding(doc["text"])
            point = PointStruct(
                id=doc["id"],
                vector=vector,
                payload={
                    "text": doc["text"],
                    "source": doc["source"]
                }
            )
            points.append(point)
        except Exception as e:
            print(f"خطأ في المستند {doc['id']}: {e}")
    
    if points:
        print("رفع المتجهات...")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print("✅ تمت الفهرسة بنجاح!")

if __name__ == "__main__":
    main()
