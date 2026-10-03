import os
import glob
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# المتغيرات البيئية
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "gemini_docs"

# تشغيل العملاء
genai_client = genai.Client(api_key=GEMINI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def init_collection():
    collections = qdrant_client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    if not exists:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"تم إنشاء المجموعة '{COLLECTION_NAME}' بنجاح.")

def ingest_documents():
    init_collection()
    files = glob.glob("docs/*.md")
    points = []
    idx = 1

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # تقسيم الملف لنصوص بطول مناسب
        chunks = [content[i:i + 1500] for i in range(0, len(content), 1500)]

        for chunk_idx, chunk in enumerate(chunks):
            response = genai_client.models.embed_content(
                model="text-embedding-004",
                contents=chunk,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    title=f"{file_path} - Part {chunk_idx + 1}"
                ),
            )
            vector = response.embedding.values

            point = PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "source": file_path,
                    "chunk": chunk_idx,
                    "text": chunk
                }
            )
            points.append(point)
            idx += 1

    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"تم إدخال {len(points)} مقطعاً داخل Qdrant بنجاح.")

if __name__ == "__main__":
    ingest_documents()
