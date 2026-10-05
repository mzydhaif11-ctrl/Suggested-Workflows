def get_embedding(text: str) -> list:
    """توليد متجهات النص باستخدام نموذج Gemini text-embedding-004"""
    if not GOOGLE_API_KEY:
        return []
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]
    except Exception as e:
        print(f"❌ خطأ في توليد التضمين: {e}")
        return []

def retrieve_context(query: str, top_k: int = 3) -> list:
    """استرجاع السياق الموحد من Qdrant + ChromaDB"""
    contexts = []
    
    # توليد المتجهات مرة واحدة للبحث في Qdrant
    query_vector = get_embedding(query)

    # 1. استرجاع من Qdrant عبر المتجهات
    qdrant = get_qdrant_client()
    if qdrant and query_vector:
        try:
            results = qdrant.search(
                collection_name="documents",
                query_vector=query_vector,  # تمرير المتجه المحسوب
                limit=top_k
            )
            for hit in results:
                if hit.payload and "content" in hit.payload:
                    contexts.append(hit.payload["content"])
        except Exception as e:
            print(f"⚠️ فشل الاسترجاع من Qdrant: {e}")

    # 2. استرجاع من ChromaDB (تتكفل بدوال التضمين تلقائياً عبر gemini_ef)
    collection = get_chroma_collection()
    if collection:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            if results and results.get("documents") and results["documents"][0]:
                for doc in results["documents"][0]:
                    contexts.append(doc)
        except Exception as e:
            print(f"⚠️ فشل الاسترجاع من ChromaDB: {e}")

    return contexts
