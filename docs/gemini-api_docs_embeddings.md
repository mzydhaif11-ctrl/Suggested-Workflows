# Embeddings in Gemini API

Official guide for generating vector embeddings using Google models.
Source: https://ai.google.dev/gemini-api/docs/embeddings

---

## 1. Overview
Vector embeddings represent text as dense numerical vectors in a high-dimensional space. Words or sentences with similar semantic meanings are placed closer together, enabling vector search, retrieval-augmented generation (RAG), and text classification.

---

## 2. Core Model
- **`text-embedding-004`**: Optimized for semantic search, retrieval tasks, and clustering with high dimensional accuracy (768 dimensions).

---

## 3. Supported Task Types
Specifying the `task_type` tunes the embeddings for optimal downstream performance:
- `RETRIEVAL_QUERY`: Specifies the text is a search query.
- `RETRIEVAL_DOCUMENT`: Specifies the text is a target document to be stored in vector databases like Qdrant.
- `SEMANTIC_SIMILARITY`: Optimizes for measuring cosine similarity between text snippets.
- `CLASSIFICATION`: Optimizes for downstream machine learning classifiers.

---

## 4. Code Example (Python)
```python
from google import genai
from google.genai import types

client = genai.Client()

# Generate an embedding for a document
result = client.models.embed_content(
    model="text-embedding-004",
    contents="Qdrant is an open-source vector search engine.",
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        title="Qdrant Overview",
    ),
)

# Extract embedding values
embedding_vector = result.embedding.values
print(f"Embedding dimension: {len(embedding_vector)}")
