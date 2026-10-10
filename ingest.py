import os
import glob
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import chromadb
from google import genai

# ──────────────────────────────────────────────
# Environment Settings
# ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

CHROMA_URL = os.getenv("CHROMA_URL")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")

# Initialize Gen AI client
_genai_client = None
if GOOGLE_API_KEY:
    _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY or GEMINI_API_KEY not found")

# Chunk size in characters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks."""
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
    """Generate 768-dim embedding using Gemini gemini-embedding-2."""
    try:
        result = _genai_client.models.embed_content(
            model="gemini-embedding-2",
            contents=text,
            task_type="retrieval_document"
        )
        return result[0].values[0]
    except Exception as e:
        print(f"Error creating embedding: {e}")
        raise


def get_qdrant_client():
    """Connect to Qdrant Cloud or locally."""
    try:
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(url=QDRANT_URL)
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return None


def get_chroma_client():
    """Connect to ChromaDB cloud or local."""
    try:
        if CHROMA_URL and CHROMA_API_KEY:
            connect_kwargs = {
                "host": CHROMA_URL,
                "headers": {"Authorization": f"Bearer {chroma_api_key}"}
            }
            if CHROMA_TENANT:
                connect_kwargs["tenant"] = CHROMA_TENANT
                connect_kwargs["database"] = "default_database"
            return chromadb.HttpClient(**connect_kwargs)
        return chromadb.Client()
    except Exception as e:
        print(f"Failed to connect to ChromaDB: {e}")
        return None


# ──────────────────────────────────────────────
# Main Ingestion Function
# ──────────────────────────────────────────────
def ingest_documents(docs_dir: str = "./docs"):
    """Import documents and generate embeddings into Qdrant and ChromaDB."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        alt_path = Path("./pages")
        if alt_path.exists():
            docs_path = alt_path
        else:
            print(f"Directory {docs_dir} not found - indexing skipped")
            return

    md_files = list(docs_path.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {docs_path}")
        return

    print(f"Importing {len(md_files)} files from: {docs_path}...")

    qdrant = get_qdrant_client()
    chroma = get_chroma_client()

    collection_name = "documents"

    # Initialize Qdrant collection with 768 dimensions matching gemini-embedding-2
    if qdrant:
        try:
            collections = qdrant.get_collections().collections
            if not any(c.name == collection_name for c in collections):
                qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                print("Qdrant collection created successfully (dimensions: 768).")
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")

    # Initialize ChromaDB collection
    chroma_collection = None
    if chroma:
        try:
            chroma_collection = chroma.get_or_create_collection(name=collection_name)
            print("ChromaDB collection ready.")
        except Exception as e:
            print(f"Error initializing ChromaDB collection: {e}")

    point_id = 0
    for md_file in md_files:
        try:
            text = md_file.read_text(encoding="utf-8")
            filename = md_file.name

            chunks = chunk_text(text)
            print(f"  Processing {filename}: split into {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                try:
                    embedding = get_embedding(chunk)

                    # Upload to Qdrant
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

                    # Upload to ChromaDB
                    if chroma_collection:
                        chroma_collection.add(
                            ids=[f"{filename}_{i}"],
                            documents=[chunk],
                            metadatas=[{"source": filename, "chunk_index": i}],
                            embeddings=[embedding]
                        )

                    point_id += 1

                except Exception as e:
                    print(f"  Error processing chunk {i} from {filename}: {e}")
                    continue

        except Exception as e:
            print(f"  Failed to read file {md_file.name}: {e}")
            continue

    print(f"Done! Embedded and uploaded {point_id} chunks to databases.")


# ──────────────────────────────────────────────
# Direct Execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    docs_directory = os.getenv("DOCS_DIR", "./docs")
    ingest_documents(docs_directory)
