from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR = Path(
    os.getenv("MAWJAH_DOCS_DIR", str(DEFAULT_DOCS_DIR))
)

CHROMA_DIR = Path(
    os.getenv(
        "CHROMA_DB_PATH",
        str(BASE_DIR / "data" / "chroma_db"),
    )
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "mawjah_knowledge",
)

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

BASE_REPO_URL = (
    "https://github.com/mzydhaif11-ctrl/"
    "Suggested-Workflows/blob/main/docs"
)

INDEX_MARKER = "mawjah_docs_index"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)


def get_api_key() -> Optional[str]:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def get_google_client(
    api_key: Optional[str] = None,
) -> genai.Client:
    api_key = api_key or get_api_key()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY "
            "is missing from environment variables."
        )

    return genai.Client(api_key=api_key)


def get_collection():
    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "purpose": "Mawjah RAG knowledge",
        },
    )


def discover_files(
    docs_dir: Optional[Path] = None,
) -> List[Path]:
    folder = Path(docs_dir or DOCS_DIR)

    if not folder.is_dir():
        logger.warning(
            "Knowledge directory does not exist: %s",
            folder,
        )
        return []

    files = sorted(
        [
            *folder.glob("*.md"),
            *folder.glob("*.txt"),
        ],
        key=lambda path: path.name.lower(),
    )

    return [
        path
        for path in files
        if path.name.lower() != "index.md"
    ]


def _strip_front_matter(text: str) -> str:
    return re.sub(
        r"\A---\s*\n.*?\n---\s*\n",
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )


def _read_document(
    path: Path,
    docs_dir: Path,
) -> Dict[str, str]:
    content = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if not content.strip():
        return {}

    url_match = re.search(
        r"original_url\s*:\s*[\"']?(https?://[^\s\"'`]+)",
        content,
        flags=re.IGNORECASE,
    )

    source_url = (
        url_match.group(1).rstrip(".,)")
        if url_match
        else ""
    )

    heading_match = re.search(
        r"^#\s+(.+)$",
        content,
        flags=re.MULTILINE,
    )

    if heading_match:
        title = heading_match.group(1).strip().strip("# ")
    else:
        title = (
            path.stem
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

    try:
        relative_name = path.relative_to(
            docs_dir
        ).as_posix()
    except ValueError:
        relative_name = path.name

    if not source_url:
        source_url = (
            f"{BASE_REPO_URL}/{relative_name}"
        )

    body = _strip_front_matter(content).strip()

    if not body:
        return {}

    return {
        "file_name": relative_name,
        "title": title,
        "source_url": source_url,
        "content": body,
    }


def _prepare_corpus(
    docs_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    folder = Path(docs_dir or DOCS_DIR)

    files = discover_files(folder)

    fingerprint = hashlib.sha256()
    rows: List[Dict[str, Any]] = []

    for file_path in files:
        try:
            document = _read_document(
                file_path,
                folder,
            )

            if not document:
                continue

            fingerprint.update(
                document["file_name"].encode("utf-8")
            )

            fingerprint.update(
                document["content"].encode("utf-8")
            )

            chunks = text_splitter.split_text(
                document["content"]
            )

            for index, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue

                embedded_text = (
                    f"Document title: {document['title']}\n\n"
                    f"{chunk_text}"
                )

                chunk_hash = hashlib.sha256(
                    (
                        f"{document['file_name']}\0"
                        f"{index}\0"
                        f"{chunk_text}"
                    ).encode("utf-8")
                ).hexdigest()

                rows.append({
                    "id": chunk_hash,
                    "document": chunk_text,
                    "embedded_text": embedded_text,
                    "metadata": {
                        "source_title": document["title"],
                        "source_url": document["source_url"],
                        "file_name": document["file_name"],
                        "chunk_index": index,
                        "source_type": INDEX_MARKER,
                    },
                })

        except (OSError, UnicodeError, ValueError):
            logger.exception(
                "Failed to read knowledge file: %s",
                file_path,
            )

    return rows, fingerprint.hexdigest()


def _embed_texts(
    texts: List[str],
    task_type: str,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    if not texts:
        return []

    client = get_google_client(api_key)

    vectors: List[List[float]] = []
    batch_size = 32

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
            ),
        )

        response_embeddings = (
            getattr(response, "embeddings", None) or []
        )

        batch_vectors = [
            list(item.values or [])
            for item in response_embeddings
        ]

        if (
            len(batch_vectors) != len(batch)
            or any(not vector for vector in batch_vectors)
        ):
            raise RuntimeError(
                "Gemini returned an incomplete embeddings response."
            )

        vectors.extend([
            [float(value) for value in vector]
            for vector in batch_vectors
        ])

    return vectors


def rebuild_index(
    docs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    rows, fingerprint = _prepare_corpus(docs_dir)

    if not rows:
        raise ValueError(
            "No non-empty Markdown/TXT chunks found in "
            f"{Path(docs_dir or DOCS_DIR)}."
        )

    collection = get_collection()

    try:
        collection.delete(
            where={"source_type": INDEX_MARKER}
        )
    except Exception:
        logger.debug(
            "No existing Mawjah index entries to delete.",
            exc_info=True,
        )

    vectors = _embed_texts(
        [row["embedded_text"] for row in rows],
        task_type="RETRIEVAL_DOCUMENT",
    )

    batch_size = 100

    for start in range(0, len(rows), batch_size):
        end = start + batch_size

        batch_rows = rows[start:end]

        collection.upsert(
            ids=[
                row["id"]
                for row in batch_rows
            ],
            embeddings=vectors[start:end],
            documents=[
                row["document"]
                for row in batch_rows
            ],
            metadatas=[
                row["metadata"]
                for row in batch_rows
            ],
        )

    metadata = dict(collection.metadata or {})
    metadata["corpus_fingerprint"] = fingerprint

    collection.modify(
        metadata=metadata,
    )

    result = {
        "files": len(
            discover_files(
                Path(docs_dir or DOCS_DIR)
            )
        ),
        "chunks": len(rows),
        "collection": COLLECTION_NAME,
        "database_path": str(CHROMA_DIR),
    }

    logger.info(
        "Chroma index rebuilt: %s",
        result,
    )

    return result


def ensure_index(
    docs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    rows, fingerprint = _prepare_corpus(docs_dir)

    if not rows:
        return {
            "indexed": False,
            "reason": "no_documents",
            "chunks": 0,
        }

    collection = get_collection()

    stored_fingerprint = (
        collection.metadata or {}
    ).get("corpus_fingerprint")

    if (
        collection.count() > 0
        and stored_fingerprint == fingerprint
    ):
        return {
            "indexed": False,
            "reason": "up_to_date",
            "chunks": collection.count(),
        }

    return {
        "indexed": True,
        **rebuild_index(docs_dir),
    }


def vector_search(
    query: str,
    top_k: int = 3,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = str(query or "").strip()

    if not query or not (
        api_key or get_api_key()
    ):
        return []

    collection = get_collection()

    total = collection.count()

    if total == 0:
        return []

    query_vector = _embed_texts(
        [query],
        task_type="RETRIEVAL_QUERY",
        api_key=api_key,
    )[0]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(
            max(int(top_k), 1),
            total,
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = (results.get("ids") or [[]])[0]
    documents = (
        results.get("documents") or [[]]
    )[0]
    metadatas = (
        results.get("metadatas") or [[]]
    )[0]
    distances = (
        results.get("distances") or [[]]
    )[0]

    output: List[Dict[str, Any]] = []
    seen_files = set()

    for item_id, text, meta, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        meta = meta or {}

        filename = str(
            meta.get("file_name", "")
        )

        if filename in seen_files:
            continue

        seen_files.add(filename)

        excerpt = re.sub(
            r"\s+",
            " ",
            str(text or ""),
        ).strip()

        if len(excerpt) > 450:
            excerpt = (
                excerpt[:450].rsplit(" ", 1)[0]
                + "..."
            )

        output.append({
            "title": str(
                meta.get("source_title")
                or filename
                or "مستند معرفي"
            ),
            "url": str(
                meta.get("source_url") or ""
            ),
            "filename": filename,
            "score": round(
                1.0 - float(distance or 0.0),
                4,
            ),
            "excerpt": excerpt,
            "chunk_id": item_id,
        })

    return output
