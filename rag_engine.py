import os
import re
import math
import json
import time
from collections import Counter
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

PAGES_DIR = os.path.join(os.path.dirname(__file__), "pages")


class GroundedRAGEngine:
    def __init__(self, pages_dir: str = PAGES_DIR):
        self.pages_dir = pages_dir
        self.documents = []
        self.avg_doc_len = 1.0
        self.chroma_client = None
        self.collection = None
        self.genai_client = None

        # 1. Initialize Google Gen AI client
        self._init_genai()

        # 2. Initialize ChromaDB connection with Gemini embeddings
        self._init_chroma()

        # 3. Index local files with text search
        self.index_documents()

    def _init_genai(self):
        """Initialize Google Gen AI SDK client."""
        try:
            from google import genai
            api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                self.genai_client = genai.Client(api_key=api_key)
        except ImportError:
            pass

    def _init_chroma(self):
        """Create secure connection to ChromaDB cloud linked with Gemini embeddings."""
        if not chromadb or not embedding_functions:
            return

        chroma_url = os.environ.get("CHROMA_URL")
        chroma_api_key = os.environ.get("CHROMA_API_KEY")
        chroma_tenant = os.environ.get("CHROMA_TENANT")
        gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if chroma_url and chroma_api_key:
            try:
                connect_kwargs = {
                    "host": chroma_url,
                    "headers": {"Authorization": f"Bearer {chroma_api_key}"}
                }
                if chroma_tenant:
                    connect_kwargs["tenant"] = chroma_tenant
                    connect_kwargs["database"] = "default_database"

                self.chroma_client = chromadb.HttpClient(**connect_kwargs)

                # Setup official embedding function from Google Gemini
                gemini_ef = None
                if gemini_api_key:
                    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                        api_key=gemini_api_key,
                        model_name="gemini-embedding-2"
                    )

                self.collection = self.chroma_client.get_or_create_collection(
                    name="mowjat_advisor_memory",
                    embedding_function=gemini_ef
                )
                print("ChromaDB memory activated with Gemini gemini-embedding-2 successfully.")
            except Exception as e:
                print(f"Failed to setup ChromaDB with Gemini embeddings: {e}")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_\u0600-\u06FF]+", text.lower())

    def index_documents(self):
        self.documents = []
        if not os.path.exists(self.pages_dir):
            return

        for filename in os.listdir(self.pages_dir):
            if filename == "index.md" or not filename.endswith(".md"):
                continue

            filepath = os.path.join(self.pages_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                original_url = "https://ai.google.dev/gemini-api/docs"
                url_match = re.search(r"^---\s*[\r\n]+original_url:\s*(\S+)", content, re.MULTILINE)
                if url_match:
                    original_url = url_match.group(1)

                heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                if heading_match:
                    title = heading_match.group(1).strip()
                    main_body = content[heading_match.start():]
                else:
                    title = filename.replace(".md", "").replace("_", " ").title()
                    main_body = re.sub(r"^---[\s\S]*?---[\r\n]+", "", content)

                tokens = self._tokenize(main_body + " " + title)

                self.documents.append({
                    "filename": filename,
                    "filepath": filepath,
                    "title": title,
                    "url": original_url,
                    "content": main_body,
                    "tokens": Counter(tokens),
                    "token_count": len(tokens)
                })
            except Exception as e:
                print(f"Error indexing file {filename}: {e}")

        if self.documents:
            total_tokens = sum(doc["token_count"] for doc in self.documents)
            self.avg_doc_len = total_tokens / len(self.documents) or 1.0

    def _search_chroma_memory(self, query: str, n_results: int = 2) -> List[str]:
        """Retrieve past conversations semantically using Gemini embeddings."""
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if results and results.get("documents") and results["documents"][0]:
                return results["documents"][0]
        except Exception as e:
            print(f"Error retrieving memory: {e}")
        return []

    def _save_to_chroma_memory(self, query: str, answer: str):
        """Store interaction in ChromaDB with auto-generated vectors via Gemini."""
        if not self.collection:
            return
        try:
            mem_id = f"mem_{int(time.time() * 1000)}"
            text_to_save = f"User question: {query}\nAdvisor answer: {answer}"
            self.collection.add(
                ids=[mem_id],
                documents=[text_to_save],
                metadatas=[{"type": "chat_memory", "timestamp": str(time.time())}]
            )
        except Exception as e:
            print(f"Error saving to memory: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.documents:
            return []

        results = []
        total_docs = len(self.documents)

        doc_freq = {}
        for token in set(query_tokens):
            doc_freq[token] = sum(1 for doc in self.documents if token in doc["tokens"])

        for doc in self.documents:
            score = 0.0
            doc_len = doc["token_count"] or 1
            lower_title = doc["title"].lower()

            for token in query_tokens:
                if token in lower_title:
                    score += 8.0
                if query.lower() in lower_title:
                    score += 25.0

            for token in query_tokens:
                tf = doc["tokens"].get(token, 0)
                if tf > 0:
                    df = doc_freq.get(token, 1)
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    k1 = 1.2
                    b = 0.75
                    norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len)))
                    score += idf * norm_tf

            if score > 0:
                excerpt = self._extract_relevant_excerpt(doc["content"], query_tokens)
                results.append({
                    "title": doc["title"],
                    "url": doc["url"],
                    "filename": doc["filename"],
                    "score": round(score, 2),
                    "excerpt": excerpt
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _extract_relevant_excerpt(self, content: str, query_tokens: list, max_chars: int = 450) -> str:
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 35]
        best_p = ""
        best_score = -1

        for p in paragraphs:
            if p.startswith(" - ") and len(p.splitlines()) > 5:
                continue
            p_lower = p.lower()
            p_score = sum(1 for t in query_tokens if t in p_lower)
            if p_score > best_score:
                best_score = p_score
                best_p = p

        if not best_p and paragraphs:
            best_p = paragraphs[0]

        clean_text = re.sub(r"#+\s*", "", best_p)
        clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars].rsplit(" ", 1)[0] + "..."
        return clean_text

    def answer_query(self, query: str, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        retrieved_docs = self.search(query, top_k=3)
        past_memories = self._search_chroma_memory(query, n_results=2)

        if api_key and self.genai_client:
            context_blocks = []
            if retrieved_docs:
                doc_context = "\n\n".join([
                    f"Document: {d['title']}\nURL: {d['url']}\nExcerpt:\n{d['excerpt']}"
                    for d in retrieved_docs
                ])
                context_blocks.append(f"[Official Documentation Excerpts]:\n{doc_context}")

            if past_memories:
                memory_context = "\n---\n".join(past_memories)
                context_blocks.append(f"[Context from Past Conversations]:\n{memory_context}")

            combined_context = "\n\n".join(context_blocks)

            system_instruction = (
                "You are a specialized consultant and technical support for Mowjn AI-Bayan platform "
                "and Google Gemini API libraries. Answer customer inquiries in Arabic with precision, "
                "professional style, and clarity, relying exclusively on the provided excerpts and documentation. "
                "If there is context from past conversations, use it to connect the discussion. "
                "Always mention the document name and its official URL when referencing documentation."
            )

            user_content = f"Context:\n{combined_context}\n\nUser Question: {query}" if combined_context else f"User Question: {query}"

            try:
                response = self.genai_client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=user_content,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 0.2,
                    }
                )
                answer_text = response.text.strip()

                self._save_to_chroma_memory(query, answer_text)

                return {
                    "query": query,
                    "answer": answer_text,
                    "sources": retrieved_docs
                }
            except Exception as e:
                print(f"Error generating response: {e}")

        if not retrieved_docs:
            return {
                "query": query,
                "answer": "Welcome to Mowjn AI-Bayan. Please ensure GEMINI_API_KEY environment variable is set for intelligent responses.",
                "sources": []
            }

        top_doc = retrieved_docs[0]
        return {
            "query": query,
            "answer": f"Based on documentation in {top_doc['title']}:\n{top_doc['excerpt']}\n\nSource: {top_doc['url']}",
            "sources": retrieved_docs
        }
