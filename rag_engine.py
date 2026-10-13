import os
import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional

from google import genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
WORKSPACE_PAGES_DIR = os.path.join(BASE_DIR, "docs")

class GroundedRAGEngine:
    def __init__(self, pages_dir: str = WORKSPACE_PAGES_DIR):
        self.pages_dir = pages_dir
        self.documents = []
        
        if not os.path.exists(self.pages_dir):
            try:
                os.makedirs(self.pages_dir, exist_ok=True)
            except:
                pass
                
        self.index_documents()

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r'\b[a-zA-Z0-9_\u0600-\u06FF]+\b', str(text).lower())

    def index_documents(self):
        self.documents = []
        if not os.path.exists(self.pages_dir):
            return

        for filename in os.listdir(self.pages_dir):
            if filename == "index.md" or not filename.endswith(".md"):
                continue

            filepath = os.path.join(self.pages_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    continue

                original_url = "https://ai.google.dev/gemini-api/docs"
                url_match = re.search(r"original_url:\s*(https?://[^\s\n]+)", content, re.IGNORECASE)
                if url_match:
                    original_url = url_match.group(1).strip()

                heading_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                if heading_match:
                    title = heading_match.group(1).strip()
                    main_body = content[heading_match.start():]
                else:
                    title = filename.replace(".md", "").replace("_", " ").title()
                    main_body = re.sub(r"^---[\s\S]*?---\n", "", content)

                tokens = self._tokenize(main_body + " " + title)
                token_count = len(tokens)

                if token_count == 0:
                    continue

                self.documents.append({
                    "filename": filename,
                    "filepath": filepath,
                    "title": title,
                    "url": original_url,
                    "content": main_body,
                    "tokens": Counter(tokens),
                    "token_count": token_count
                })
            except:
                pass

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
            doc_len = doc["token_count"]
            lower_title = doc["title"].lower()

            for token in query_tokens:
                if token in lower_title:
                    score += 8.0
                if query.lower() in lower_
