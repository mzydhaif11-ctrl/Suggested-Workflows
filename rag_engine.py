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
                # تم إصلاح هذا السطر ليكون كاملاً وصحيحاً
                if query.lower() in lower_title:
                    score += 25.0

                tf = doc["tokens"].get(token, 0)
                if tf > 0:
                    df = doc_freq.get(token, 1)
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    k1 = 1.2
                    b = 0.75
                    avg_len = 1200
                    norm_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
                    score += (idf * norm_tf)

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

    def _extract_relevant_excerpt(self, content: str, query_tokens: List[str], max_chars: int = 450) -> str:
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 35]
        best_p = ""
        best_score = -1

        for p in paragraphs:
            if p.startswith("-") and len(p.splitlines()) > 5:
                continue
            p_lower = p.lower()
            p_score = sum(1 for t in query_tokens if t in p_lower)
            if p_score > best_score:
                best_score = p_score
                best_p = p

        if not best_p and paragraphs:
            best_p = paragraphs[0]

        clean_text = re.sub(r'#+\s*', '', best_p)
        clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars].rsplit(" ", 1)[0] + "..."
            
        return clean_text

    def answer_query(self, query: str, provided_api_key: Optional[str] = None) -> Dict[str, Any]:
        api_key = provided_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        retrieved_docs = self.search(query, top_k=3)

        # 1. إذا وجد مستندات مطابقة في مجلد docs
        if retrieved_docs:
            if api_key:
                context_text = "\n\n".join([
                    f"Document: {d['title']}\nURL: {d['url']}\nContent Excerpt:\n{d['excerpt']}"
                    for d in retrieved_docs
                ])
                system_msg = (
                    "You are an expert technical support assistant. "
                    "Answer the customer's inquiry in Arabic accurately and helpfully, strictly grounded in the official documentation excerpts provided. "
                    "Always cite the source document name and include its official URL."
                )
                prompt = f"{system_msg}\n\nDocumentation Context:\n{context_text}\n\nCustomer Inquiry: {query}\n\nHelpful response in Arabic:"
                
                try:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt,
                    )
                    return {
                        "query": query,
                        "answer": response.text,
                        "sources": retrieved_docs
                    }
                except Exception as e:
                    top_doc = retrieved_docs[0]
                    return {
                        "query": query,
                        "answer": f"تنبيه: حدث خطأ أثناء الاتصال بنموذج Gemini ({str(e)}).\n\n📌 **مقتطف من التوثيق المتوفر:**\n{top_doc['excerpt']}\n🔗 **المصدر:** {top_doc['url']}",
                        "sources": retrieved_docs
                    }

            # خطة بديلة في حال عدم توفر المفتاح
            top_doc = retrieved_docs[0]
            answer_parts = [
                f"بناءً على التوثيق الرسمي في قسم **{top_doc['title']}**:\n",
                f"📌 **مقتطف من التوثيق:**\n{top_doc['excerpt']}\n",
                f"🔗 **المصدر:** {top_doc['url']}"
            ]
            if len(retrieved_docs) > 1:
                answer_parts.append(f"\n💡 **للمزيد من التفاصيل:** راجع أيضاً قسم **{retrieved_docs[1]['title']}**.")

            return {
                "query": query,
                "answer": "\n".join(answer_parts),
                "sources": retrieved_docs
            }

        # 2. إذا لم يجد أي تطابق في التوثيق -> يتحول تلقائياً إلى Gemini كذكاء اصطناعي عام
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=query,
                )
                return {
                    "query": query,
                    "answer": response.text,
                    "sources": []
                }
            except Exception as e:
                return {
                    "query": query,
                    "answer": f"خطأ في الاتصال بنموذج الذكاء الاصطناعي: {str(e)}",
                    "sources": []
                }

        # في حال عدم وجود المفتاح إطلاقاً
        return {
            "query": query,
            "answer": "تنبيه: مفتاح GEMINI_API_KEY غير موجود في متغيرات البيئة على Render.",
            "sources": []
        }
