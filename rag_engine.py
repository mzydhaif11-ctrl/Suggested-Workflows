import os
import re
import math
import json
import time
import urllib.request
import urllib.error
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
        
        # 1. تهيئة الاتصال بذاكرة ChromaDB باستخدام تضمين Gemini
        self._init_chroma()
        
        # 2. فهرسة الملفات المحلية بالبحث النصي
        self.index_documents()

    def _init_chroma(self):
        """إنشاء اتصال آمن مع سحابة ChromaDB وربطها بنموذج text-embedding-004"""
        if not chromadb or not embedding_functions:
            return

        chroma_url = os.environ.get("CHROMA_URL")
        chroma_api_key = os.environ.get("CHROMA_API_KEY")
        chroma_tenant = os.environ.get("CHROMA_TENANT")
        gemini_api_key = os.environ.get("GEMINI_API_KEY")

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

                # إعداد دالة التضمين الرسمية من Google Gemini
                gemini_ef = None
                if gemini_api_key:
                    gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                        api_key=gemini_api_key,
                        model_name="models/text-embedding-004"
                    )

                self.collection = self.chroma_client.get_or_create_collection(
                    name="mowjat_advisor_memory",
                    embedding_function=gemini_ef
                )
                print(" تم تفعيل ذاكرة ChromaDB بنموذج Gemini text-embedding-004 بنجاح.")
            except Exception as e:
                print(f" تعذر إعداد ChromaDB مع تضمين Gemini: {e}")

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
                print(f"خطأ أثناء فهرسة الملف {filename}: {e}")

        if self.documents:
            total_tokens = sum(doc["token_count"] for doc in self.documents)
            self.avg_doc_len = total_tokens / len(self.documents) or 1.0

    def _search_chroma_memory(self, query: str, n_results: int = 2) -> List[str]:
        """استرجاع المحادثات السابقة دلالياً باستخدام تضمين text-embedding-004"""
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
            print(f"خطأ أثناء استرجاع الذاكرة: {e}")
        return []

    def _save_to_chroma_memory(self, query: str, answer: str):
        """تخزين التفاعل في ChromaDB وتوليد المتجهات تلقائياً عبر Gemini"""
        if not self.collection:
            return
        try:
            mem_id = f"mem_{int(time.time() * 1000)}"
            text_to_save = f"سؤال المستخدم السابق: {query}\nإجابة المستشار السابقة: {answer}"
            self.collection.add(
                ids=[mem_id],
                documents=[text_to_save],
                metadatas=[{"type": "chat_memory", "timestamp": str(time.time())}]
            )
        except Exception as e:
            print(f"خطأ أثناء الحفظ في الذاكرة: {e}")

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

    def _extract_relevant_excerpt(self, content: str, query_tokens: List[str], max_chars: int = 450) -> str:
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
        clean_text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars].rsplit(" ", 1)[0] + "..."
        return clean_text

    def answer_query(self, query: str, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        retrieved_docs = self.search(query, top_k=3)
        past_memories = self._search_chroma_memory(query, n_results=2)

        if api_key:
            context_blocks = []
            if retrieved_docs:
                doc_context = "\n\n".join([
                    f"وثيقة: {d['title']}\nالرابط: {d['url']}\nالمقتطف:\n{d['excerpt']}"
                    for d in retrieved_docs
                ])
                context_blocks.append(f"[مقتطفات التوثيق الرسمي]:\n{doc_context}")
            
            if past_memories:
                memory_context = "\n---\n".join(past_memories)
                context_blocks.append(f"[سياق من محادثات سابقة]:\n{memory_context}")

            combined_context = "\n\n".join(context_blocks)

            system_instruction = (
                "أنت مستشار ودعم فني متخصص لمنصة موجة البيان ومكتبات Google Gemini API. "
                "أجب عن استفسار العميل باللغة العربية بدقة وأسلوب مهني وواضح، معتمداً حصراً على المقتطفات والتوثيق المرفق. "
                "إذا كان هناك سياق من محادثات سابقة فاستخدمه لربط الحديث. "
                "اذكر دائماً اسم الوثيقة ورابطها الرسمي عند الإشارة إلى التوثيق."
            )
            
            user_content = f"السياق:\n{combined_context}\n\nسؤال المستخدم: {query}" if combined_context else f"سؤال المستخدم: {query}"

            models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash"]
            for model_name in models_to_try:
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                payload = {
                    "system_instruction": {
                        "parts": [{"text": system_instruction}]
                    },
                    "contents": [{
                        "parts": [{"text": user_content}]
                    }],
                    "generationConfig": {
                        "temperature": 0.2
                    }
                }
                try:
                    req = urllib.request.Request(
                        api_url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": api_key
                        }
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        answer_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                        
                        self._save_to_chroma_memory(query, answer_text)

                        return {
                            "query": query,
                            "answer": answer_text,
                            "sources": retrieved_docs
                        }
                except urllib.error.HTTPError as http_err:
                    error_details = http_err.read().decode("utf-8", errors="ignore")
                    print(f"فشل الطلب مع {model_name} (HTTP {http_err.code}): {error_details}")
                    continue
                except Exception as e:
                    print(f"خطأ أثناء الاتصال بنموذج {model_name}: {e}")
                    continue

        if not retrieved_docs:
            return {
                "query": query,
                "answer": "مرحباً بك في موجة البيان. يرجى التأكد من ضبط متغير البيئة GEMINI_API_KEY للحصول على إجابات ذكية.",
                "sources": []
            }

        top_doc = retrieved_docs[0]
        return {
            "query": query,
            "answer": f"بناءً على التوثيق في {top_doc['title']}:\n{top_doc['excerpt']}\n\nالمصدر: {top_doc['url']}",
            "sources": retrieved_docs
        }
