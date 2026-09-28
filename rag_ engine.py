import os, re, math, json, urllib.request from collections import Counter from typing import List, Dict,
Any, Optional WORKSPACE
PAGES
DIR =
_
_
"/.agents/workspace/pages" class GroundedRAGEngine: def
init
__
__(self,
pages
dir: str = WORKSPACE
PAGES
_
_
_
DIR): self.pages
_
dir = pages
dir self.documents = [] self.index
_
_
documents() def
_
tokenize(self, text: str) -> List[str]: return re.findall(r'\b[a-zA-Z0-9
\u0600-\u06FF]+\b'
_
, text.lower()) def
index
_
documents(self): self.documents = [] if not os.path.exists(self.pages
_
dir): return for filename in
os.listdir(self.pages
_
dir): if filename ==
"index.md" or not filename.endswith("
.md"): continue filepath =
os.path.join(self.pages
_
dir, filename) try: with open(filepath,
"r"
, encoding=
"utf-8") as f: content = f.read()
original
url =
_
"https://ai.google.dev/gemini-api/docs" url
_
match = re.search(r"^
---\s*\noriginal
url:\s*
_
(https?://[^\s\n]+)\s*\n---
"
, content, re.MULTILINE) if url
_
match: original
url = url
_
_
match.group(1)
heading_
match = re.search(r"^#\s+(.+)$"
, content, re.MULTILINE) title = heading_
match.group(1).strip() if
heading_
match else filename.replace("
.md"
,
"").title() main
_
body = content[heading_
match.start():] if
heading_
match else content tokens = self.
_
tokenize(main
_
body + " " + title) self.documents.append({ "filename"
:
filename,
"title"
: title,
"url"
: original
url,
"content"
: main
_
_
body,
"tokens"
: Counter(tokens),
"token
count"
:
_
len(tokens) }) except Exception as e: print(f"Indexing error: {e}") def search(self, query: str, top_
k: int = 3)
-> List[Dict[str, Any]]: query_
tokens = self.
_
tokenize(query) if not query_
tokens or not self.documents: return
[] results = [] total
_
docs = len(self.documents) doc
_
freq = {t: sum(1 for d in self.documents if t in
d["tokens"]) for t in set(query_
tokens)} for doc in self.documents: score = 25.0 if query.lower() in
doc["title"].lower() else 0.0 doc
len = doc["token
_
_
count"] or 1 for token in query_
tokens: tf =
doc["tokens"].get(token, 0) if tf > 0: df = doc
_
freq.get(token, 1) idf = math.log((total
_
docs - df + 0.5) / (df
+ 0.5) + 1.0) norm
_
tf = (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (doc
_
len / 1200))) score += idf * norm
tf if
_
score > 0: results.append({"title"
: doc["title"],
"url"
: doc["url"],
"score"
: round(score, 2),
"content"
:
doc["content"][:400]}) results.sort(key=lambda x: x["score"], reverse=True) return results[:top_
k] def
answer
_query(self, query: str, gemini
_
api
_
key: Optional[str] = None) -> Dict[str, Any]: api
_
key = gemini
_
api
_
key
or os.environ.get("GEMINI
API
_
_
KEY") retrieved
_
docs = self.search(query, top_
k=3) if not retrieved
docs: return
_
{"query"
: query,
"answer"
:
.لم يتم العثور على توثيق مباشر مطابق"
"
,
"sources"
: []} if api
_
key: context =
"\n\n"
ة"join([f.
:وثيق
{d['title']}\nرابط: {d['url']}\nمحتوى: {d['content']}" for d in retrieved
_
سياق"docs]) prompt = f
ر المصادرquery}\n\n{ :سؤال المستخدمn{context}\n\n\:التوثيق
:اﻹجابة بالعربية مع ذك
" api
url =
_
f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent" payload =
{"contents"
: [{"parts"
: [{"text"
: prompt}]}]} try: req = urllib.request.Request(api
url,
_
data=json.dumps(payload).encode("utf-8"), headers={"Content-Type"
:
"application/json"
,
"X-goog-api-key"
:
api
_
key}) with urllib.request.urlopen(req, timeout=12) as resp: res = json.loads(resp.read().decode("utf-8"))
return {"query"
: query,
"answer"
: res["candidates"][0]["content"]["parts"][0]["text"],
"sources"
:
retrieved
_
docs} except Exception: pass top = retrieved
_
docs[0] return {"query"
: query,
"answer"
ة"f :
** بناءً على وثيق
{top['title']}**:\n{top['content']}"
,
"sources"
: retrieved
_
docs}
