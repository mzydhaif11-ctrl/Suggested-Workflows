# Context Caching in Gemini API

Official guide for caching repetitive context and large token collections.
Source: https://ai.google.dev/gemini-api/docs/caching

---

## 1. Overview
Context caching allows developers to store large amounts of static context—such as lengthy technical manuals, codebases, or extensive chat histories—on Google's servers and reuse it across multiple generation requests without paying the token input fee repeatedly.

---

## 2. Key Benefits
- **Cost Efficiency:** Significantly reduces input token costs for recurring prompts.
- **Latency Reduction:** Speeds up response times by eliminating repeated processing of large static prefixes.
- **High Capacity:** Allows caching millions of tokens for long-context reasoning.

---

## 3. How It Works
- Cached content has a configurable **Time-to-Live (TTL)**.
- Once created, a unique cache name is generated (e.g., `cachedContents/abc123xyz`).
- Subsequent `generateContent` calls can pass `cached_content` to reference the cached context.

---

## 4. Code Example (Python)
```python
from google import genai
from google.genai import types

client = genai.Client()

# Create cached content
cache = client.caches.create(
    model="gemini-2.5-flash",
    config=types.CreateCachedContentConfig(
        contents=["Detailed API documentation, system guidelines, and full knowledge base..."],
        ttl="3600s",
    ),
)

# Query using the cache
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Summarize the core requirements",
    config=types.GenerateContentConfig(
        cached_content=cache.name,
    ),
)

print(response.text)
