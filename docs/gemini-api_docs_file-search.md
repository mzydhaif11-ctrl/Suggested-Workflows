# File Search and Retrieval in Gemini API

Official guide for searching documents and grounding responses using uploaded files.
Source: https://ai.google.dev/gemini-api/docs/file-search

---

## 1. Overview
Gemini allows developers to upload diverse file collections and search their content during model inference, providing accurate citations and grounding answers directly in source materials.

---

## 2. Key Capabilities
- **Direct Semantic Search:** Automatically extracts relevant text snippets from uploaded files to answer specific questions.
- **Support for Varied Formats:** Ingests plain text, PDFs, Markdown documentation, and source code files.
- **Source Attributions:** Returns references and grounding metadata pointing to specific segments of the uploaded documents.

---

## 3. Code Example (Python)
```python
from google import genai

client = genai.Client()

# Upload a source document
corpus_doc = client.files.upload(file="knowledge_base.pdf")

# Generate grounded answers using the document
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        corpus_doc,
        "Find and explain the instructions for configuring Qdrant storage."
    ]
)

print(response.text)
