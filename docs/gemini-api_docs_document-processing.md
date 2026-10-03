# Document Processing and PDF Understanding in Gemini API

Official guide for processing PDF documents, text extraction, and structural analysis.
Source: https://ai.google.dev/gemini-api/docs/document-processing

---

## 1. Overview
Gemini models feature native multimodal processing for PDF documents, allowing direct analysis of mixed text, tables, and embedded images without requiring external OCR preprocessing tools.

---

## 2. Key Capabilities
- **Direct PDF Ingestion:** Upload and process documents containing up to thousands of pages within the context window.
- **Table and Visual Extraction:** Extract data from complex multi-column tables, charts, and infographics accurately.
- **Visual Question Answering:** Query specific sections, clauses, or diagrams directly from page layouts.

---

## 3. Uploading Documents via Files API (Python)
For large documents exceeding inline size limits, use the Files API before generating content:

```python
from google import genai

client = genai.Client()

# Upload the document
sample_file = client.files.upload(file="path/to/report.pdf")

# Generate insights using the uploaded document
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[sample_file, "Summarize the key financial findings in this document."]
)

print(response.text)
