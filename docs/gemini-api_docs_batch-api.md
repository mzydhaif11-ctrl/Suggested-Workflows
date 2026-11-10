# Batch API in Gemini API

Official guide for asynchronous, high-throughput batch processing of prompts and embeddings.
Source: https://ai.google.dev/gemini-api/docs/batch-api

---

## 1. Overview
The Batch API allows submitting large volumes of requests asynchronously without blocking standard interactive rate limits (RPM/TPM). It is ideal for non-real-time workloads such as backfilling dataset annotations, mass evaluations, and bulk vector embedding generation.

---

## 2. Submission Methods
- **Inline Requests:** For smaller batches (under 20MB payload), requests can be supplied directly in an array within the creation call.
- **Input File (JSONL):** For large jobs (up to 2GB), requests are written line-by-line into a JSON Lines (`.jsonl`) file, uploaded via the Files API, and referenced by name.

---

## 3. Code Example: Batch Job Creation (Python)
```python
from google import genai
from google.genai import types

client = genai.Client()

# Example using inline batch requests
inline_requests = [
    types.GenerateContentRequest(
        contents=[types.Content(parts=[types.Part.from_text("Explain quantum computing")])],
    ),
    types.GenerateContentRequest(
        contents=[types.Content(parts=[types.Part.from_text("Explain vector databases")])],
    ),
]

batch_job = client.batches.create(
    model="gemini-2.5-flash",
    src=inline_requests,
    config={"display_name": "example-inline-batch"},
)

print(f"Created batch job: {batch_job.name}")
