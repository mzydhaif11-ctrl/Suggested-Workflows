# Gemini API Reference Documentation

Official Reference for the Google Gemini Developer API.
Source: https://ai.google.dev/api

---

## 1. Authentication
All API requests require authentication via an API Key.
- Pass the key as a URL parameter: `?key=YOUR_API_KEY`
- Or pass via the `x-goog-api-key` HTTP request header.

---

## 2. Base Endpoints
- **REST Base URL:** `https://generativelanguage.googleapis.com/v1beta/`
- **SDK Import (Python):** `from google import genai`

---

## 3. Core Models
- `gemini-2.5-flash`: Fast, optimized for multimodal tasks and low latency.
- `gemini-2.5-pro`: Advanced reasoning, complex code generation, and complex analysis.
- `text-embedding-004`: Text embeddings generation with 768 dimensions.

---

## 4. Primary Methods

### `generateContent`
Generates a response from the model given input prompts (text, images, audio, video).
- **Endpoint:** `POST /v1beta/models/{model}:generateContent`
- **Request Body Parameters:**
  - `contents`: Array of `Content` objects containing `role` ('user' or 'model') and `parts`.
  - `systemInstruction`: Optional system instructions directing the model's behavior.
  - `generationConfig`: Controls generation parameters (`temperature`, `topP`, `maxOutputTokens`, `responseMimeType`).
  - `safetySettings`: Thresholds for blocking harmful content categories.

### `streamGenerateContent`
Streams chunks of model generation in real-time as Server-Sent Events (SSE).
- **Endpoint:** `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`

### `embedContent`
Generates vector embeddings for a given piece of text for semantic search and RAG.
- **Endpoint:** `POST /v1beta/models/text-embedding-004:embedContent`
- **Request Body Parameters:**
  - `content`: The text content to embed.
  - `taskType`: Optional task type (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`).

---

## 5. Structured Outputs
You can force the model to respond in JSON adhering to a specific schema:
```python
from google import genai
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str
    price: float

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Give me details of a laptop',
    config={
        'response_mime_type': 'application/json',
        'response_schema': ProductInfo,
    },
)
print(response.text)
