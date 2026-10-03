# Text Generation in Gemini API

Official guide for prompt completion, streaming, and multi-turn conversations.
Source: https://ai.google.dev/gemini-api/docs/text-generation

---

## 1. Overview
Text generation covers fundamental capabilities of Gemini models, ranging from direct single-turn question answering to interactive multi-turn conversations and long-form content generation.

---

## 2. Generation Parameters
- **`temperature`:** Controls randomness. Lower values (e.g., 0.2) yield focused and deterministic output; higher values (e.g., 1.0) increase diversity and creativity.
- **`top_p`:** Nucleus sampling threshold. Selects tokens from the smallest subset whose cumulative probability exceeds `top_p`.
- **`max_output_tokens`:** Restricts the upper bound of generated tokens in the response.
- **`stop_sequences`:** A set of character sequences that cease further token output when encountered.

---

## 3. Streaming Responses (Python)
Stream tokens incrementally as they are generated to deliver instant user feedback:

```python
from google import genai

client = genai.Client()

response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Write a concise guide on using Qdrant vector database."
)

for chunk in response:
    print(chunk.text, end="")
