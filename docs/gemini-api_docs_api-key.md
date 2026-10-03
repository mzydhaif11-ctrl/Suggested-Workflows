# Using and Securing Gemini API Keys

Official guide for generating, securing, and managing Google Gemini API keys.
Source: https://ai.google.dev/gemini-api/docs/api-key

---

## 1. Getting an API Key
- Access the key creation console at **Google AI Studio** (`aistudio.google.com`).
- Create a new project or select an existing Google Cloud project.
- Click **Get API key** and generate a new key for development.

---

## 2. Best Practices for Key Security
- **Never hardcode keys:** Do not commit API keys directly into public repositories, frontend client scripts, or static HTML files.
- **Use Environment Variables:** Load keys via environment variables on the backend runtime (e.g., `os.environ["GEMINI_API_KEY"]`).
- **Use API Key Restrictions:** Restrict key usage in Google Cloud Console by limiting endpoints strictly to Generative Language APIs and restricting allowed IP addresses or referrers.
- **Rotate Compromised Keys:** If an API key is accidentally committed or leaked, immediately delete or regenerate it via Google AI Studio.

---

## 3. Passing Keys in Client Requests
### Standard Environment Variable Loading (Python)
The `google-genai` SDK automatically detects the `GEMINI_API_KEY` environment variable:
```python
import os
from google import genai

# Automatically reads os.environ["GEMINI_API_KEY"]
client = genai.Client()
