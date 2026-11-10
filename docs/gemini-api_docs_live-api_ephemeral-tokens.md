# Ephemeral Tokens in Gemini Live API

Official guide for generating short-lived authentication tokens for client-side Live API connections.
Source: https://ai.google.dev/gemini-api/docs/live-api/ephemeral-tokens

---

## 1. Overview
Direct WebSocket connections from frontend applications (web browsers, mobile apps) must never expose primary API keys. Ephemeral tokens provide short-lived, constrained credentials minted by a trusted backend server to allow clients to initiate Live API sessions securely.

---

## 2. Token Lifecycle and Security
- **Short Duration:** Tokens are valid for a brief window (typically up to 30 minutes) and expire automatically.
- **Single Session Scope:** Designed exclusively to authenticate one active bidirectional streaming session.
- **Zero Frontend Leakage:** The master `GEMINI_API_KEY` remains strictly protected inside backend environment variables.

---

## 3. Generating Tokens on the Backend (Python)
Your FastAPI server can generate an ephemeral token on behalf of an authenticated user:

```python
from google import genai
from google.genai import types

client = genai.Client()

def create_client_session_token():
    token = client.auth_tokens.create(
        config=types.CreateAuthTokenConfig(
            ttl="1800s", # 30 minutes validity
        )
    )
    return token.name
