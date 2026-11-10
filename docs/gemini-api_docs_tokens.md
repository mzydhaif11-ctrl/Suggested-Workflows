# Understand and Count Tokens in Gemini API

Official guide for calculating token counts and managing context limits.
Source: https://ai.google.dev/gemini-api/docs/tokens

---

## 1. Overview
Models process information in units called **Tokens**. A token can be a single character, a subword, or a full word depending on the language. Counting tokens prior to sending requests helps calculate costs accurately and ensures prompts remain within the model's context window.

---

## 2. Counting Tokens Before Generation
You can calculate the exact token count of any input payload using the dedicated `countTokens` endpoint without consuming generation quota:

```python
from google import genai

client = genai.Client()

response = client.models.count_tokens(
    model="gemini-2.5-flash",
    contents="Calculate the exact token count for this input text."
)

print(f"Total Tokens: {response.total_tokens}")
