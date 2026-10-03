# Prompt Design Strategies for Gemini API

Official guide for optimizing prompts, controlling output quality, and guiding model reasoning.
Source: https://ai.google.dev/gemini-api/docs/prompting-strategies

---

## 1. General Principles
- **Be Clear and Specific:** Provide detailed context, expected role, task boundaries, and clear constraints.
- **Provide Few-Shot Examples:** Supply high-quality input-output pairs to illustrate the exact target format.
- **Instruct Rather Than Restrict:** Focus on telling the model what to do instead of only what to avoid.

---

## 2. Key Prompting Techniques

### System Instructions
Define system-level behavior and identity separately from standard conversational turns:
```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="How do I store vectors?",
    config=types.GenerateContentConfig(
        system_instruction="You are Mowjh Al-Bayan technical assistant. Answer concisely with code samples.",
    ),
)
print(response.text)
