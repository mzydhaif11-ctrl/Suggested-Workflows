# Image Understanding in Gemini API

Official guide for analyzing images, multi-image reasoning, and visual task processing.
Source: https://ai.google.dev/gemini-api/docs/image-understanding

---

## 1. Overview
Gemini models provide native support for image understanding tasks including detailed image description, text reading (OCR), chart interpretation, object localization, and visual comparison across multiple images.

---

## 2. Input Methods
- **Inline Image Data:** Pass images encoded in Base64 directly within prompt parts (suitable for single or small images under 20MB).
- **Files API:** Upload images via `client.files.upload` prior to inference (recommended for large images, high-resolution formats, or multi-image workflows).

---

## 3. Code Example: Image Analysis (Python)
```python
from google import genai
from PIL import Image

client = genai.Client()

# Load an image locally
image = Image.open("sample.jpg")

# Generate description and answers
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[image, "Describe the primary elements and details visible in this image."],
)

print(response.text)
