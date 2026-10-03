# Function Calling and Tools in Gemini API

Official guide for extending Gemini capabilities using structured tool calls and functions.
Source: https://ai.google.dev/gemini-api/docs/tools

---

## 1. Overview
Function calling lets developers define custom client-side functions and supply them to the model. Gemini determines when to call a tool based on the user's prompt and outputs structured JSON arguments matching the function schema.

---

## 2. Supported Tool Types
- **Custom Functions:** User-defined Python functions or schema-based tools.
- **Google Search Grounding:** Real-time web search tool integrated directly into model reasoning.
- **Code Execution:** Built-in Python code sandbox that allows Gemini to execute code and observe results.

---

## 3. Function Calling Workflow
1. Declare function signatures and definitions in the API request configuration.
2. Send the conversation prompt to Gemini.
3. If necessary, Gemini returns a `functionCall` part specifying function name and arguments.
4. Execute the function locally in your application environment.
5. Return the result back to Gemini in a `functionResponse` part to generate the final user reply.

---

## 4. Code Example (Python)
```python
from google import genai
from google.genai import types

def get_current_weather(location: str) -> str:
    """Gets the current weather for a given city location."""
    return f"The weather in {location} is 24°C and sunny."

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the weather like in Riyadh right now?",
    config=types.GenerateContentConfig(
        tools=[get_current_weather],
    ),
)

# Access generated tool call or final grounded reply
print(response.function_calls or response.text)
