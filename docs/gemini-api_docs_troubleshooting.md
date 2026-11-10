# Gemini API Troubleshooting and Error Handling

Official guide for resolving common errors, rate limit exceptions, and status codes.
Source: https://ai.google.dev/gemini-api/docs/troubleshooting

---

## 1. Common HTTP Status Codes

### `400 Bad Request / INVALID_ARGUMENT`
- **Cause:** Missing required parameters, malformed JSON structure, unsupported model name, or invalid schema configuration.
- **Resolution:** Verify model name strings (e.g., `gemini-2.5-flash`), check token counts against payload limits, and ensure correct data types.

### `403 Permission Denied / PERMISSION_DENIED`
- **Cause:** Invalid API key, key without billing enabled, or region restrictions.
- **Resolution:** Verify the API key in Google AI Studio, ensure IP/HTTP referrer restrictions match the runtime server, and check geographic availability.

### `404 Not Found / NOT_FOUND`
- **Cause:** Requesting a deprecated or non-existent model resource path.
- **Resolution:** Check the active models list and update outdated model identifiers.

### `429 Too Many Requests / RESOURCE_EXHAUSTED`
- **Cause:** Exceeding requests-per-minute (RPM), tokens-per-minute (TPM), or daily quota (RPD).
- **Resolution:** Implement exponential backoff retries, switch to Pay-as-you-go billing, or apply context caching to reduce token volume.

### `500 / 503 Internal Server Error`
- **Cause:** Transient backend issues or heavy infrastructure load.
- **Resolution:** Retry the request using an exponential backoff strategy with jitter.

---

## 2. Blocked Content and Safety Filters
If a response returns empty text with a `finish_reason` set to `SAFETY`:
- Inspect `prompt_feedback.block_reason` or candidate `safety_ratings`.
- Adjust safety filter thresholds in `safety_settings` to allow acceptable use cases while maintaining compliance.

---

## 3. Empty Generation or Truncation
- If `finish_reason` is `MAX_TOKENS`, increase `max_output_tokens` in `generation_config`.
- If `finish_reason` is `RECITATION`, ensure prompts do not require reciting copyrighted material verbatim.
