# Gemini API Pricing and Rate Limits

Official reference for API pricing tiers, token costs, and rate limits.
Source: https://ai.google.dev/gemini-api/docs/pricing

---

## 1. Overview of Tiers
The Gemini API provides two primary usage models:
- **Free Tier:** Enables prototyping and development with rate limits (RPM, TPM, RPD) at no financial cost. User prompt data may be logged to improve Google products.
- **Pay-as-you-go (Paid Tier):** Higher rate limits, enterprise-level scale, and private prompt handling without logging for product improvement.

---

## 2. Rate Limit Metrics
- **RPM (Requests Per Minute):** Maximum number of API calls permitted in a 60-second window.
- **TPM (Tokens Per Minute):** Maximum combined input and output tokens processed per minute.
- **RPD (Requests Per Day):** Total quota of allowable API calls in a 24-hour cycle.

---

## 3. Pricing Architecture (General Model)
- **Input Tokens:** Billed per 1 million prompt tokens. Prompts longer than 128k tokens have tiered pricing.
- **Output Tokens:** Billed per 1 million generated tokens.
- **Context Caching:** Billed at a fraction of the standard input rate per 1M tokens cached, plus a storage fee per hour for the Time-to-Live (TTL).

---

## 4. Best Practices for Cost Management
- Utilize **Flash models** (`gemini-2.5-flash`) for low-cost, high-volume production endpoints.
- Enable **Context Caching** for repetitive payloads (e.g., system instructions or document context exceeding 32k tokens).
- Implement client-side exponential backoff handling to manage `429 (Too Many Requests)` rate-limiting responses gracefully.

