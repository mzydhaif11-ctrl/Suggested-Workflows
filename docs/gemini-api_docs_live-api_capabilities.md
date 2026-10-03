# Live API Capabilities Guide

Official guide covering detailed features, modalities, and configuration in the Gemini Live API.
Source: https://ai.google.dev/gemini-api/docs/live-api/capabilities

---

## 1. Overview
The Live API provides advanced real-time controls beyond simple audio exchange, including function execution during live streams, voice configuration, and context updates mid-session.

---

## 2. Voice Configuration and Audio Formats
- **Input Audio:** Linear PCM 16-bit little-endian, sample rate typically 16kHz or 24kHz.
- **Output Audio:** 24kHz PCM audio chunks streamed in real-time.
- **Voice Selection:** Configure pre-built voices (such as Aoede, Puck, Charon, Fenrir, Kore) via the session configuration message.

---

## 3. Real-Time Tool Calling
- Define functions in the initial `setup` message.
- When an action is needed during conversation, the model yields a structured tool call immediately while holding conversational context.
- The client executes the tool locally and returns the result back over the same active WebSocket channel.

---

## 4. Session Controls and Context Updates
- Dynamic system prompts: Send context updates mid-conversation to adapt model behavior.
- Turn-taking management: Configure end-of-turn thresholds and interruptions to control conversational pacing.
