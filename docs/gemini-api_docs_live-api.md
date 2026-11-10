# Gemini Live API Overview

Official guide for real-time, low-latency, bidirectional audio and video interactions.
Source: https://ai.google.dev/gemini-api/docs/live-api

---

## 1. Overview
The Gemini Live API enables developers to build low-latency, conversational voice and video applications. Built on WebSockets, it supports real-time multimodal streaming and natural turn-taking with model interruption handling.

---

## 2. Key Capabilities
- **Bidirectional Streaming:** Continuous streaming of audio/video frames to the model while concurrently receiving streaming audio/text responses.
- **Barge-in / Interruption:** Users can interrupt the model's speech at any moment; the model stops generating audio immediately.
- **Multimodal Inputs:** Accepts real-time webcam streams, microphone audio buffers, and text messages concurrently.

---

## 3. Communication Protocol
- Uses secure **WebSockets** (`wss://generativelanguage.googleapis.com/ws/...`).
- Client messages send real-time chunks using `realtimeInput` messages (base64 PCM audio or JPEG frames).
- Server messages stream back `serverContent` containing incremental audio/text chunks and tool-call proposals.

---

## 4. Ephemeral Tokens
For frontend mobile or web clients, avoid bundling your primary API key. Request short-lived, single-use ephemeral tokens from your backend server to authenticate client WebSocket connections directly with Google.
