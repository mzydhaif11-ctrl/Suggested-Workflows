
import os
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
from adaptive_memory_engine import AdaptiveMemoryEngine

# ─── Logging Setup ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────
class Config:
    """Centralized configuration from environment variables."""

    def __init__(self):
        self.api_key: str = os.environ.get("GOOGLE_API_KEY", "")
        self.model_name: str = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
        self.embedding_model: str = os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-004"
        )
        self.max_tokens: int = int(os.environ.get("MAX_TOKENS", "1024"))
        self.temperature: float = float(os.environ.get("TEMPERATURE", "0.7"))
        self.max_workers: int = int(os.environ.get("MAX_WORKERS", "5"))
        self.memory_db_path: str = os.environ.get(
            "MEMORY_DB_PATH", "mowjn_memory.db"
        )
        self.decay_rate: float = float(os.environ.get("DECAY_RATE", "0.01"))
        self.similarity_threshold: float = float(
            os.environ.get("SIMILARITY_THRESHOLD", "0.5")
        )
        self.top_k_memories: int = int(os.environ.get("TOP_K_MEMORIES", "3"))

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Set it via os.environ or a .env file."
            )


config = Config()


# ─── Client Initialization ──────────────────────────────────────
def init_client() -> None:
    """Initialize the Gemini client with API key."""
    try:
        genai.configure(api_key=config.api_key)
        logger.info(f"Gemini client initialized — model: {config.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        raise


# ─── Embedding Helper ───────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text using Gemini."""
    try:
        result = genai.embed_content(
            model=config.embedding_model,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed for '{text[:50]}...': {e}")
        return []


# ─── Memory Engine Singleton ────────────────────────────────────
memory_engine: Optional[AdaptiveMemoryEngine] = None


def init_memory() -> AdaptiveMemoryEngine:
    """Initialize the adaptive memory engine with SQLite persistence."""
    global memory_engine
    memory_engine = AdaptiveMemoryEngine(
        decay_rate=config.decay_rate,
        similarity_threshold=config.similarity_threshold,
        db_path=config.memory_db_path,
    )
    stats = memory_engine.get_stats()
    logger.info(
        f"Memory engine ready — stored memories: {stats['total_memories']}"
    )
    return memory_engine


def close_memory() -> None:
    """Close the memory engine database connection."""
    global memory_engine
    if memory_engine:
        memory_engine.close()
        logger.info("Memory engine closed.")


# ─── Context Enrichment ─────────────────────────────────────────
def enrich_prompt_with_memory(prompt: str) -> tuple[str, list[dict]]:
    """Retrieve relevant memories and prepend them to the prompt as context."""
    if not memory_engine:
        return prompt, []

    query_emb = get_embedding(prompt)
    if not query_emb:
        return prompt, []

    relevant = memory_engine.retrieve(query_emb, top_k=config.top_k_memories)
    if not relevant:
        return prompt, []

    # Build context block from retrieved memories
    context_parts = []
    for mem in relevant:
        context_parts.append(f"- {mem['content']} (score: {mem['final_score']})")

    context_block = "\n\n📋 سياق سابق مرتبط:\n" + "\n".join(context_parts)
    enriched = context_block + "\n\n---\n" + prompt

    logger.info(f"Enriched prompt with {len(relevant)} relevant memories")
    return enriched, relevant


# ─── Core Functions ──────────────────────────────────────────────
def generate_response(prompt: str, system_instruction: Optional[str] = None) -> dict:
    """
    Generate a response from Gemini with memory-augmented context.

    :return: dict with 'response', 'used_memories', and 'saved_to_memory' flags
    """
    if not prompt or not prompt.strip():
        logger.warning("Received empty prompt, returning early.")
        return {"response": "", "used_memories": [], "saved_to_memory": False}

    try:
        # Step 1: Enrich prompt with relevant memories
        enriched_prompt, used_memories = enrich_prompt_with_memory(prompt)

        # Step 2: Generate response
        model = genai.GenerativeModel(
            model_name=config.model_name,
            system_instruction=system_instruction,
        )
        response = model.generate_content(
            enriched_prompt,
            generation_config={
                "max_output_tokens": config.max_tokens,
                "temperature": config.temperature,
            },
        )
        result = response.text.strip()
        logger.info(f"Generated response (length={len(result)} chars)")

        # Step 3: Save prompt + response to memory
        saved = _save_conversation_turn(prompt, result)

        return {
            "response": result,
            "used_memories": used_memories,
            "saved_to_memory": saved,
        }

    except Exception as e:
        logger.error(f"Error generating response for prompt '{prompt[:50]}...': {e}")
        return {
            "response": f"[خطأ في التوليد: {e}]",
            "used_memories": [],
            "saved_to_memory": False,
        }


def _save_conversation_turn(prompt: str, response: str) -> bool:
    """Save a conversation turn to the memory store."""
    if not memory_engine:
        return False

    import uuid

    combined = f"{prompt}\n{response}"
    emb = get_embedding(combined)
    if not emb:
        return False

    mem_id = f"CONV_{uuid.uuid4().hex[:12]}"
    success = memory_engine.add_memory(mem_id, combined, emb)
    if success:
        logger.info(f"Conversation turn saved as {mem_id}")
    return success


def process_batch(prompts: list[str], system_instruction: Optional[str] = None) -> list[dict]:
    """Process a batch of prompts concurrently with memory support."""
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_idx = {
            executor.submit(generate_response, p, system_instruction): i
            for i, p in enumerate(prompts)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                data = future.result()
                results.append({"index": idx, "prompt": prompts[idx], **data})
            except Exception as e:
                logger.error(f"Batch item {idx} failed: {e}")
                results.append({
                    "index": idx,
                    "prompt": prompts[idx],
                    "response": f"[خطأ: {e}]",
                    "used_memories": [],
                    "saved_to_memory": False,
                })

    results.sort(key=lambda x: x["index"])
    logger.info(f"Batch complete — processed {len(results)}/{len(prompts)} items")
    return results


# ─── Main Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    init_client()
    init_memory()

    try:
        sample_prompts = [
            "اكتب مقدمة عن الذكاء الاصطناعي",
            "ما هي فوائد التعلم الآلي؟",
            "شرح مختصر لمفهوم NLP",
        ]

        outputs = process_batch(sample_prompts)
        for item in outputs:
            print(f"\n{'='*60}")
            print(f"Prompt: {item['prompt']}")
            print(f"Response: {item['response']}")
            print(f"Used memories: {len(item['used_memories'])}")
            print(f"Saved to memory: {item['saved_to_memory']}")
    finally:
        close_memory()