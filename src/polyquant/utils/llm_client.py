"""
Shared LLM Client for PolyQuant 2.0

Uses OpenRouter's OpenAI-compatible API to route requests to free models.
This replaces direct Google Gemini SDK calls across all agents.

USAGE:
------
    from polyquant.utils.llm_client import get_llm_client

    client = get_llm_client()
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

import json
import time
from functools import lru_cache
from typing import Any

from openai import OpenAI

from polyquant.utils import config, get_logger

logger = get_logger(__name__)

# OpenRouter base URL (OpenAI-compatible)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI | None:
    """
    Get a shared OpenAI-compatible client pointed at OpenRouter.

    Returns:
        OpenAI client configured for OpenRouter, or None if no API key.
    """
    # Use config value, fall back to empty if missing
    try:
        api_key_wrapped = getattr(config, "gemini_api_key", None)
        api_key = api_key_wrapped.get_secret_value() if api_key_wrapped else ""
    except Exception:
        api_key = ""

    if not api_key or len(api_key) < 10 or "your-" in api_key.lower():
        logger.warning("LLM keys not configured — LLM clustering/matching disabled")
        return None

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=20.0,  # Fail fast on hung free models so the auto-router can rotate
        default_headers={
            "HTTP-Referer": "https://github.com/polyquant",
            "X-OpenRouter-Title": "PolyQuant",
        },
    )

    logger.info("OpenRouter LLM client initialized", base_url=OPENROUTER_BASE_URL)
    return client


def call_llm_json(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    temperature: float | None = None,
) -> dict[str, Any] | None:
    """
    Call the LLM and parse a JSON response.

    Args:
        prompt: User message content
        system_prompt: System instruction (optional)
        model: Model identifier (defaults to config.llm_model)
        temperature: Sampling temperature (defaults to config.llm_temperature)

    Returns:
        Parsed JSON dict, or None on failure
    """
    client = get_llm_client()
    if not client:
        return None

    # Use configuration defaults if not provided
    model = model or config.llm_model
    temperature = temperature if temperature is not None else config.llm_temperature

    messages = []
    # FIX: Some free models (Gemma via Google AI Studio) reject the 'system' role
    # with "Developer instruction is not enabled". We merge it into the user prompt.
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"[SYSTEM_INSTRUCTION]\n{system_prompt}\n\n[USER_PROMPT]\n{prompt}"
    
    messages.append({"role": "user", "content": full_prompt})

    prompt_chars = len(full_prompt)
    print(f"\n--- 🤖 LLM START: {model} | Prompt: {prompt_chars:,} chars ---")
    t0 = time.time()

    max_retries = 3
    base_wait = 10.0 # Start with 10 seconds wait on first 429

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )

            elapsed = time.time() - t0
            content = response.choices[0].message.content
            actual_model = getattr(response, "model", model)
            logger.debug("LLM response received", model_used=actual_model)

            # Parse JSON — handle markdown code blocks if present
            if content and "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif content and "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content.strip()) if content else None
            print(f"--- ✅ LLM SUCCESS [{elapsed:.1f}s] | Model: {actual_model} ---\n")
            return parsed

        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            is_rate_limit = "429" in error_str
            is_timeout = "timeout" in error_lower or "timed out" in error_lower

            if (is_rate_limit or is_timeout) and attempt < max_retries - 1:
                if is_rate_limit:
                    wait_time = base_wait * (2 ** attempt)  # 10s, 20s...
                    print(f"--- ⚠️ LLM RATE LIMITED (429) | Waiting {wait_time}s before retry ({attempt+1}/{max_retries}) ---\n")
                    logger.warning(f"LLM 429 Rate Limit. Backing off for {wait_time}s", attempt=attempt+1)
                else:
                    wait_time = 2.0
                    print(f"--- ⏱️ LLM TIMEOUT | Retrying ({attempt+1}/{max_retries}) to rotate auto-router ---\n")
                    logger.warning(f"LLM timeout on {model}. Retrying for fresh routing", attempt=attempt+1)
                time.sleep(wait_time)
                continue

            elapsed = time.time() - t0
            print(f"--- ❌ LLM FAILED [{elapsed:.1f}s] | Error: {e} ---\n")
            logger.error("LLM call failed", error=error_str, model=model)
            return None
