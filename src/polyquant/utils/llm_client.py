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
from functools import lru_cache
from typing import Any

from openai import OpenAI

from polyquant.utils import config, get_logger

logger = get_logger(__name__)

# OpenRouter base URL (OpenAI-compatible)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model: free router picks from available free models
DEFAULT_MODEL = "openrouter/free"


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI | None:
    """
    Get a shared OpenAI-compatible client pointed at OpenRouter.

    Returns:
        OpenAI client configured for OpenRouter, or None if no API key.
    """
    api_key = config.gemini_api_key.get_secret_value()

    if not api_key or "your-" in api_key:
        logger.warning("OPENROUTER_API_KEY / GEMINI_API_KEY not set — LLM features disabled")
        return None

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
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
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
) -> dict[str, Any] | None:
    """
    Call the LLM and parse a JSON response.

    Args:
        prompt: User message content
        system_prompt: System instruction (optional)
        model: Model identifier (default: openrouter/free)
        temperature: Sampling temperature

    Returns:
        Parsed JSON dict, or None on failure
    """
    client = get_llm_client()
    if not client:
        return None

    if temperature is None:
        temperature = config.llm_temperature

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        actual_model = getattr(response, "model", model)
        logger.debug("LLM response received", model_used=actual_model)

        # Parse JSON — handle markdown code blocks if present
        if content and "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif content and "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip()) if content else None

    except Exception as e:
        logger.error("LLM call failed", error=str(e), model=model)
        return None
