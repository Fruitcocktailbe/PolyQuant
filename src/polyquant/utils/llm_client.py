"""
Shared LLM Client for PolyQuant 2.0

Calls Google AI Studio directly via its OpenAI-compatible endpoint. Each agent
passes its own model id (see config.llm_model_*) so per-tier free quotas can
be exploited independently.

USAGE:
------
    from polyquant.utils.llm_client import get_llm_client

    client = get_llm_client()
    response = client.chat.completions.create(
        model="gemini-2.5-flash-lite",
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

# Google AI Studio OpenAI-compatible endpoint
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI | None:
    """
    Get a shared OpenAI-compatible client pointed at Google AI Studio.

    Returns:
        OpenAI client configured for Google's OpenAI-compat endpoint, or None
        if the configured key is missing/invalid.
    """
    try:
        api_key_wrapped = getattr(config, "gemini_api_key", None)
        api_key = api_key_wrapped.get_secret_value() if api_key_wrapped else ""
    except Exception:
        api_key = ""

    if not api_key or len(api_key) < 10 or "your-" in api_key.lower():
        logger.warning("LLM keys not configured — LLM clustering/matching disabled")
        return None

    if api_key.startswith("sk-or-"):
        logger.error(
            "GEMINI_API_KEY looks like an OpenRouter key (sk-or-...). "
            "PolyQuant now calls Google AI Studio directly — generate a new key "
            "at https://aistudio.google.com/apikey (format: AIzaSy...) and put "
            "it in .env as GEMINI_API_KEY. LLM features disabled until fixed."
        )
        return None

    client = OpenAI(
        base_url=GOOGLE_OPENAI_BASE_URL,
        api_key=api_key,
        timeout=45.0,
    )

    logger.info("Google AI Studio LLM client initialized", base_url=GOOGLE_OPENAI_BASE_URL)
    return client


def _extract_json_block(content: str) -> str:
    """Strip common markdown code-fence wrappers around a JSON payload."""
    if not content:
        return ""
    # Prefer explicit json fences
    if "```json" in content:
        try:
            return content.split("```json", 1)[1].split("```", 1)[0].strip()
        except Exception:
            pass
    # Generic fences
    if "```" in content:
        try:
            return content.split("```", 1)[1].split("```", 1)[0].strip()
        except Exception:
            pass
    return content.strip()


def _try_once(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    use_json_mode: bool,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    """
    Single attempt against a specific model.

    Returns (parsed_dict_or_None, status) where status is one of:
        "ok"           — parsed a non-empty dict
        "empty"        — model returned empty/None content
        "parse_fail"   — content present but JSON parse failed
        "empty_parsed" — parsed successfully but result was empty/None/{}
        "rate_limit"   — 429 from upstream (caller should back off)
        "json_mode_unsupported" — model rejected response_format; caller should retry without it
        "error"        — any other exception
    """
    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)

    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            return None, "rate_limit"
        # Some free models reject response_format — detect and signal retry.
        # Match only on the parameter name itself to avoid false positives on
        # generic "not supported" errors unrelated to JSON mode.
        lowered = error_str.lower()
        if use_json_mode and (
            "response_format" in lowered
            or "json_object" in lowered
        ):
            logger.info(
                "Model rejected response_format — retrying without JSON mode",
                model=model,
            )
            return None, "json_mode_unsupported"
        logger.error("LLM call raised", error=error_str, model=model)
        return None, "error"

    try:
        choice = response.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
        actual_model = getattr(response, "model", model)
    except Exception as e:
        logger.error("LLM response had unexpected shape", error=str(e), model=model)
        return None, "error"

    logger.debug(
        "LLM response received",
        model_used=actual_model,
        finish_reason=finish_reason,
        content_len=len(content) if content else 0,
    )

    if not content or not content.strip():
        logger.warning(
            "LLM returned empty content",
            model=actual_model,
            finish_reason=finish_reason,
        )
        print(f"--- ⚠️ LLM EMPTY | model={actual_model} | finish_reason={finish_reason} ---")
        return None, "empty"

    cleaned = _extract_json_block(content)
    try:
        parsed = json.loads(cleaned) if cleaned else None
    except json.JSONDecodeError as e:
        preview = content[:300].replace("\n", " ")
        logger.warning(
            "LLM returned non-JSON content",
            model=actual_model,
            finish_reason=finish_reason,
            parse_error=str(e),
            content_preview=preview,
        )
        print(
            f"--- ⚠️ LLM PARSE FAIL | model={actual_model} | finish_reason={finish_reason}\n"
            f"    preview: {preview}"
        )
        return None, "parse_fail"

    if not parsed:
        preview = content[:300].replace("\n", " ")
        logger.warning(
            "LLM parsed to empty payload",
            model=actual_model,
            finish_reason=finish_reason,
            content_preview=preview,
        )
        print(
            f"--- ⚠️ LLM EMPTY PARSED | model={actual_model} | finish_reason={finish_reason}\n"
            f"    preview: {preview}"
        )
        return None, "empty_parsed"

    return parsed, "ok"


def call_llm_json(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any] | None:
    """
    Call the LLM and parse a JSON response.

    Tries the primary model first. On empty / unparseable / empty-parsed responses
    (but not on network errors), falls back through `config.llm_fallback_models`.
    Retries rate-limit errors with exponential backoff.

    Args:
        prompt: User message content
        system_prompt: System instruction (optional)
        model: Model identifier (defaults to config.llm_model)
        temperature: Sampling temperature (defaults to config.llm_temperature)

    Returns:
        Parsed JSON dict, or None on terminal failure (all models exhausted).
    """
    client = get_llm_client()
    if not client:
        return None

    primary = model or config.llm_model
    temperature = temperature if temperature is not None else config.llm_temperature
    effective_max_tokens = max_tokens if max_tokens is not None else getattr(config, "llm_max_tokens", 4096)

    # Build the model chain: primary first, then fallbacks (deduped, primary excluded)
    fallbacks = list(getattr(config, "llm_fallback_models", []) or [])
    model_chain: list[str] = [primary]
    for m in fallbacks:
        if m and m not in model_chain:
            model_chain.append(m)

    # FIX: Some free models (Gemma via Google AI Studio) reject the 'system' role
    # with "Developer instruction is not enabled". We merge it into the user prompt.
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"[SYSTEM_INSTRUCTION]\n{system_prompt}\n\n[USER_PROMPT]\n{prompt}"

    messages = [{"role": "user", "content": full_prompt}]
    prompt_chars = len(full_prompt)

    max_retries = 3
    base_wait = 10.0  # First 429 wait

    for model_idx, candidate_model in enumerate(model_chain):
        is_fallback = model_idx > 0
        prefix = "🤖 LLM FALLBACK" if is_fallback else "🤖 LLM START"
        print(f"\n--- {prefix}: {candidate_model} | Prompt: {prompt_chars:,} chars ---")
        t0 = time.time()
        use_json_mode = True

        for attempt in range(max_retries):
            parsed, status = _try_once(
                client=client,
                model=candidate_model,
                messages=messages,
                temperature=temperature,
                use_json_mode=use_json_mode,
                max_tokens=effective_max_tokens,
            )

            if status == "ok":
                elapsed = time.time() - t0
                print(f"--- ✅ LLM SUCCESS [{elapsed:.1f}s] | Model: {candidate_model} ---\n")
                return parsed

            if status == "rate_limit":
                if attempt < max_retries - 1:
                    wait_time = base_wait * (2 ** attempt)
                    print(
                        f"--- ⚠️ LLM RATE LIMITED (429) | Waiting {wait_time}s "
                        f"before retry ({attempt+1}/{max_retries}) ---\n"
                    )
                    logger.warning(
                        f"LLM 429 Rate Limit. Backing off for {wait_time}s",
                        attempt=attempt + 1,
                        model=candidate_model,
                    )
                    time.sleep(wait_time)
                    continue
                # Exhausted retries for this model — move on to the next one
                elapsed = time.time() - t0
                print(f"--- ❌ LLM RATE LIMIT EXHAUSTED [{elapsed:.1f}s] | {candidate_model} ---\n")
                break

            if status == "json_mode_unsupported":
                use_json_mode = False
                continue  # retry same model without response_format

            # empty / parse_fail / empty_parsed / error → stop retrying this model,
            # fall through to the next candidate
            elapsed = time.time() - t0
            print(
                f"--- ❌ LLM FAIL [{elapsed:.1f}s] | {candidate_model} | reason={status} ---\n"
            )
            break

    logger.error(
        "All LLM models exhausted",
        models_tried=model_chain,
    )
    return None
