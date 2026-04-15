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

import html
import json
import threading
import time
from functools import lru_cache
from typing import Any

from openai import OpenAI

from polyquant.utils import config, get_logger

logger = get_logger(__name__)

# Google AI Studio OpenAI-compatible endpoint
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Google AI Studio free-tier request-per-minute limits. Source:
# https://ai.google.dev/gemini-api/docs/rate-limits
# We pace calls at (60 / RPM) * safety_margin to stay below the ceiling.
_MODEL_RPM: dict[str, int] = {
    "gemini-2.5-pro": 5,
    "gemini-2.5-flash": 10,
    "gemini-2.5-flash-lite": 15,
}
_DEFAULT_RPM = 5  # Conservative fallback for unknown models
_RPM_SAFETY_MARGIN = 1.15  # Pad the interval by 15% to account for clock skew / request overhead
_DEFAULT_RETRY_AFTER_S = 60.0  # If 429 lacks Retry-After, wait one full minute
_MAX_BLOCK_SKIP_S = 120.0  # Skip a model in the fallback chain if it's blocked this long or more


class _RateLimiter:
    """
    Thread-safe per-model token bucket.

    Enforces a minimum interval between successive calls to each model based on
    its free-tier RPM, and records 429 back-off windows so concurrent callers
    share the same block. `wait_if_needed` sleeps before the call (proactive
    pacing); `record_429` sets a hard block consumed by subsequent callers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_call: dict[str, float] = {}
        self._blocked_until: dict[str, float] = {}

    def _min_interval(self, model: str) -> float:
        rpm = _MODEL_RPM.get(model, _DEFAULT_RPM)
        return (60.0 / rpm) * _RPM_SAFETY_MARGIN

    def wait_if_needed(self, model: str) -> None:
        """Block until the next call to `model` is allowed under RPM + 429 state."""
        while True:
            with self._lock:
                now = time.time()
                blocked = self._blocked_until.get(model, 0.0)
                last = self._last_call.get(model, 0.0)
                next_allowed = max(blocked, last + self._min_interval(model))
                wait = next_allowed - now
                if wait <= 0:
                    # Reserve this slot immediately so concurrent waiters space out.
                    self._last_call[model] = now
                    return
            # Sleep outside the lock so other threads can compute their own waits.
            logger.debug(f"Rate limiter: waiting {wait:.1f}s for {model}")
            time.sleep(wait)

    def record_429(self, model: str, retry_after_s: float) -> None:
        with self._lock:
            self._blocked_until[model] = max(
                self._blocked_until.get(model, 0.0),
                time.time() + retry_after_s,
            )

    def blocked_for(self, model: str) -> float:
        """Seconds until `model` becomes available again (0 if free)."""
        with self._lock:
            blocked = self._blocked_until.get(model, 0.0)
            return max(0.0, blocked - time.time())


_rate_limiter = _RateLimiter()


def _parse_retry_after(exc: Exception) -> float:
    """
    Extract a retry-after duration from a Google AI Studio / OpenAI 429 error.

    Checks in order: the `Retry-After` header on the response, Google's
    `retryDelay` field inside the error body details, and finally a sensible
    default.
    """
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                if retry_after:
                    return float(retry_after)
    except Exception:
        pass

    try:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            details = body.get("error", {}).get("details", []) or []
            for detail in details:
                if isinstance(detail, dict) and "retryDelay" in detail:
                    # Format is e.g. "30s"
                    delay_str = str(detail["retryDelay"]).rstrip("s")
                    return float(delay_str)
    except Exception:
        pass

    return _DEFAULT_RETRY_AFTER_S


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
    """
    Extract a JSON payload from an LLM response.

    Tries, in order: explicit ```json fence, generic ``` fence, the substring
    from the first `{` to the last `}` (or `[`/`]` for array roots), and finally
    the stripped content as-is. Also unescapes HTML entities defensively in case
    Gemini returns escaped output.
    """
    if not content:
        return ""

    content = html.unescape(content)

    if "```json" in content:
        try:
            return content.split("```json", 1)[1].split("```", 1)[0].strip()
        except Exception:
            pass

    if "```" in content:
        try:
            return content.split("```", 1)[1].split("```", 1)[0].strip()
        except Exception:
            pass

    first_obj = content.find("{")
    last_obj = content.rfind("}")
    first_arr = content.find("[")
    last_arr = content.rfind("]")

    obj_valid = first_obj != -1 and last_obj > first_obj
    arr_valid = first_arr != -1 and last_arr > first_arr

    if obj_valid and arr_valid:
        if first_obj <= first_arr:
            return content[first_obj : last_obj + 1].strip()
        return content[first_arr : last_arr + 1].strip()
    if obj_valid:
        return content[first_obj : last_obj + 1].strip()
    if arr_valid:
        return content[first_arr : last_arr + 1].strip()

    return content.strip()


def _try_once(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float,
    use_json_mode: bool,
) -> tuple[dict[str, Any] | None, str]:
    """
    Single attempt against a specific model.

    Returns (parsed_dict_or_None, status) where status is one of:
        "ok"           — parsed a non-empty dict
        "empty"        — model returned empty/None content
        "parse_fail"   — content present but JSON parse failed
        "empty_parsed" — parsed successfully but result was empty/None/{}
        "truncated"    — finish_reason="length"; response cut off, do not parse
        "rate_limit"   — 429 from upstream (caller should back off)
        "json_mode_unsupported" — model rejected response_format; caller should retry without it
        "error"        — any other exception
    """
    # Proactive per-model pacing: sleep until we're inside the RPM budget.
    _rate_limiter.wait_if_needed(model)

    try:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)

    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            retry_after = _parse_retry_after(e)
            _rate_limiter.record_429(model, retry_after)
            logger.warning(
                "LLM 429 received — model blocked",
                model=model,
                retry_after_s=retry_after,
            )
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

    if finish_reason == "length":
        logger.warning(
            "LLM response truncated (finish_reason=length)",
            model=actual_model,
            content_len=len(content),
        )
        print(
            f"--- ⚠️ LLM TRUNCATED | model={actual_model} | content_len={len(content)} ---"
        )
        return None, "truncated"

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

    # Build the model chain: primary first, then fallbacks (deduped, primary excluded)
    fallbacks = list(getattr(config, "llm_fallback_models", []) or [])
    model_chain: list[str] = [primary]
    for m in fallbacks:
        if m and m not in model_chain:
            model_chain.append(m)

    # Some free models (Gemma via Google AI Studio) reject the 'system' role
    # with "Developer instruction is not enabled". Merge it into the user prompt.
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"[SYSTEM_INSTRUCTION]\n{system_prompt}\n\n[USER_PROMPT]\n{prompt}"

    # Google AI Studio's response_format={"type":"json_object"} requires the
    # literal word "json" somewhere in the prompt or it silently degrades.
    if "json" not in full_prompt.lower():
        full_prompt = f"{full_prompt}\n\nRespond with a single JSON object."

    messages = [{"role": "user", "content": full_prompt}]
    prompt_chars = len(full_prompt)

    max_retries = 2  # Per-model retries for rate-limit / json-mode quirks
    last_failure_was_rate_limit = False

    for model_idx, candidate_model in enumerate(model_chain):
        is_fallback = model_idx > 0

        # Skip any model that's 429-blocked for longer than we're willing to wait
        # on a sibling. Sibling models share the same account quota, so there's
        # rarely any point in cross-falling-back when the block is short.
        block_s = _rate_limiter.blocked_for(candidate_model)
        if is_fallback and block_s > _MAX_BLOCK_SKIP_S:
            print(
                f"--- ⏭️  LLM SKIP: {candidate_model} | blocked for {block_s:.0f}s ---"
            )
            logger.info(
                "Skipping fallback model — still rate-limited",
                model=candidate_model,
                block_s=block_s,
            )
            continue

        prefix = "🤖 LLM FALLBACK" if is_fallback else "🤖 LLM START"
        print(f"\n--- {prefix}: {candidate_model} | Prompt: {prompt_chars:,} chars ---")
        t0 = time.time()
        use_json_mode = True
        last_failure_was_rate_limit = False

        for attempt in range(max_retries):
            parsed, status = _try_once(
                client=client,
                model=candidate_model,
                messages=messages,
                temperature=temperature,
                use_json_mode=use_json_mode,
            )

            if status == "ok":
                elapsed = time.time() - t0
                print(f"--- ✅ LLM SUCCESS [{elapsed:.1f}s] | Model: {candidate_model} ---\n")
                return parsed

            if status == "rate_limit":
                last_failure_was_rate_limit = True
                block_s = _rate_limiter.blocked_for(candidate_model)
                # Only retry the same model if the block will clear quickly.
                # Otherwise drop through: siblings are likely blocked too, but
                # we still give them one shot further down the chain.
                if attempt < max_retries - 1 and 0 < block_s <= _MAX_BLOCK_SKIP_S:
                    print(
                        f"--- ⚠️  LLM RATE LIMITED (429) | {candidate_model} | "
                        f"waiting {block_s:.0f}s (retry {attempt + 1}/{max_retries}) ---\n"
                    )
                    time.sleep(block_s)
                    continue
                elapsed = time.time() - t0
                print(
                    f"--- ❌ LLM RATE LIMITED [{elapsed:.1f}s] | {candidate_model} | "
                    f"block={block_s:.0f}s ---\n"
                )
                break

            if status == "json_mode_unsupported":
                use_json_mode = False
                continue  # retry same model without response_format

            # empty / parse_fail / empty_parsed / truncated / error → stop retrying
            # this model, fall through to the next candidate
            elapsed = time.time() - t0
            print(
                f"--- ❌ LLM FAIL [{elapsed:.1f}s] | {candidate_model} | reason={status} ---\n"
            )
            break

    logger.error(
        "All LLM models exhausted",
        models_tried=model_chain,
        last_failure_was_rate_limit=last_failure_was_rate_limit,
    )
    return None
