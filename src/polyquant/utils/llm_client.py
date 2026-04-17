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

# OpenRouter — a proxy that exposes many model providers (DeepSeek, Meta,
# Anthropic, Google, Qwen, Mistral) through one OpenAI-compatible endpoint.
# We use it as a second-provider safety net: when Google AI Studio's daily
# quota runs out, OpenRouter's free-tier models (identified by a "/" in the
# id and a ":free" suffix) keep the pipeline alive on an independent account.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-account pooled RPM for OpenRouter free-tier models. OpenRouter enforces
# limits at the account level, not per-model — rotating between free models
# doesn't multiply the budget, it just distributes load across providers for
# latency/reliability. 20 RPM is OpenRouter's documented free-tier ceiling.
_OPENROUTER_FREE_POOL_KEY = "__openrouter_free__"
_OPENROUTER_FREE_POOL_RPM = 20

# Google AI Studio free-tier request-per-minute limits. Source:
# https://ai.google.dev/gemini-api/docs/rate-limits
# We pace calls at (60 / RPM) * safety_margin to stay below the ceiling.
_MODEL_RPM: dict[str, int] = {
    "gemini-2.5-pro": 5,
    "gemini-2.5-flash": 10,
    "gemini-2.5-flash-lite": 15,
    # OpenRouter pool gets rate-limited as a single bucket via the key above.
    _OPENROUTER_FREE_POOL_KEY: _OPENROUTER_FREE_POOL_RPM,
}
_DEFAULT_RPM = 5  # Conservative fallback for unknown models
_RPM_SAFETY_MARGIN = 1.15  # Pad the interval by 15% to account for clock skew / request overhead
_DEFAULT_RETRY_AFTER_S = 60.0  # If 429 lacks Retry-After, wait one full minute
# Any block longer than this makes the chain jump to the next unblocked
# fallback (when one exists). Kept small — even 10s of Gemini sleep wastes
# meaningful throughput when 5 OpenRouter fallbacks are idle and ready.
_MAX_BLOCK_SKIP_S = 5.0


def _is_openrouter_model(model: str) -> bool:
    """OpenRouter model ids always contain a '/' (e.g. 'deepseek/deepseek-chat:free').
    Google AI Studio model ids never do. This is the single source of truth for
    provider routing."""
    return "/" in model


def _rate_limit_key(model: str) -> str:
    """Key used by the rate limiter. All OpenRouter free-tier models share a
    single pooled bucket because OpenRouter enforces limits at the account
    level. The pool covers:
      - Individual free models (id ends with ":free", e.g.
        "meta-llama/llama-3.3-70b-instruct:free")
      - The `openrouter/free` meta-router, which internally dispatches to a
        free model and therefore bills against the same free-tier quota
    """
    if _is_openrouter_model(model) and (
        model.endswith(":free") or model == "openrouter/free"
    ):
        return _OPENROUTER_FREE_POOL_KEY
    return model


class _RateLimiter:
    """
    Thread-safe per-model token bucket.

    Enforces a minimum interval between successive calls to each model based on
    its free-tier RPM, and records 429 back-off windows so concurrent callers
    share the same block. `wait_if_needed` sleeps before the call (proactive
    pacing); `record_429` sets a hard block consumed by subsequent callers.

    Models are keyed by `_rate_limit_key(model)` so OpenRouter free-tier
    models share a single pool (matching OpenRouter's account-level billing)
    while each Google model has its own bucket (Google meters per-model).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_call: dict[str, float] = {}
        self._blocked_until: dict[str, float] = {}

    def _min_interval(self, key: str) -> float:
        rpm = _MODEL_RPM.get(key, _DEFAULT_RPM)
        return (60.0 / rpm) * _RPM_SAFETY_MARGIN

    def wait_if_needed(self, model: str) -> None:
        """Block until the next call to `model` is allowed under RPM + 429 state."""
        key = _rate_limit_key(model)
        while True:
            with self._lock:
                now = time.time()
                blocked = self._blocked_until.get(key, 0.0)
                last = self._last_call.get(key, 0.0)
                next_allowed = max(blocked, last + self._min_interval(key))
                wait = next_allowed - now
                if wait <= 0:
                    # Reserve this slot immediately so concurrent waiters space out.
                    self._last_call[key] = now
                    return
            # Sleep outside the lock so other threads can compute their own waits.
            logger.debug(f"Rate limiter: waiting {wait:.1f}s for {model} (key={key})")
            time.sleep(wait)

    def record_429(self, model: str, retry_after_s: float) -> None:
        key = _rate_limit_key(model)
        with self._lock:
            self._blocked_until[key] = max(
                self._blocked_until.get(key, 0.0),
                time.time() + retry_after_s,
            )

    def blocked_for(self, model: str) -> float:
        """Seconds until `model` becomes available again (0 if free)."""
        key = _rate_limit_key(model)
        with self._lock:
            blocked = self._blocked_until.get(key, 0.0)
            return max(0.0, blocked - time.time())


_rate_limiter = _RateLimiter()


# Models that returned a permanent error (typically 404 "no endpoints found"
# on OpenRouter when we pin to a stale/deprecated id). Populated at runtime
# inside _try_once; read by call_llm_json when building the per-call chain.
# Module-global so the state persists across calls in the same process.
_permanently_dead_models: set[str] = set()
_dead_models_lock = threading.Lock()


def _mark_model_dead(model: str, reason: str) -> None:
    with _dead_models_lock:
        if model not in _permanently_dead_models:
            _permanently_dead_models.add(model)
            logger.warning(
                "Marking model permanently dead for the rest of this run",
                model=model,
                reason=reason,
                hint="Fix the id in config.llm_fallback_models "
                "(check https://openrouter.ai/models for current :free ids).",
            )


def _is_model_dead(model: str) -> bool:
    with _dead_models_lock:
        return model in _permanently_dead_models


def _parse_retry_after(exc: Exception) -> float:
    """
    Extract a retry-after duration from a 429 error.

    Checks in order:
    1. `Retry-After` header (standard — set by both Google and OpenRouter).
    2. `X-RateLimit-Reset` header (OpenRouter only — unix epoch ms timestamp
       when the bucket refills; we convert to a relative delay).
    3. Google's `retryDelay` field inside the error body details.
    4. Sensible default.
    """
    try:
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None)
            if headers:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                if retry_after:
                    return float(retry_after)
                # OpenRouter ships X-RateLimit-Reset as a future unix-ms timestamp.
                reset = (
                    headers.get("x-ratelimit-reset")
                    or headers.get("X-RateLimit-Reset")
                )
                if reset:
                    try:
                        reset_ms = float(reset)
                        delta = (reset_ms / 1000.0) - time.time()
                        if delta > 0:
                            return min(delta, 300.0)  # cap at 5 min
                    except (TypeError, ValueError):
                        pass
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


def _read_api_key(attr: str) -> str:
    """Unwrap a SecretStr config field; return empty string when missing or
    obviously a placeholder."""
    try:
        wrapped = getattr(config, attr, None)
        key = wrapped.get_secret_value() if wrapped else ""
    except Exception:
        key = ""
    if not key or len(key) < 10 or "your-" in key.lower():
        return ""
    return key


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI | None:
    """
    Get a shared OpenAI-compatible client pointed at Google AI Studio.

    Returns:
        OpenAI client configured for Google's OpenAI-compat endpoint, or None
        if the configured key is missing/invalid. OpenRouter is handled by a
        separate lazy client factory (`_get_openrouter_client`).
    """
    api_key = _read_api_key("gemini_api_key")
    if not api_key:
        logger.warning("Google AI Studio key not configured — LLM clustering/matching disabled")
        return None

    if api_key.startswith("sk-or-"):
        logger.error(
            "GEMINI_API_KEY looks like an OpenRouter key (sk-or-...). "
            "Google and OpenRouter keys are now handled as separate fields — "
            "put your Google AI Studio key (format: AIzaSy...) in GEMINI_API_KEY "
            "and your OpenRouter key (sk-or-v1-...) in OPENROUTER_API_KEY."
        )
        return None

    client = OpenAI(
        base_url=GOOGLE_OPENAI_BASE_URL,
        api_key=api_key,
        timeout=45.0,
    )

    logger.info("Google AI Studio LLM client initialized", base_url=GOOGLE_OPENAI_BASE_URL)
    return client


@lru_cache(maxsize=1)
def _get_openrouter_client() -> OpenAI | None:
    """Lazy singleton for the OpenRouter client. Returns None when no
    openrouter_api_key is set (silently — OpenRouter is an optional
    safety-net provider, not a hard dependency)."""
    api_key = _read_api_key("openrouter_api_key")
    if not api_key:
        return None

    # OpenRouter recommends identifying the caller via HTTP-Referer/X-Title
    # headers. These are public app identifiers, not secrets, and help
    # OpenRouter surface per-app analytics in their dashboard.
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=45.0,
        default_headers={
            "HTTP-Referer": "https://github.com/polyquant",
            "X-Title": "PolyQuant MapMaker",
        },
    )
    logger.info("OpenRouter LLM client initialized", base_url=OPENROUTER_BASE_URL)
    return client


def _client_for_model(model: str) -> OpenAI | None:
    """Route to the right provider client based on the model id format.

    Returns None when the required API key isn't configured — callers treat
    that as a silent skip (the model is removed from the fallback chain)."""
    if _is_openrouter_model(model):
        return _get_openrouter_client()
    return get_llm_client()


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
        "model_dead"   — 404 / "no endpoints" / invalid-model — permanently unavailable this run
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
        # Permanent model errors — 404 / "no endpoints found" / "invalid model".
        # These typically mean the id in our fallback list is deprecated or
        # wrong; retrying will never succeed. Mark the model dead so future
        # call chains skip it immediately.
        lowered = error_str.lower()
        if (
            "404" in error_str
            or "no endpoints" in lowered
            or "invalid model" in lowered
            or "model not found" in lowered
            or "model_not_found" in lowered
        ):
            _mark_model_dead(model, reason=error_str[:200])
            logger.error(
                "LLM model unavailable (permanent) — skipping for rest of run",
                error=error_str,
                model=model,
            )
            return None, "model_dead"
        # Some free models reject response_format — detect and signal retry.
        # Match only on the parameter name itself to avoid false positives on
        # generic "not supported" errors unrelated to JSON mode.
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
    # Note: we resolve a client per-model inside the loop below because the
    # chain can mix Google and OpenRouter models. When neither provider has a
    # usable key, the chain is empty and we bail out early.
    primary = model or config.llm_model
    temperature = temperature if temperature is not None else config.llm_temperature

    # Build the model chain: primary first, then fallbacks (deduped, primary excluded)
    fallbacks = list(getattr(config, "llm_fallback_models", []) or [])
    raw_chain: list[str] = [primary]
    for m in fallbacks:
        if m and m not in raw_chain:
            raw_chain.append(m)

    # Filter out any model whose provider isn't configured. OpenRouter models
    # are silently dropped when no OPENROUTER_API_KEY is set — the user can
    # add the key later without any other changes. Also drop any model that
    # was marked permanently dead earlier in the run (stale id / 404 /
    # "no endpoints found") so we never waste another round trip on it.
    model_chain: list[str] = []
    dropped_no_provider: list[str] = []
    dropped_dead: list[str] = []
    for candidate in raw_chain:
        if _is_model_dead(candidate):
            dropped_dead.append(candidate)
            continue
        if _client_for_model(candidate) is not None:
            model_chain.append(candidate)
        else:
            dropped_no_provider.append(candidate)
    if dropped_no_provider:
        logger.debug(
            "Skipped models with no configured provider key",
            skipped=dropped_no_provider,
        )
    if dropped_dead:
        logger.debug(
            "Skipped models marked permanently dead this run",
            skipped=dropped_dead,
        )
    if not model_chain:
        logger.warning(
            "No LLM provider configured — LLM clustering/matching disabled. "
            "Set GEMINI_API_KEY and/or OPENROUTER_API_KEY."
        )
        return None

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

    # Per-model retries are only useful for `json_mode_unsupported` (retry
    # without response_format) — 429 drops immediately to the next model now
    # because the old "sleep 60s mid-chain then retry" behaviour was wasting
    # minutes per call when multiple fallbacks were ready.
    max_retries = 2
    last_failure_was_rate_limit = False

    for model_idx, candidate_model in enumerate(model_chain):
        is_fallback = model_idx > 0

        # Skip any currently-blocked model when a later fallback is ready. This
        # matters for the primary too, not just fallbacks — if Google's daily
        # quota is exhausted (blocked 60s+) and OpenRouter is idle, we should
        # jump straight to OpenRouter instead of sleeping for 60s of pacing.
        block_s = _rate_limiter.blocked_for(candidate_model)
        if block_s > _MAX_BLOCK_SKIP_S:
            later_ready = any(
                _rate_limiter.blocked_for(later) <= _MAX_BLOCK_SKIP_S
                for later in model_chain[model_idx + 1 :]
            )
            if later_ready:
                print(
                    f"--- ⏭️  LLM SKIP: {candidate_model} | blocked for "
                    f"{block_s:.0f}s; jumping to unblocked fallback ---"
                )
                logger.info(
                    "Skipping blocked model — fallback available",
                    model=candidate_model,
                    block_s=block_s,
                )
                continue

        # Resolve the right client for this model — Google AI Studio or
        # OpenRouter — from the shared cached factories. The chain-building
        # step above already dropped models with no configured provider, so
        # this lookup should always succeed; defensive None-check anyway.
        candidate_client = _client_for_model(candidate_model)
        if candidate_client is None:
            logger.debug(
                "Candidate model has no provider client at call time — skipping",
                model=candidate_model,
            )
            continue

        prefix = "🤖 LLM FALLBACK" if is_fallback else "🤖 LLM START"
        print(f"\n--- {prefix}: {candidate_model} | Prompt: {prompt_chars:,} chars ---")
        t0 = time.time()
        use_json_mode = True

        for attempt in range(max_retries):
            parsed, status = _try_once(
                client=candidate_client,
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
                elapsed = time.time() - t0
                # Don't waste 60s sleeping on the SAME model when 5 fallbacks
                # are waiting — jump straight to the next candidate. The block
                # persists in the rate-limiter so the chain-level skip above
                # will automatically route around this model on the next call.
                print(
                    f"--- ❌ LLM RATE LIMITED [{elapsed:.1f}s] | {candidate_model} | "
                    f"block={block_s:.0f}s — trying next fallback ---\n"
                )
                break

            if status == "model_dead":
                # Model is permanently unavailable (404 / stale id). Dead-list
                # is now populated; future calls will skip it at chain build.
                elapsed = time.time() - t0
                print(
                    f"--- 🚫 LLM DEAD [{elapsed:.1f}s] | {candidate_model} | "
                    f"removed from fallback chain for this run ---\n"
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

    if last_failure_was_rate_limit:
        # Every provider returned 429. Most likely the user has burned the
        # free-tier daily cap on both Google AI Studio and OpenRouter. Give a
        # concrete next step instead of just "exhausted".
        logger.error(
            "All LLM providers rate-limited. Likely daily quota exhausted on "
            "Google AI Studio AND OpenRouter. Options: (a) wait for quota "
            "reset (~midnight Pacific for Google), (b) deposit $10 one-time "
            "at openrouter.ai/credits to lift the 50/day cap to 1000/day, "
            "(c) add more providers (Groq, Cerebras) to llm_fallback_models.",
            models_tried=model_chain,
        )
    else:
        logger.error(
            "All LLM models exhausted",
            models_tried=model_chain,
            last_failure_was_rate_limit=last_failure_was_rate_limit,
        )
    return None
