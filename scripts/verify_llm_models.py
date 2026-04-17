"""
LLM Model Verification Script for PolyQuant.

Pings every model in `config.llm_fallback_models` (plus the primary from
`config.llm_model`) with a one-token prompt and reports live status. Use
this before a map-maker session to confirm which models in your chain
actually work — OpenRouter's :free catalog rotates frequently and stale
ids are invisible on their dashboard.

USAGE:
------
    python scripts/verify_llm_models.py

EXIT CODES:
-----------
    0 — at least one model in the chain is live
    1 — every model in the chain is DEAD / NO-KEY / rate-limited

Output columns:
    STATUS     — OK / DEAD-404 / RATE-LIMITED / NO-KEY / ERROR
    MODEL      — the id as configured
    PROVIDER   — Google (AI Studio direct) / OpenRouter
    LATENCY    — round-trip ms on success (blank otherwise)
    DETAIL     — error excerpt or 'reply: ...' for successes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src to path for imports — mirrors the other scripts in this directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polyquant.utils import config, get_logger  # noqa: E402
from polyquant.utils.llm_client import (  # noqa: E402
    _client_for_model,
    _is_openrouter_model,
)

logger = get_logger(__name__)


def _provider_label(model: str) -> str:
    return "OpenRouter" if _is_openrouter_model(model) else "Google"


def _probe_model(model: str) -> tuple[str, str, float | None, str]:
    """Return (status, provider, latency_ms_or_None, detail)."""
    provider = _provider_label(model)
    client = _client_for_model(model)
    if client is None:
        return ("NO-KEY", provider, None, f"set {'OPENROUTER_API_KEY' if provider == 'OpenRouter' else 'GEMINI_API_KEY'} in .env")

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with just: ok"}],
            temperature=0.0,
            max_tokens=4,
        )
    except Exception as e:  # noqa: BLE001 — probe should never re-raise
        elapsed_ms = (time.time() - t0) * 1000
        msg = str(e)
        lowered = msg.lower()
        if "429" in msg:
            return ("RATE-LIMITED", provider, elapsed_ms, msg[:120])
        if (
            "404" in msg
            or "no endpoints" in lowered
            or "invalid model" in lowered
            or "model not found" in lowered
        ):
            return ("DEAD-404", provider, elapsed_ms, msg[:120])
        return ("ERROR", provider, elapsed_ms, msg[:120])

    elapsed_ms = (time.time() - t0) * 1000
    try:
        content = response.choices[0].message.content or ""
    except Exception:  # noqa: BLE001
        content = "(no content)"
    return ("OK", provider, elapsed_ms, f"reply: {content.strip()[:60]!r}")


def _chain_from_config() -> list[str]:
    """Build the same chain call_llm_json builds: primary + fallbacks, deduped."""
    primary = config.llm_model
    fallbacks = list(getattr(config, "llm_fallback_models", []) or [])
    chain: list[str] = [primary]
    for m in fallbacks:
        if m and m not in chain:
            chain.append(m)
    return chain


def main() -> int:
    chain = _chain_from_config()
    print(f"\nProbing {len(chain)} models from config.llm_fallback_models chain\n")
    print(f"{'STATUS':<14} {'PROVIDER':<11} {'LATENCY':>10}  MODEL")
    print("-" * 80)

    any_ok = False
    rows: list[tuple[str, str, float | None, str, str]] = []
    for model in chain:
        status, provider, latency_ms, detail = _probe_model(model)
        rows.append((status, provider, latency_ms, model, detail))
        latency_str = f"{latency_ms:>6.0f} ms" if latency_ms is not None else "      —"
        print(f"{status:<14} {provider:<11} {latency_str:>10}  {model}")
        if status == "OK":
            any_ok = True

    print("-" * 80)
    print("\nDetails:")
    for status, _provider, _latency, model, detail in rows:
        if status == "OK":
            print(f"  ✅ {model}: {detail}")
        elif status == "NO-KEY":
            print(f"  ⏭  {model}: {detail}")
        elif status == "DEAD-404":
            print(f"  🚫 {model}: {detail}")
        elif status == "RATE-LIMITED":
            print(f"  ⚠  {model}: {detail}")
        else:
            print(f"  ❌ {model}: {detail}")

    # Surface a concrete recommendation when the chain is dry.
    if not any_ok:
        dead_only = all(r[0] in ("DEAD-404", "NO-KEY") for r in rows)
        all_rate_limited = any(r[0] == "RATE-LIMITED" for r in rows) and not any(
            r[0] == "OK" for r in rows
        )
        print("\n⚠️  No model in the chain is currently usable.\n")
        if dead_only:
            print(
                "Every configured model returned 404 or has no API key. The ids in\n"
                "config.llm_fallback_models are likely deprecated. Pick fresh ones from\n"
                "https://openrouter.ai/models (filter ':free') and/or set GEMINI_API_KEY."
            )
        elif all_rate_limited:
            print(
                "Every live model is rate-limited. Options:\n"
                "  - wait ~midnight Pacific for Google's daily quota reset\n"
                "  - top up openrouter.ai/credits to $10 (lifts 50 → 1000 req/day)\n"
                "  - add paid OpenRouter models (google/gemini-2.5-flash-lite,\n"
                "    openai/gpt-4.1-nano, deepseek/deepseek-v3.1) — ~$0.07/map run"
            )
        return 1

    live = sum(1 for r in rows if r[0] == "OK")
    print(f"\n✅ {live}/{len(rows)} models in the chain are live. Map maker is safe to run.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
