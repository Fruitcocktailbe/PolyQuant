"""Tests for the LLM-client fallback-chain behaviour.

Before the map-maker could reliably use OpenRouter as a safety net, the
chain spent ~2 minutes per call sleeping inside Gemini's 60s rate-limit
windows before even touching a fallback. These tests lock in the new rules:

- 429 on a model immediately moves to the next fallback (no same-model sleep).
- 404 / "no endpoints found" marks a model permanently dead for the run; it's
  filtered out of future chains without another round trip.
- A blocked primary yields to any unblocked later fallback instead of waiting.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from polyquant.utils import llm_client as llm


@pytest.fixture(autouse=True)
def _reset_llm_state(monkeypatch):
    """Each test starts with an empty dead-list and a fresh rate limiter so
    earlier tests can't bleed block state into later ones."""
    llm._permanently_dead_models.clear()
    monkeypatch.setattr(llm, "_rate_limiter", llm._RateLimiter())
    yield
    llm._permanently_dead_models.clear()


def _fake_response(content: str) -> SimpleNamespace:
    """Minimal duck-type matching the OpenAI SDK's ChatCompletion response."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="fake-model")


class _ScriptedClient:
    """Fake OpenAI client whose `chat.completions.create` returns (or
    raises) whatever the scripted list dictates on each call. Lets us
    exercise the fallback chain deterministically."""

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._script:
            raise RuntimeError("script exhausted")
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _make_exc(message: str) -> Exception:
    """Build an exception whose str() contains the message so
    `_try_once`'s error classifier can match substrings on it."""
    return Exception(message)


# --------------------------------------------------------- 404 dead-listing


def test_404_no_endpoints_marks_model_dead_for_session(monkeypatch):
    monkeypatch.setattr(llm, "get_llm_client", lambda: _ScriptedClient([]))
    monkeypatch.setattr(llm, "_get_openrouter_client", lambda: None)

    fake_client = _ScriptedClient(
        [_make_exc("Error code: 404 - {'error': 'No endpoints found for deepseek/stale:free'}")]
    )

    parsed, status = llm._try_once(
        client=fake_client,
        model="deepseek/stale:free",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        use_json_mode=False,
    )
    assert parsed is None
    assert status == "model_dead"
    assert llm._is_model_dead("deepseek/stale:free")


def test_dead_model_is_filtered_from_chain(monkeypatch):
    llm._permanently_dead_models.add("dead/model:free")
    monkeypatch.setattr(llm, "get_llm_client", lambda: object())
    monkeypatch.setattr(llm, "_get_openrouter_client", lambda: object())

    success_client = _ScriptedClient([_fake_response('{"ok": true}')])
    monkeypatch.setattr(
        llm, "_client_for_model",
        lambda m: success_client if m == "gemini-2.5-flash" else object(),
    )

    # Also bypass the per-model rate limiter delay so the test runs fast.
    monkeypatch.setattr(llm._rate_limiter, "wait_if_needed", lambda m: None)

    monkeypatch.setattr(
        llm.config, "llm_fallback_models",
        ["dead/model:free", "gemini-2.5-flash"],
        raising=False,
    )
    result = llm.call_llm_json("hello", model="dead/model:free")
    # With dead/model:free dead-listed AND primary, the chain becomes
    # [gemini-2.5-flash] only. Success on first (and only) attempt.
    assert result == {"ok": True}
    # Exactly one HTTP call (to the live fallback) — zero to the dead model.
    assert len(success_client.calls) == 1
    assert success_client.calls[0]["model"] == "gemini-2.5-flash"


# ------------------------------------------- no same-model retry on 429


def test_429_does_not_sleep_same_model_when_fallback_available(monkeypatch):
    """On 429 for a model, call_llm_json must move to the next fallback
    immediately instead of sleeping inside the same model's retry loop."""
    gemini_client = _ScriptedClient([_make_exc("429 Too Many Requests")])
    openrouter_client = _ScriptedClient([_fake_response('{"result": "ok"}')])

    def _router(m):
        return gemini_client if m.startswith("gemini") else openrouter_client

    monkeypatch.setattr(llm, "_client_for_model", _router)
    monkeypatch.setattr(llm._rate_limiter, "wait_if_needed", lambda m: None)
    # Patch _parse_retry_after so the rate-limiter records a long block (which
    # the old code would sleep through).
    monkeypatch.setattr(llm, "_parse_retry_after", lambda e: 60.0)

    monkeypatch.setattr(
        llm.config, "llm_fallback_models",
        ["deepseek/deepseek-chat:free"],
        raising=False,
    )

    # time.sleep is the smoking gun — the old code called it for ~60s between
    # same-model retries. With the fix in place, call_llm_json never sleeps.
    sleep_calls: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleep_calls.append(s))

    t0 = time.time()
    result = llm.call_llm_json("hi", model="gemini-2.5-flash-lite")
    elapsed = time.time() - t0

    assert result == {"result": "ok"}
    # Exactly 1 call each — no same-model retry.
    assert len(gemini_client.calls) == 1
    assert len(openrouter_client.calls) == 1
    # No long sleeps happened inside the chain.
    assert all(s < 1.0 for s in sleep_calls), f"unexpected long sleeps: {sleep_calls}"
    # Wall-clock should be tiny because nothing waited.
    assert elapsed < 2.0


# ---------------------------------- skip blocked primary when fallback ready


def test_blocked_primary_skipped_when_later_fallback_unblocked(monkeypatch):
    """If Google is blocked for 60s and OpenRouter is ready, the chain must
    skip Google entirely — not call wait_if_needed on it."""
    # Pre-populate the rate-limiter with a fresh 60s block on flash-lite.
    llm._rate_limiter.record_429("gemini-2.5-flash-lite", retry_after_s=60.0)

    openrouter_client = _ScriptedClient([_fake_response('{"ok": 1}')])
    gemini_wait_called = []

    def _router(m):
        if m.startswith("gemini"):
            return _ScriptedClient(
                [_make_exc("should not be called")]
            )
        return openrouter_client

    monkeypatch.setattr(llm, "_client_for_model", _router)

    real_wait = llm._rate_limiter.wait_if_needed

    def _tracking_wait(m):
        if m.startswith("gemini"):
            gemini_wait_called.append(m)
        # Don't actually sleep for anything.

    monkeypatch.setattr(llm._rate_limiter, "wait_if_needed", _tracking_wait)
    monkeypatch.setattr(
        llm.config, "llm_fallback_models",
        ["deepseek/deepseek-chat:free"],
        raising=False,
    )

    result = llm.call_llm_json("hi", model="gemini-2.5-flash-lite")
    assert result == {"ok": 1}
    # Google was blocked > _MAX_BLOCK_SKIP_S (5s) and OpenRouter wasn't,
    # so gemini's wait_if_needed should never have run.
    assert gemini_wait_called == []
    assert len(openrouter_client.calls) == 1


def test_all_blocked_still_calls_primary(monkeypatch):
    """When every model in the chain is blocked we must still try the
    primary rather than returning None without a single HTTP call."""
    llm._rate_limiter.record_429("gemini-2.5-flash-lite", retry_after_s=60.0)
    llm._rate_limiter.record_429("deepseek/deepseek-chat:free", retry_after_s=60.0)

    primary_client = _ScriptedClient([_fake_response('{"ok": "primary"}')])

    def _router(m):
        return primary_client if m.startswith("gemini") else _ScriptedClient([])

    monkeypatch.setattr(llm, "_client_for_model", _router)
    monkeypatch.setattr(llm._rate_limiter, "wait_if_needed", lambda m: None)
    monkeypatch.setattr(
        llm.config, "llm_fallback_models",
        ["deepseek/deepseek-chat:free"],
        raising=False,
    )

    result = llm.call_llm_json("hi", model="gemini-2.5-flash-lite")
    assert result == {"ok": "primary"}
    assert len(primary_client.calls) == 1


def test_invalid_model_phrase_also_triggers_dead_listing(monkeypatch):
    monkeypatch.setattr(llm, "get_llm_client", lambda: object())
    fake_client = _ScriptedClient(
        [_make_exc("Error: invalid model 'qwen/foo:free'")]
    )
    parsed, status = llm._try_once(
        client=fake_client,
        model="qwen/foo:free",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        use_json_mode=False,
    )
    assert status == "model_dead"
    assert llm._is_model_dead("qwen/foo:free")
