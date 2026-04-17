"""Tests for the OpenRouter-aware LLM client routing.

The live map-maker run kept exhausting Google AI Studio's daily quota — the
fallback chain was all Gemini, all on the same account, so hitting one
ceiling hit every fallback. The fix adds OpenRouter as a second provider:
models are routed by id format (presence of '/'), the rate limiter treats
OpenRouter's free tier as a single pooled bucket, and models whose provider
key is missing are silently dropped from the fallback chain.

These tests exercise that routing and pooling logic without making any
network calls.
"""

from __future__ import annotations

from polyquant.utils import llm_client as llm


def test_is_openrouter_model_by_slash_rule():
    # Google AI Studio ids never contain a slash.
    assert llm._is_openrouter_model("gemini-2.5-flash-lite") is False
    assert llm._is_openrouter_model("gemini-2.5-pro") is False
    # OpenRouter ids always do.
    assert llm._is_openrouter_model("deepseek/deepseek-chat-v3-0324:free") is True
    assert llm._is_openrouter_model("meta-llama/llama-3.3-70b-instruct:free") is True
    assert llm._is_openrouter_model("google/gemini-2.0-flash-exp:free") is True


def test_free_openrouter_models_share_pool_key():
    # All OpenRouter :free models are rate-limited as a single pool because
    # OpenRouter bills at the account level, not per model. The
    # `openrouter/free` meta-router also dispatches through the free pool and
    # must share the same bucket — otherwise it'd wall-clock ahead of the
    # individual free models under concurrent load.
    a = llm._rate_limit_key("deepseek/deepseek-chat-v3-0324:free")
    b = llm._rate_limit_key("meta-llama/llama-3.3-70b-instruct:free")
    c = llm._rate_limit_key("qwen/qwen-2.5-72b-instruct:free")
    router = llm._rate_limit_key("openrouter/free")
    assert a == b == c == router == llm._OPENROUTER_FREE_POOL_KEY


def test_openrouter_free_router_routes_to_openrouter_provider(monkeypatch):
    # Sanity: `openrouter/free` contains a '/', so _client_for_model must
    # hand it to the OpenRouter factory, not Google AI Studio.
    google_sentinel = object()
    openrouter_sentinel = object()
    monkeypatch.setattr(llm, "get_llm_client", lambda: google_sentinel)
    monkeypatch.setattr(llm, "_get_openrouter_client", lambda: openrouter_sentinel)
    assert llm._client_for_model("openrouter/free") is openrouter_sentinel


def test_google_models_each_get_their_own_bucket():
    # Google meters per model, so each gets a unique rate-limit bucket.
    assert llm._rate_limit_key("gemini-2.5-flash") == "gemini-2.5-flash"
    assert llm._rate_limit_key("gemini-2.5-flash-lite") == "gemini-2.5-flash-lite"
    assert llm._rate_limit_key("gemini-2.5-pro") == "gemini-2.5-pro"


def test_rate_limiter_shares_blocks_across_pooled_openrouter_models():
    # A 429 on one OpenRouter free model must throttle the rest because they
    # share an account-level bucket.
    limiter = llm._RateLimiter()
    limiter.record_429("deepseek/deepseek-chat-v3-0324:free", retry_after_s=30.0)
    # Sibling free model is now blocked too.
    assert limiter.blocked_for("meta-llama/llama-3.3-70b-instruct:free") > 0
    # But Google models are unaffected.
    assert limiter.blocked_for("gemini-2.5-flash") == 0


def test_rate_limiter_does_not_cross_contaminate_google_models():
    # Google models are metered independently — a 429 on flash must not
    # block flash-lite.
    limiter = llm._RateLimiter()
    limiter.record_429("gemini-2.5-flash", retry_after_s=60.0)
    assert limiter.blocked_for("gemini-2.5-flash-lite") == 0
    assert limiter.blocked_for("gemini-2.5-flash") > 0


def test_openrouter_free_pool_has_rpm_entry():
    # The pool key must be registered in the RPM table or the default 5 RPM
    # kicks in — much too slow (we want 20 RPM).
    assert llm._MODEL_RPM[llm._OPENROUTER_FREE_POOL_KEY] == llm._OPENROUTER_FREE_POOL_RPM
    assert llm._OPENROUTER_FREE_POOL_RPM >= 10  # sanity


def test_client_for_model_routes_by_id(monkeypatch):
    # Stub both factories: return a unique sentinel per provider so we can
    # assert the dispatcher routed correctly without any real network client.
    google_sentinel = object()
    openrouter_sentinel = object()

    monkeypatch.setattr(llm, "get_llm_client", lambda: google_sentinel)
    monkeypatch.setattr(llm, "_get_openrouter_client", lambda: openrouter_sentinel)

    assert llm._client_for_model("gemini-2.5-flash") is google_sentinel
    assert llm._client_for_model("gemini-2.5-pro") is google_sentinel
    assert (
        llm._client_for_model("deepseek/deepseek-chat-v3-0324:free")
        is openrouter_sentinel
    )


def test_client_for_model_returns_none_when_provider_missing(monkeypatch):
    # With OpenRouter key unset, the OpenRouter factory returns None.
    # _client_for_model must pass that through so call_llm_json can drop
    # the model from the fallback chain silently.
    monkeypatch.setattr(llm, "get_llm_client", lambda: object())
    monkeypatch.setattr(llm, "_get_openrouter_client", lambda: None)

    assert llm._client_for_model("deepseek/deepseek-chat-v3-0324:free") is None
    # Google-side still resolves.
    assert llm._client_for_model("gemini-2.5-flash") is not None


def test_parse_retry_after_handles_openrouter_x_ratelimit_reset():
    # OpenRouter ships X-RateLimit-Reset as a future unix-ms timestamp
    # instead of Retry-After. _parse_retry_after must convert that to a
    # relative delay.
    import time

    class FakeHeaders:
        def __init__(self, data):
            self._data = data

        def get(self, key):
            return self._data.get(key.lower()) or self._data.get(key)

    class FakeResponse:
        def __init__(self, headers):
            self.headers = headers

    class FakeExc(Exception):
        def __init__(self, response):
            self.response = response

    future_ms = (time.time() + 45) * 1000  # 45s in the future
    exc = FakeExc(FakeResponse(FakeHeaders({"x-ratelimit-reset": str(future_ms)})))
    delay = llm._parse_retry_after(exc)
    # Should be ~45s, definitely not the 60s default.
    assert 30 <= delay <= 60


def test_parse_retry_after_falls_back_to_default_when_no_headers():
    class FakeExc(Exception):
        pass

    delay = llm._parse_retry_after(FakeExc())
    assert delay == llm._DEFAULT_RETRY_AFTER_S


# ---------------------------------------------------------------------------
# JSON-mode gating — openrouter/free dispatches to a grab-bag of free models,
# some of which reject response_format. The rest of these tests lock in which
# model ids get JSON mode and which go text-mode.
# ---------------------------------------------------------------------------


class _CaptureClient:
    """Minimal stub for OpenAI client — records the kwargs of the last .create() call
    and returns a canned JSON response so call_llm_json completes successfully."""

    def __init__(self):
        self.last_kwargs: dict | None = None
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.last_kwargs = kwargs

                class _Msg:
                    content = '{"ok": true}'

                class _Choice:
                    message = _Msg()
                    finish_reason = "stop"

                class _Resp:
                    choices = [_Choice()]
                    model = kwargs.get("model", "unknown")

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_openrouter_free_does_not_request_json_mode(monkeypatch):
    # The free router fails on models that don't support response_format, so
    # we drop the param up front and rely on text-mode JSON parsing.
    client = _CaptureClient()
    monkeypatch.setattr(llm, "_client_for_model", lambda m: client)
    monkeypatch.setattr(llm, "_is_model_dead", lambda m: False)

    result = llm.call_llm_json("Return JSON.", model="openrouter/free")
    assert result == {"ok": True}
    assert client.last_kwargs is not None
    assert "response_format" not in client.last_kwargs
    # require_parameters is gated on use_json_mode, so no extra_body either.
    assert "extra_body" not in client.last_kwargs


def test_openrouter_pinned_free_model_still_uses_json_mode(monkeypatch):
    # Guard against over-broad gating: pinned `:free` slugs route to one known
    # provider whose JSON-mode support we've vetted when we added it.
    client = _CaptureClient()
    monkeypatch.setattr(llm, "_client_for_model", lambda m: client)
    monkeypatch.setattr(llm, "_is_model_dead", lambda m: False)

    result = llm.call_llm_json(
        "Return JSON.", model="deepseek/deepseek-chat-v3-0324:free"
    )
    assert result == {"ok": True}
    assert client.last_kwargs is not None
    assert client.last_kwargs.get("response_format") == {"type": "json_object"}
    # OpenRouter models also get the require_parameters hint.
    assert client.last_kwargs.get("extra_body") == {
        "provider": {"require_parameters": True}
    }


def test_google_model_still_uses_json_mode(monkeypatch):
    # Gemini 2.5 family fully supports response_format — no reason to drop it.
    client = _CaptureClient()
    monkeypatch.setattr(llm, "_client_for_model", lambda m: client)
    monkeypatch.setattr(llm, "_is_model_dead", lambda m: False)

    result = llm.call_llm_json("Return JSON.", model="gemini-2.5-flash")
    assert result == {"ok": True}
    assert client.last_kwargs is not None
    assert client.last_kwargs.get("response_format") == {"type": "json_object"}
    # Google models get no extra_body (no OpenRouter-specific hints).
    assert "extra_body" not in client.last_kwargs


def test_json_mode_unsupported_detects_gemma_phrasing(monkeypatch):
    # Safety net for the detection pattern: if someone re-enables JSON mode
    # on `openrouter/free` and hits gemma, the in-loop retry must fire.
    class _FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError(
                "Error code: 400 - JSON mode is not enabled for "
                "models/gemma-3-4b-it"
            )

    class _FailingChat:
        completions = _FailingCompletions()

    class FailingClient:
        chat = _FailingChat()

    # Skip the rate limiter in this test — we're only exercising error
    # classification, not pacing behavior.
    monkeypatch.setattr(llm._rate_limiter, "wait_if_needed", lambda m: None)

    parsed, status = llm._try_once(
        client=FailingClient(),
        model="openrouter/free",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.0,
        use_json_mode=True,
    )
    assert parsed is None
    assert status == "json_mode_unsupported"
