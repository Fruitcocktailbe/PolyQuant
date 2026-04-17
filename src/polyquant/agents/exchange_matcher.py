"""
Exchange Matcher Agent

Four-layer funnel matching Polymarket and Limitless markets that resolve on
the same real-world event:

  Layer 0: fetch + harmonized liquidity filter
  Layer 1: structural pre-filter (expiry / outcome count / numeric / domain)
  Layer 2: bidirectional top-K embedding recall (question-only)
  Layer 3: LLM semantic verification
  Layer 4: persistent decision cache (accepted + rejected)

Accepted pairs are reused across runs and never re-verified. Rejected pairs
are skipped before they reach the LLM (delete the entry to force re-check).
"""

import gc
import os
import json
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple
from pathlib import Path

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from polyquant.data.polymarket_client import PolymarketClient
from polyquant.data.limitless_client import LimitlessClient
from polyquant.utils.llm_client import call_llm_json
from polyquant.utils import get_logger, config
from polyquant.agents.match_prefilter import (
    Fingerprint,
    fingerprint_polymarket,
    fingerprint_limitless,
    compatible,
    _limitless_outcome_count,
)

logger = get_logger(__name__)

CACHE_DIR = Path(".polyquant/constraints")
CACHE_FILE = CACHE_DIR / "market_pairs.json"

LIMITLESS_PAGE_SIZE = 20
EMBEDDING_TOP_K = 5
SIMILARITY_FLOOR = 0.50      # absolute floor — below this we don't even consider
SIMILARITY_VERIFY = 0.65     # threshold for sending to LLM
LLM_RPS_SEMAPHORE = 2
LLM_PACING_SECONDS = 1.0
ENCODE_CHUNK_SIZE = 500      # bounds peak RAM on the 2 GB Lightsail VM

LLM_VERIFY_PROMPT = """
You are an arbitrage trading engine.
Your task is to determine if two prediction markets resolve based on the SAME real-world event.

CRITICAL DISTINCTIONS:
- Different resolution SOURCES (e.g. "AP call" vs "NYT projection" for the same election winner)
  describing the SAME real-world event are A MATCH. Different oracles reporting the same fact
  still pay out on the same fact.
- The same source FAMILY pointing at DIFFERENT real-world events is NOT a match. A CoinGecko
  12:00 UTC BTC snapshot and a Coinbase 12:05 UTC BTC snapshot are different events even though
  both are crypto price oracles.
- Different timeframes for the same underlying question are NOT a match (e.g. "BTC hits 100k in
  May" vs "BTC hits 100k in June"). Small phrasing differences around the same deadline are fine.

POLARITY — THIS IS LOAD-BEARING:
Both exchanges label outcomes "Yes" / "No". A match is only usable if Polymarket YES resolves
on the SAME condition as Limitless YES. If the questions are worded in opposite directions,
the Yes/No tokens represent OPPOSITE events even though both are labeled "Yes". Trading the
pair as aligned would produce a sign-inverted position — real capital loss.

Examples of MATCH (aligned polarity):
- "Will BTC hit $100k by 2026?" vs "Will Bitcoin reach $100k before Jan 2026?" → YES aligned
- "Trump to win 2024 election" vs "Donald Trump victor in 2024" → YES aligned
- "AP calls election for Harris" vs "NYT projects Harris wins" → YES aligned

Examples of MATCH but INVERTED polarity:
- "Will Trump be elected?" (YES = Trump elected) vs "Will Trump fail to be elected?"
  (YES = Trump NOT elected) — same event, opposite polarity → set yes_polarity_aligned=false
- "Will the bill pass?" vs "Will the bill be rejected?" — same event, inverted → false

Examples of NOT A MATCH:
- "Who will win the election?" vs "Will Trump win the election?" (multi-choice vs binary)
- "Will BTC hit 100k in May?" vs "Will BTC hit 100k in June?" (different timeframes)
- "Will ETH be above $3000?" vs "Will ETH be above $3000 OR BTC above $100k?" (extra conditions)
- "BTC price at 12:00 UTC" vs "BTC price at 12:05 UTC" (same source family, different snapshots)

Polymarket Question: {p_q}
Polymarket Description: {p_d}
Polymarket Resolution Source: {p_res}
Polymarket End Date: {p_end}

Limitless Candidate Question: {l_q}
Limitless Candidate Description: {l_d}
Limitless Resolution Source: {l_res}
Limitless End Date: {l_end}

SPORTS MONEYLINE PROJECTION (only relevant when one side is a 3-way market
home/draw/away and the other is 2-way YES/NO):
When pairing a Polymarket 3-way moneyline with a Limitless 2-way, resolution on
a draw is the key subtlety. Set "draw_rule" to:
- "draw_is_no": Limitless YES resolves only when the home side wins outright; draw
  and away both pay NO. This is the normal "outright win" bookmaker behaviour.
- "double_chance": Limitless YES covers home-or-draw (home doesn't lose); only an
  away win pays NO.
- "none": this is not a 3-way ↔ 2-way pairing (both sides are binary, or both
  are 3-way).
- "unclear": the Limitless rules text doesn't make the draw handling obvious.
  Return unclear whenever you can't tell with confidence — we won't trade the
  pair if draw handling is ambiguous.
Also return "home_outcome_name": the exact outcome name on the Polymarket 3-way
side that corresponds to the Limitless YES leg (e.g. "Manchester United" for a
"Will Manchester United win?" binary). Leave empty when draw_rule is "none".

Respond ONLY in JSON with these fields:
- "is_match" (bool): overall judgment — do these pay out on the same tradable outcome?
- "same_real_world_event" (bool): do both markets pay out on the same underlying real-world
  event/fact, regardless of source wording or question direction? This is the core correctness
  question — a false here must make the pair unsafe to trade even if surface text looks similar.
- "timeframes_match" (bool): are the resolution windows effectively identical, allowing for
  natural phrasing differences (e.g. "by end of 2026" vs "before Jan 1 2027")?
- "yes_polarity_aligned" (bool): when both markets resolve to YES, are they resolving on the
  SAME direction of the real-world event? True for normal matches; false when one question is
  phrased as the negation of the other (e.g. "Will X happen?" vs "Will X fail to happen?").
  If you cannot tell with confidence, return false — a wrong true here is a sign-inverted trade.
- "draw_rule" (str): one of "none" / "draw_is_no" / "double_chance" / "unclear" as described
  above. MUST be "none" for binary-vs-binary pairs; MUST be one of the other three values for
  3-way vs 2-way sports pairs.
- "home_outcome_name" (str): exact Polymarket outcome name mapped to Limitless YES when the
  pair is 3-way vs 2-way; empty string otherwise.
- "reasoning" (str): short justification, mention polarity and draw handling explicitly.

{{
    "is_match": true/false,
    "same_real_world_event": true/false,
    "timeframes_match": true/false,
    "yes_polarity_aligned": true/false,
    "draw_rule": "none" | "draw_is_no" | "double_chance" | "unclear",
    "home_outcome_name": "...",
    "reasoning": "..."
}}
"""


# Bump when the accepted-entry schema changes in a way that makes old entries
# untrustworthy (e.g. a new correctness field is added). _load_cache drops any
# accepted entry written under an older version, forcing next-run re-verification.
CACHE_SCHEMA_VERSION = 3


VALID_DRAW_RULES = frozenset({"none", "draw_is_no", "double_chance"})


@dataclass
class _AcceptedEntry:
    limitless_id: str
    similarity: float
    reasoning: str
    verified_at: str
    # When True, PM-YES ≡ LM-YES (standard alignment). When False, PM-YES ≡ LM-NO
    # (inverted polarity — questions phrased as negations of each other). The
    # solver must swap YES↔NO tokens during cross-exchange injection when this
    # is False. Entries loaded from pre-v2 cache get polarity_aligned=None and
    # are dropped on load so the LLM re-verifies under the new schema.
    polarity_aligned: bool = True
    # Sports moneyline projection metadata. "none" means this is a standard
    # binary-vs-binary pair; "draw_is_no" / "double_chance" describe how a
    # 3-way Polymarket moneyline projects onto a 2-way Limitless binary. The
    # injection path consumes these to synthesise the correct linear relation
    # (see map_maker._inject_threeway_moneyline_equivalencies).
    draw_rule: str = "none"
    home_outcome_name: str = ""
    schema_version: int = CACHE_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _RejectedEntry:
    similarity: float
    reasoning: str
    rejected_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_monitor() -> Any:
    try:
        from polyquant.api.server import monitor as _m
        return _m
    except Exception:
        return None


async def _notify_llm_progress(stage: str, *, done: int, total: int | None = None, current: str = "") -> None:
    m = _get_monitor()
    if m is None:
        return
    try:
        if total is not None:
            await m.update_llm_progress(stage, done=done, total=total, current=current)
        else:
            await m.update_llm_progress(stage, done=done, current=current)
    except Exception:
        pass


async def _notify_mapped_pair(ui_pair: dict) -> None:
    m = _get_monitor()
    if m is None:
        return
    try:
        current_pairs = list(m.state.mapped_pairs)
        current_pairs.append(ui_pair)
        await m.update_status(mapped_pairs=current_pairs)
    except Exception:
        pass


def _limitless_dollars(market: dict) -> Tuple[float, float]:
    """Return (volume, liquidity) in dollars, handling raw micro-USDC."""
    try:
        vol = float(market.get("volumeFormatted", 0) or market.get("volume", 0) or 0)
        liq = float(market.get("liquidityFormatted", 0) or market.get("liquidity", 0) or 0)
    except (ValueError, TypeError):
        return 0.0, 0.0
    if vol > 1_000_000:
        vol /= 1_000_000
    if liq > 1_000_000:
        liq /= 1_000_000
    return vol, liq


class ExchangeMatcher:
    def __init__(self):
        self._accepted: Dict[str, _AcceptedEntry] = {}
        self._rejected: Dict[str, _RejectedEntry] = {}
        self._model: Any = None
        self._load_cache()

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            return None
        logger.info("Loading semantic embedding model (all-MiniLM-L6-v2)...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    # ------------------------------------------------------------------ cache

    @property
    def mapped_pairs(self) -> Dict[str, str]:
        """Backwards-compat view: {polymarket_id: limitless_id}."""
        return {pid: e.limitless_id for pid, e in self._accepted.items()}

    def get_pair_polarity(self, polymarket_id: str) -> bool:
        """Return True if PM-YES ≡ LM-YES for this pair (standard alignment),
        False if PM-YES ≡ LM-NO (inverted). Default True so callers that don't
        find the pair fall back to the safe-looking assumption, but callers
        should always check that the pair exists before using this."""
        entry = self._accepted.get(polymarket_id)
        if entry is None:
            return True
        return entry.polarity_aligned

    def get_pair_projection(self, polymarket_id: str) -> tuple[str, str]:
        """Return (draw_rule, home_outcome_name) for the 3-way ↔ 2-way
        sports moneyline projection recorded with this pair. Returns
        ("none", "") when the pair is a standard binary-vs-binary match or
        when no pair is cached for the given Polymarket id."""
        entry = self._accepted.get(polymarket_id)
        if entry is None:
            return ("none", "")
        return (entry.draw_rule, entry.home_outcome_name)

    def _load_cache(self) -> None:
        if not CACHE_FILE.exists():
            return
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load market pairs cache: {e}")
            return

        # Historical corruption guard: earlier parser bugs caused some
        # Polymarket markets to have an empty market_id, which got written
        # into the cache as an empty-string key. An empty key in mapped_pairs
        # causes phantom Limitless injection on every cluster via
        # mapped.get("") → returns the same slug for every market. Drop any
        # entry with an empty key (or empty Limitless id) on load so old
        # poisoned caches self-clean after the parser fix ships.
        dropped_empty_keys = 0
        dropped_empty_values = 0
        # Pre-v2 entries predate the polarity field — treating them as aligned
        # would silently reintroduce sign-inverted trades, so they're dropped
        # and re-verified on next run.
        dropped_pre_schema = 0

        if isinstance(data, dict) and ("accepted" in data or "rejected" in data):
            for pid, entry in (data.get("accepted") or {}).items():
                try:
                    if not pid:
                        dropped_empty_keys += 1
                        continue
                    lid = entry["limitless_id"]
                    # Drop legacy entries that stored Limitless's numeric `id`
                    # instead of the URL `slug`. The orderbook endpoint expects
                    # a slug, so any all-digit value here is a stale entry from
                    # before the slug fix and would 404 downstream.
                    if not isinstance(lid, str) or not lid or lid.isdigit():
                        dropped_empty_values += 1
                        continue
                    entry_version = entry.get("schema_version", 1)
                    if entry_version < CACHE_SCHEMA_VERSION:
                        dropped_pre_schema += 1
                        continue
                    draw_rule = entry.get("draw_rule", "none")
                    if draw_rule not in VALID_DRAW_RULES:
                        draw_rule = "none"
                    self._accepted[pid] = _AcceptedEntry(
                        limitless_id=lid,
                        similarity=float(entry.get("similarity", 0.0)),
                        reasoning=entry.get("reasoning", ""),
                        verified_at=entry.get("verified_at", ""),
                        polarity_aligned=bool(entry.get("polarity_aligned", True)),
                        draw_rule=draw_rule,
                        home_outcome_name=str(entry.get("home_outcome_name", "") or ""),
                        schema_version=int(entry_version),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            ttl_days = int(getattr(config, "rejection_cache_ttl_days", 0) or 0)
            ttl_cutoff = (
                datetime.now(timezone.utc) - timedelta(days=ttl_days)
                if ttl_days > 0
                else None
            )
            expired_rejections = 0
            for key, entry in (data.get("rejected") or {}).items():
                try:
                    if not key:
                        dropped_empty_keys += 1
                        continue
                    rejected_at_raw = entry.get("rejected_at", "") or ""
                    if ttl_cutoff is not None and rejected_at_raw:
                        # A malformed timestamp is itself a reason to re-verify
                        # rather than trust indefinitely, so parse failures fall
                        # through to expiry handling.
                        try:
                            rejected_at_dt = datetime.fromisoformat(rejected_at_raw)
                        except ValueError:
                            rejected_at_dt = None
                        if rejected_at_dt is None or rejected_at_dt < ttl_cutoff:
                            expired_rejections += 1
                            continue
                    self._rejected[key] = _RejectedEntry(
                        similarity=float(entry.get("similarity", 0.0)),
                        reasoning=entry.get("reasoning", ""),
                        rejected_at=rejected_at_raw,
                    )
                except (TypeError, ValueError):
                    continue
            if expired_rejections:
                logger.info(
                    "Re-verifying expired rejection-cache entries",
                    expired=expired_rejections,
                    ttl_days=ttl_days,
                    hint="markets whose LLM rejection is older than ttl will "
                    "be re-sent to the LLM on this run",
                )
        elif isinstance(data, dict):
            # Legacy flat format {poly_id: limitless_id} predates the polarity
            # field entirely — drop everything and let next run re-verify.
            dropped_pre_schema = len(data)

        if dropped_empty_keys or dropped_empty_values or dropped_pre_schema:
            logger.warning(
                "Dropped entries from market_pairs.json on load",
                dropped_empty_keys=dropped_empty_keys,
                dropped_empty_values=dropped_empty_values,
                dropped_pre_schema=dropped_pre_schema,
                hint=(
                    "empty-key/value from historical parser bug; pre-schema "
                    f"entries (<v{CACHE_SCHEMA_VERSION}) lack polarity_aligned "
                    "and will be re-verified next run"
                ),
            )

        logger.info(
            f"Loaded matcher cache: {len(self._accepted)} accepted, {len(self._rejected)} rejected "
            "(pre-schema-v2 entries retained; delete manually to force re-verify under the hardened "
            "same_real_world_event + timeframes_match gate)."
        )

    def _save_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Sanitize on write too — even if a runtime bug slips an empty-key
        # pair into self._accepted / self._rejected, we refuse to persist it
        # so the cache never re-poisons itself.
        payload = {
            "accepted": {
                pid: e.to_dict()
                for pid, e in self._accepted.items()
                if pid and e.limitless_id
            },
            "rejected": {
                k: e.to_dict()
                for k, e in self._rejected.items()
                if k
            },
        }
        tmp = CACHE_FILE.with_suffix(CACHE_FILE.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, CACHE_FILE)

    def save_cache(self) -> None:
        self._save_cache()

    # ----------------------------------------------------------- fetch markets

    async def _fetch_polymarket(self) -> tuple[list[Any], int]:
        floor = config.min_liquidity_matcher_polymarket
        logger.info(
            f"Polymarket: fetching markets at liquidity floor ${floor:,.0f} "
            f"(server-side filter)"
        )
        markets: list[Any] = []
        async with PolymarketClient() as p_client:
            offset = 0
            page_size = 500
            while True:
                batch, raw_count = await p_client.get_active_markets(
                    limit=page_size, offset=offset, min_liquidity=floor
                )
                markets.extend(batch)
                offset += page_size
                if raw_count < page_size:
                    break
        return markets, len(markets)

    async def _fetch_limitless(self) -> list[dict]:
        try:
            async with LimitlessClient() as l_client:
                return await l_client.get_markets(limit=LIMITLESS_PAGE_SIZE)
        except Exception as e:
            logger.error(f"LimitlessClient context failed: {e}")
        # Fallback: raw HTTP without signing
        out: list[dict] = []
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as raw:
                page = 1
                while True:
                    resp = await raw.get(
                        f"{config.limitless_api_url}/markets/active",
                        params={"limit": LIMITLESS_PAGE_SIZE, "page": page},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    batch = data.get("data", []) or data.get("markets", [])
                    if not batch:
                        break
                    out.extend(batch)
                    page += 1
                    if len(batch) < LIMITLESS_PAGE_SIZE:
                        break
            logger.info(f"Fallback fetch succeeded: {len(out)} Limitless markets")
        except Exception as fallback_err:
            logger.error(f"Fallback Limitless fetch also failed: {fallback_err}")
        return out

    # ----------------------------------------------------- main pipeline entry

    async def run_matching_pipeline(self) -> tuple[Dict[str, str], Dict[str, Any]]:
        if SentenceTransformer is None:
            logger.error("sentence-transformers is not installed. Run `pip install sentence-transformers`")
            return self.mapped_pairs, {"error": "sentence-transformers not installed"}

        # ---------- Layer 0: floor-alignment audit (Gap 5a) ----------
        # The matcher fetches every Polymarket market above its own floor,
        # independent of Discovery's clustering. If the matcher floor is
        # *higher* than the Discovery floor, we silently miss pairs for
        # markets Discovery saw. Warn loudly so operators can align floors.
        matcher_floor = config.min_liquidity_matcher_polymarket
        discovery_floor = config.min_liquidity
        if matcher_floor > discovery_floor:
            logger.warning(
                "Matcher Polymarket floor is STRICTER than Discovery floor — cross-exchange "
                "matching will miss markets Discovery sees. Consider aligning the two.",
                matcher_floor=matcher_floor,
                discovery_floor=discovery_floor,
                gap=matcher_floor - discovery_floor,
            )

        # ---------- Layer 0: fetch + harmonized liquidity filter ----------
        logger.info("Layer 0: Fetching markets from both exchanges...")
        poly_markets, poly_fetched_total = await self._fetch_polymarket()
        limit_markets_raw = await self._fetch_limitless()

        limitless_floor = config.min_liquidity_matcher_limitless
        logger.info(
            f"Limitless: filtering {len(limit_markets_raw)} raw markets at "
            f"liquidity floor ${limitless_floor:,.0f} (client-side filter)"
        )
        limit_markets: list[dict] = []
        for m in limit_markets_raw:
            vol, liq = _limitless_dollars(m)
            if max(vol, liq) >= limitless_floor:
                limit_markets.append(m)

        logger.info(
            f"Layer 0 complete: {len(poly_markets)} Polymarket "
            f"(floor ${config.min_liquidity_matcher_polymarket:,.0f}) | "
            f"{len(limit_markets)} Limitless (floor ${limitless_floor:,.0f}) "
            f"after filter."
        )

        pipeline_stats: Dict[str, Any] = {
            "polymarket_fetched": poly_fetched_total,
            "polymarket_fetched_at_floor": len(poly_markets),
            "limitless_fetched_raw": len(limit_markets_raw),
            "limitless_kept_at_floor": len(limit_markets),
            "limitless_discarded": len(limit_markets_raw) - len(limit_markets),
            "already_accepted": len(self._accepted),
            "already_rejected": len(self._rejected),
            "prefilter_dropped": 0,
            "rejection_cache_hits": 0,
            "candidate_pairs_after_prefilter": 0,
            "llm_verifications_sent": 0,
            "llm_matches_confirmed": 0,
            "llm_matches_rejected": 0,
            "llm_errors": 0,
            "new_pairs_found": 0,
            "new_rejections_cached": 0,
            "total_pairs_after": 0,
            "matched_pairs_detail": [],
        }

        if not poly_markets or not limit_markets:
            pipeline_stats["total_pairs_after"] = len(self._accepted)
            return self.mapped_pairs, pipeline_stats

        if not config.enable_semantic_matching:
            logger.info("Semantic matching disabled by config; returning cached pairs only.")
            pipeline_stats["total_pairs_after"] = len(self._accepted)
            return self.mapped_pairs, pipeline_stats

        from polyquant.utils.llm_client import get_llm_client
        if get_llm_client() is None:
            logger.info("LLM key missing; returning cached pairs only.")
            pipeline_stats["total_pairs_after"] = len(self._accepted)
            return self.mapped_pairs, pipeline_stats

        # ---------- Layer 1: build fingerprints ----------
        logger.info("Layer 1: Building structural fingerprints...")
        poly_fps: list[Fingerprint] = [fingerprint_polymarket(m) for m in poly_markets]
        limit_fps: list[Fingerprint] = [fingerprint_limitless(m) for m in limit_markets]

        # ---------- Layer 2: question-only embeddings ----------
        try:
            model = self._get_model()
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            pipeline_stats["total_pairs_after"] = len(self._accepted)
            return self.mapped_pairs, pipeline_stats
        if model is None:
            logger.error("Semantic model unavailable; returning cached pairs only.")
            pipeline_stats["total_pairs_after"] = len(self._accepted)
            return self.mapped_pairs, pipeline_stats

        p_docs = [m.question for m in poly_markets]
        l_docs = [m.get("title", "") for m in limit_markets]

        logger.info(
            f"Encoding {len(p_docs)} Polymarket + {len(l_docs)} Limitless questions "
            f"(chunk={ENCODE_CHUNK_SIZE})..."
        )

        def _encode_chunked(docs: list[str]) -> np.ndarray:
            if not docs:
                return np.zeros(
                    (0, model.get_sentence_embedding_dimension()), dtype=np.float32
                )
            chunks: list[np.ndarray] = []
            for start in range(0, len(docs), ENCODE_CHUNK_SIZE):
                chunks.append(
                    model.encode(
                        docs[start:start + ENCODE_CHUNK_SIZE],
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                )
            return np.vstack(chunks)

        p_emb = _encode_chunked(p_docs)
        l_emb = _encode_chunked(l_docs)

        # Pure-numpy cosine so we can release torch tensors before the slow Layer 3.
        p_norms = np.linalg.norm(p_emb, axis=1, keepdims=True)
        l_norms = np.linalg.norm(l_emb, axis=1, keepdims=True)
        p_normed = p_emb / np.where(p_norms == 0, 1.0, p_norms)
        l_normed = l_emb / np.where(l_norms == 0, 1.0, l_norms)
        sim_matrix = p_normed @ l_normed.T
        logger.info("Similarity matrix computed.")

        # Drop the ~300 MB SentenceTransformer + all intermediates before Layer 3's
        # LLM calls — sim_matrix is tiny (len(p)*len(l)*4 bytes) and is all we need.
        # _encode_chunked closes over `model`, so it must be deleted too.
        del model, p_emb, l_emb, p_normed, l_normed, p_norms, l_norms, _encode_chunked
        self._model = None
        gc.collect()

        # ---------- Layer 2 cont.: bidirectional top-K union ----------
        K = min(EMBEDDING_TOP_K, len(limit_markets))
        candidate_pairs: dict[tuple[int, int], float] = {}

        # Polymarket -> top-K Limitless
        for p_idx in range(len(poly_markets)):
            top = np.argsort(sim_matrix[p_idx])[-K:][::-1]
            for l_idx in top:
                sim = float(sim_matrix[p_idx][l_idx])
                if sim < SIMILARITY_FLOOR:
                    continue
                candidate_pairs[(p_idx, int(l_idx))] = sim

        # Limitless -> top-K Polymarket
        K2 = min(EMBEDDING_TOP_K, len(poly_markets))
        for l_idx in range(len(limit_markets)):
            top = np.argsort(sim_matrix[:, l_idx])[-K2:][::-1]
            for p_idx in top:
                sim = float(sim_matrix[p_idx][l_idx])
                if sim < SIMILARITY_FLOOR:
                    continue
                candidate_pairs[(int(p_idx), l_idx)] = sim

        logger.info(f"Bidirectional top-K union: {len(candidate_pairs)} raw candidate pairs")

        # ---------- Apply Layer 1 prefilter, rejection cache, accepted skip ----------
        # Iterate in similarity-desc order and keep only the best candidate per
        # Polymarket market — subsequent alternates would be discarded anyway by
        # the post-LLM leader-wins guard, so we save LLM spend up front.
        survivors: list[tuple[int, int, float]] = []
        seen_poly: set[str] = set()
        for (p_idx, l_idx), sim in sorted(
            candidate_pairs.items(), key=lambda kv: kv[1], reverse=True
        ):
            p_market = poly_markets[p_idx]
            l_market = limit_markets[l_idx]
            l_id = l_market.get("slug")
            if not isinstance(l_id, str) or not l_id:
                continue
            # Already accepted? (Polymarket id already mapped)
            if p_market.market_id in self._accepted:
                continue
            if p_market.market_id in seen_poly:
                continue
            # Already rejected?
            if f"{p_market.market_id}|{l_id}" in self._rejected:
                pipeline_stats["rejection_cache_hits"] += 1
                continue
            # Prefilter — expiry tolerance is config-driven (default 24h; was a
            # far-too-loose 1-week bucket before Gap 2 hardening). Crypto pairs
            # get a tighter tolerance because same-day BTC/ETH snapshot markets
            # can diverge meaningfully at even minute-scale offsets.
            if not compatible(
                poly_fps[p_idx],
                limit_fps[l_idx],
                expiry_tolerance_hours=config.cross_exchange_expiry_tolerance_hours,
                crypto_expiry_tolerance_hours=getattr(
                    config, "crypto_expiry_tolerance_hours", None
                ),
            ):
                pipeline_stats["prefilter_dropped"] += 1
                continue
            survivors.append((p_idx, l_idx, sim))
            seen_poly.add(p_market.market_id)

        # Drop pairs below the LLM-verify floor
        to_verify = [(p, l, s) for (p, l, s) in survivors if s >= SIMILARITY_VERIFY]

        pipeline_stats["candidate_pairs_after_prefilter"] = len(survivors)

        if not to_verify:
            logger.info("No candidate pairs survived prefilter + similarity threshold.")
            pipeline_stats["total_pairs_after"] = len(self._accepted)
            if pipeline_stats["new_rejections_cached"] > 0:
                self._save_cache()
            return self.mapped_pairs, pipeline_stats

        # ---------- Layer 3: LLM verification ----------
        logger.info(
            f"Layer 3: Sending {len(to_verify)} pairs to LLM (rate-limited, sem={LLM_RPS_SEMAPHORE})..."
        )
        pipeline_stats["llm_verifications_sent"] = len(to_verify)

        await _notify_llm_progress("MATCHING", done=0, total=len(to_verify), current="")

        sem = asyncio.Semaphore(LLM_RPS_SEMAPHORE)
        completed = 0
        completed_lock = asyncio.Lock()

        async def verify_one(p_idx: int, l_idx: int, sim: float) -> tuple[int, int, float, Any]:
            nonlocal completed
            p_market = poly_markets[p_idx]
            l_market = limit_markets[l_idx]
            prompt = LLM_VERIFY_PROMPT.format(
                p_q=p_market.question,
                p_d=p_market.description,
                p_res=p_market.resolution_source or "Not specified",
                p_end=str(p_market.end_date or "Not specified"),
                l_q=l_market.get("title", ""),
                l_d=l_market.get("description", ""),
                l_res=(
                    l_market.get("resolutionSource", "")
                    or l_market.get("rules", "")
                    or (l_market.get("description", "") or "")[:200]
                    or "Not specified"
                ),
                l_end=l_market.get("expirationDate", "")
                or l_market.get("expirationTimestamp", "")
                or l_market.get("endDate", "")
                or "Not specified",
            )
            async with sem:
                res = await asyncio.to_thread(
                    call_llm_json,
                    prompt=prompt,
                    system_prompt="Answer JSON only.",
                    temperature=0.1,
                    model=config.llm_model_matcher,
                )
                await asyncio.sleep(LLM_PACING_SECONDS)
            async with completed_lock:
                completed += 1
                await _notify_llm_progress(
                    "MATCHING",
                    done=completed,
                    current=(l_market.get("title", "") or "")[:60],
                )
            return p_idx, l_idx, sim, res

        results = await asyncio.gather(
            *(verify_one(p, l, s) for (p, l, s) in to_verify),
            return_exceptions=False,
        )

        # ---------- Layer 4: process + persist decisions ----------
        # Iterate in similarity-desc order; first true match per Polymarket wins.
        results.sort(key=lambda r: r[2], reverse=True)

        for p_idx, l_idx, sim, resp in results:
            p_market = poly_markets[p_idx]
            l_market = limit_markets[l_idx]
            l_id = l_market.get("slug")
            if not isinstance(l_id, str) or not l_id:
                continue

            if p_market.market_id in self._accepted:
                continue  # leader already won for this Polymarket

            if resp is None:
                pipeline_stats["llm_errors"] += 1
                logger.warning(
                    "LLM verification returned no result — leaving pair unverified",
                    poly=p_market.question[:40],
                    limitless=(l_market.get("title", "") or "")[:40],
                )
                continue

            is_match = resp.get("is_match") is True
            reasoning = resp.get("reasoning", "")
            # Gap-2 hardening: require both same-real-world-event and timeframe
            # agreement. Missing fields default to False (conservative) so a
            # model that ignores the new schema can never produce an accept.
            same_event = resp.get("same_real_world_event") is True
            timeframes_match = resp.get("timeframes_match") is True
            # Polarity: prompt asks the model to return false when the pair
            # is inverted (e.g. "Will X happen" vs "Will X fail to happen").
            # An inverted pair is still usable — we just flip YES↔NO at
            # injection time. Presence of this field is required; absence
            # means the model ignored the schema, which is treated as a reject
            # downstream (schema version bump forces re-verify).
            polarity_field = resp.get("yes_polarity_aligned")
            polarity_seen = isinstance(polarity_field, bool)
            polarity_aligned = polarity_field is True if polarity_seen else True

            # Sports moneyline projection: 3-way PM ↔ 2-way Limitless only
            # traded when the model can state how the draw leg resolves. The
            # prefilter lets these pairs through; downstream injection synthesises
            # the appropriate linear relation. We refuse "unclear" outright.
            raw_draw = resp.get("draw_rule")
            draw_rule = raw_draw if isinstance(raw_draw, str) else ""
            draw_rule_valid = draw_rule in VALID_DRAW_RULES
            draw_rule_explicit = draw_rule_valid or draw_rule == "unclear"
            home_outcome_name_raw = resp.get("home_outcome_name")
            home_outcome_name = (
                home_outcome_name_raw.strip()
                if isinstance(home_outcome_name_raw, str)
                else ""
            )

            # 3-way projection requires a non-"none" draw_rule AND a home name;
            # binary-vs-binary requires "none" (anything else indicates model
            # confusion about the pair shape).
            pm_is_3way = len(p_market.outcomes) == 3
            lm_outcome_count = _limitless_outcome_count(l_market)
            # lm_outcome_count is None when the raw market didn't ship an
            # outcomes list; treat that as "binary or unknown" so a 3-way PM
            # pair can still land if the LLM identifies a valid projection.
            lm_compatible_with_binary = lm_outcome_count in (2, None)
            is_threeway_pair = pm_is_3way and lm_compatible_with_binary
            if is_threeway_pair:
                projection_ok = (
                    draw_rule in ("draw_is_no", "double_chance")
                    and bool(home_outcome_name)
                )
            else:
                projection_ok = draw_rule == "none"

            # Accept when match semantics hold AND the model explicitly
            # returned both a polarity verdict and a draw-rule verdict
            # appropriate to the pair shape.
            accept = (
                is_match
                and same_event
                and timeframes_match
                and polarity_seen
                and draw_rule_explicit
                and projection_ok
            )

            if accept:
                stored_draw_rule = draw_rule if draw_rule_valid else "none"
                self._accepted[p_market.market_id] = _AcceptedEntry(
                    limitless_id=l_id,
                    similarity=round(sim, 4),
                    reasoning=reasoning,
                    verified_at=_now_iso(),
                    polarity_aligned=polarity_aligned,
                    draw_rule=stored_draw_rule,
                    home_outcome_name=home_outcome_name,
                )
                pipeline_stats["llm_matches_confirmed"] += 1
                pipeline_stats["new_pairs_found"] += 1
                # Log Polymarket resolution-source string so operators can grep
                # for suspicious oracle mismatches post-hoc (see §2.3 downgrade).
                pipeline_stats["matched_pairs_detail"].append({
                    "polymarket": p_market.question[:80],
                    "limitless": (l_market.get("title", "") or "")[:80],
                    "similarity": round(sim, 3),
                    "method": "LLM Verified",
                    "polarity_aligned": polarity_aligned,
                    "draw_rule": stored_draw_rule,
                    "home_outcome_name": home_outcome_name,
                    "polymarket_resolution_source": p_market.resolution_source or "",
                    "limitless_resolution_source": (
                        l_market.get("resolutionSource")
                        or l_market.get("rules", "")[:120]
                        or ""
                    ),
                })
                await _notify_mapped_pair({
                    "polymarket_question": p_market.question,
                    "limitless_title": l_market.get("title", ""),
                    "polymarket_id": p_market.market_id,
                    "limitless_id": l_id,
                    "similarity": round(sim, 2),
                    "polarity_aligned": polarity_aligned,
                    "draw_rule": stored_draw_rule,
                })
                polarity_tag = "" if polarity_aligned else " [INVERTED]"
                draw_tag = (
                    "" if stored_draw_rule == "none" else f" [draw={stored_draw_rule}]"
                )
                logger.info(
                    f"MATCH [{sim:.2f}]{polarity_tag}{draw_tag}: "
                    f"{p_market.question[:40]} == {l_market.get('title', '')[:40]}"
                )
            else:
                key = f"{p_market.market_id}|{l_id}"
                # Compose a diagnostic reason that makes the schema-gate visible
                # in the cache. Helps operators spot cases where is_match=true
                # was overridden by a failing sub-flag.
                if is_match and not same_event:
                    gate = "gated: same_real_world_event=false"
                elif is_match and not timeframes_match:
                    gate = "gated: timeframes_match=false"
                elif is_match and not polarity_seen:
                    gate = "gated: yes_polarity_aligned missing from response"
                elif is_match and draw_rule == "unclear":
                    gate = "gated: draw_rule=unclear (ambiguous rules)"
                elif is_match and is_threeway_pair and not projection_ok:
                    gate = "gated: 3-way pair missing draw_rule/home_outcome_name"
                elif is_match and not is_threeway_pair and draw_rule not in ("none", ""):
                    gate = f"gated: unexpected draw_rule={draw_rule!r} for binary pair"
                else:
                    gate = ""
                merged_reason = f"{gate} | {reasoning}" if gate else reasoning
                self._rejected[key] = _RejectedEntry(
                    similarity=round(sim, 4),
                    reasoning=merged_reason,
                    rejected_at=_now_iso(),
                )
                pipeline_stats["llm_matches_rejected"] += 1
                pipeline_stats["new_rejections_cached"] += 1

        if pipeline_stats["new_pairs_found"] or pipeline_stats["new_rejections_cached"]:
            self._save_cache()

        pipeline_stats["total_pairs_after"] = len(self._accepted)
        logger.info(
            f"Pipeline complete. {pipeline_stats['new_pairs_found']} new pairs, "
            f"{pipeline_stats['new_rejections_cached']} new rejections cached. "
            f"Total accepted: {len(self._accepted)}"
        )
        return self.mapped_pairs, pipeline_stats
