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
from datetime import datetime, timezone
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
)

logger = get_logger(__name__)

CACHE_DIR = Path(".polyquant/constraints")
CACHE_FILE = CACHE_DIR / "market_pairs.json"

MIN_LIQUIDITY = 2500.0
LIMITLESS_PAGE_SIZE = 20
EMBEDDING_TOP_K = 5
SIMILARITY_FLOOR = 0.50      # absolute floor — below this we don't even consider
SIMILARITY_VERIFY = 0.65     # threshold for sending to LLM
LLM_RPS_SEMAPHORE = 2
LLM_PACING_SECONDS = 1.0
ENCODE_CHUNK_SIZE = 500      # bounds peak RAM on the 2 GB Lightsail VM

LLM_VERIFY_PROMPT = """
You are an arbitrage trading engine.
Your task is to determine if two prediction markets represent the SAME real-world outcome.
They must be logically equivalent, even if they use different wording or different standard sources.

Examples of MATCH:
- "Will BTC hit $100k in 2025?" vs "Will Bitcoin reach $100k before 2026?" (Same outcome)
- "Trump to win election" vs "Donald Trump victor in 2024" (Same logical outcome)
- "Fed cuts rates in Sept" vs "Federal Reserve 25bps+ rate cut by Sept" (Equivalent financial outcome)

Examples of NOT A MATCH:
- "Who will win the election?" vs "Will Trump win the election?" (Different structure: multiple choice vs binary)
- "Will BTC hit 100k in May?" vs "Will BTC hit 100k in June?" (Different timeframes)
- "Will ETH be above $3000?" vs "Will ETH be above $3000 OR BTC above $100k?" (One has extra conditions)

Polymarket Question: {p_q}
Polymarket Description: {p_d}
Polymarket Resolution Source: {p_res}
Polymarket End Date: {p_end}

Limitless Candidate Question: {l_q}
Limitless Candidate Description: {l_d}
Limitless Resolution Source: {l_res}
Limitless End Date: {l_end}

Respond ONLY in JSON. Return a boolean "is_match" and a short "reasoning".
{{
    "is_match": true/false,
    "reasoning": "..."
}}
"""


@dataclass
class _AcceptedEntry:
    limitless_id: str
    similarity: float
    reasoning: str
    verified_at: str

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

    def _load_cache(self) -> None:
        if not CACHE_FILE.exists():
            return
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load market pairs cache: {e}")
            return

        if isinstance(data, dict) and ("accepted" in data or "rejected" in data):
            for pid, entry in (data.get("accepted") or {}).items():
                try:
                    lid = entry["limitless_id"]
                    # Drop legacy entries that stored Limitless's numeric `id`
                    # instead of the URL `slug`. The orderbook endpoint expects
                    # a slug, so any all-digit value here is a stale entry from
                    # before the slug fix and would 404 downstream.
                    if not isinstance(lid, str) or lid.isdigit():
                        continue
                    self._accepted[pid] = _AcceptedEntry(
                        limitless_id=lid,
                        similarity=float(entry.get("similarity", 0.0)),
                        reasoning=entry.get("reasoning", ""),
                        verified_at=entry.get("verified_at", ""),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            for key, entry in (data.get("rejected") or {}).items():
                try:
                    self._rejected[key] = _RejectedEntry(
                        similarity=float(entry.get("similarity", 0.0)),
                        reasoning=entry.get("reasoning", ""),
                        rejected_at=entry.get("rejected_at", ""),
                    )
                except (TypeError, ValueError):
                    continue
        elif isinstance(data, dict):
            # Legacy flat format: {poly_id: limitless_id}
            for pid, lid in data.items():
                if isinstance(lid, str):
                    self._accepted[pid] = _AcceptedEntry(
                        limitless_id=lid,
                        similarity=0.0,
                        reasoning="legacy import",
                        verified_at="",
                    )
        logger.info(
            f"Loaded matcher cache: {len(self._accepted)} accepted, {len(self._rejected)} rejected"
        )

    def _save_cache(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "accepted": {pid: e.to_dict() for pid, e in self._accepted.items()},
            "rejected": {k: e.to_dict() for k, e in self._rejected.items()},
        }
        tmp = CACHE_FILE.with_suffix(CACHE_FILE.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, CACHE_FILE)

    def save_cache(self) -> None:
        self._save_cache()

    # ----------------------------------------------------------- fetch markets

    async def _fetch_polymarket(self) -> tuple[list[Any], int]:
        markets: list[Any] = []
        async with PolymarketClient() as p_client:
            offset = 0
            page_size = 500
            while True:
                batch, raw_count = await p_client.get_active_markets(
                    limit=page_size, offset=offset, min_liquidity=MIN_LIQUIDITY
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

        # ---------- Layer 0: fetch + harmonized liquidity filter ----------
        logger.info("Layer 0: Fetching markets from both exchanges...")
        poly_markets, poly_fetched_total = await self._fetch_polymarket()
        limit_markets_raw = await self._fetch_limitless()

        limit_markets: list[dict] = []
        for m in limit_markets_raw:
            vol, liq = _limitless_dollars(m)
            if max(vol, liq) >= MIN_LIQUIDITY:
                limit_markets.append(m)

        logger.info(
            f"Layer 0 complete: {len(poly_markets)} Polymarket | {len(limit_markets)} Limitless after filter."
        )

        pipeline_stats: Dict[str, Any] = {
            "polymarket_fetched": poly_fetched_total,
            "polymarket_after_filter": len(poly_markets),
            "limitless_fetched": len(limit_markets_raw),
            "limitless_after_filter": len(limit_markets),
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
            # Prefilter
            if not compatible(poly_fps[p_idx], limit_fps[l_idx]):
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
                try:
                    res = await asyncio.to_thread(
                        call_llm_json,
                        prompt=prompt,
                        system_prompt="Answer JSON only.",
                        temperature=0.1,
                        model=config.llm_model_matcher,
                    )
                    await asyncio.sleep(LLM_PACING_SECONDS)
                except Exception as e:
                    await asyncio.sleep(LLM_PACING_SECONDS)
                    res = e
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

            if isinstance(resp, Exception) or resp is None:
                pipeline_stats["llm_errors"] += 1
                logger.warning(
                    "LLM verification errored — leaving pair unverified",
                    poly=p_market.question[:40],
                    limitless=(l_market.get("title", "") or "")[:40],
                    err=str(resp) if resp is not None else "None",
                )
                continue

            is_match = resp.get("is_match") is True
            reasoning = resp.get("reasoning", "")

            if is_match:
                self._accepted[p_market.market_id] = _AcceptedEntry(
                    limitless_id=l_id,
                    similarity=round(sim, 4),
                    reasoning=reasoning,
                    verified_at=_now_iso(),
                )
                pipeline_stats["llm_matches_confirmed"] += 1
                pipeline_stats["new_pairs_found"] += 1
                pipeline_stats["matched_pairs_detail"].append({
                    "polymarket": p_market.question[:80],
                    "limitless": (l_market.get("title", "") or "")[:80],
                    "similarity": round(sim, 3),
                    "method": "LLM Verified",
                })
                await _notify_mapped_pair({
                    "polymarket_question": p_market.question,
                    "limitless_title": l_market.get("title", ""),
                    "polymarket_id": p_market.market_id,
                    "limitless_id": l_id,
                    "similarity": round(sim, 2),
                })
                logger.info(
                    f"MATCH [{sim:.2f}]: {p_market.question[:40]} == {l_market.get('title', '')[:40]}"
                )
            else:
                key = f"{p_market.market_id}|{l_id}"
                self._rejected[key] = _RejectedEntry(
                    similarity=round(sim, 4),
                    reasoning=reasoning,
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
