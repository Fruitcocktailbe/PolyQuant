"""
Exchange Matcher Agent

Implements a 3-stage funnel to match Polymarket and Limitless markets:
1. Liquidity Pre-filtering (removes dead/resolved/illiquid markets)
2. Vector Embeddings (TF-IDF/Cosine Similarity for top-N candidates)
3. LLM Semantic Verification (final 1:1 match confirmation)

This matched schema is cached to avoid repeated LLM API costs.
"""

import os
import json
import asyncio
from typing import Any, Dict, List
from pathlib import Path
from polyquant.api.server import monitor

import numpy as np
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

from polyquant.data.polymarket_client import PolymarketClient
from polyquant.data.limitless_client import LimitlessClient
from polyquant.utils.llm_client import call_llm_json
from polyquant.utils import get_logger, config

logger = get_logger(__name__)

CACHE_DIR = Path(".polyquant/constraints")
CACHE_FILE = CACHE_DIR / "market_pairs.json"

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

class ExchangeMatcher:
    def __init__(self):
        self.mapped_pairs: Dict[str, str] = self._load_cache()
        
    def _load_cache(self) -> Dict[str, str]:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load market pairs cache: {e}")
        return {}

    def _save_cache(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(self.mapped_pairs, f, indent=2)

    async def run_matching_pipeline(self) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Runs the 3-stage funnel to find new cross-exchange arb pairs.
        
        Returns:
            Tuple of (mapped_pairs dict, pipeline_stats dict)
        """
        if SentenceTransformer is None:
            logger.error("sentence-transformers is not installed. Run `pip install sentence-transformers`")
            return self.mapped_pairs, {"error": "sentence-transformers not installed"}

        # Stage 1: Liquidity Pre-filtering
        logger.info("Stage 1: Fetching and filtering active markets from both exchanges...")
        
        # Fetch ALL Polymarket markets via pagination (not just first page)
        poly_markets = []
        async with PolymarketClient() as p_client:
            offset = 0
            page_size = 500
            while True:
                batch, raw_count = await p_client.get_active_markets(
                    limit=page_size, offset=offset, min_liquidity=2500.0
                )
                poly_markets.extend(batch)
                offset += page_size
                if raw_count < page_size:
                    break  # No more pages
        
        poly_fetched_total = len(poly_markets)
            
        limit_markets_raw = []
        try:
            async with LimitlessClient() as l_client:
                # Fetch all active markets via pagination (larger page size for speed)
                limit_markets_raw = await l_client.get_markets(limit=20)
        except Exception as e:
            logger.error(f"LimitlessClient context failed: {e}")
            # Fallback: Try fetching markets with a raw HTTP call (no signing/nonce needed)
            try:
                import httpx
                async with httpx.AsyncClient(timeout=15.0) as raw_client:
                    page = 1
                    while True:
                        resp = await raw_client.get(
                            f"{config.limitless_api_url}/markets/active",
                            params={"limit": 20, "page": page}
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        batch = data.get("data", []) or data.get("markets", [])
                        if not batch:
                            break
                        limit_markets_raw.extend(batch)
                        page += 1
                        if len(batch) < 20:
                            break
                logger.info(f"Fallback fetch succeeded: {len(limit_markets_raw)} Limitless markets")
            except Exception as fallback_err:
                logger.error(f"Fallback Limitless fetch also failed: {fallback_err}")
            
        # Filter Limitless for minimum activity ($2500 in volume or liquidity)
        # Note: Limitless returns raw USDC ints (6 decimals) AND formatted dollar strings
        limit_markets = []
        for m in limit_markets_raw:
            try:
                vol = float(m.get("volumeFormatted", 0) or m.get("volume", 0))
                liq = float(m.get("liquidityFormatted", 0) or m.get("liquidity", 0))
                # volumeFormatted is in dollars; raw volume is in micro-USDC
                # If value is huge (>1M), it's raw — divide by 1e6
                if vol > 1_000_000:
                    vol = vol / 1_000_000
                if liq > 1_000_000:
                    liq = liq / 1_000_000
                if vol > 2500.0 or liq > 2500.0:
                    limit_markets.append(m)
            except (ValueError, TypeError):
                continue
        
        logger.info(f"Stage 1 Complete: {len(poly_markets)} Polymarket | {len(limit_markets)} Limitless targets.")

        # Track stats for the report
        pipeline_stats: Dict[str, Any] = {
            "polymarket_fetched": poly_fetched_total,
            "polymarket_after_filter": len(poly_markets),
            "limitless_fetched": len(limit_markets_raw),
            "limitless_after_filter": len(limit_markets),
            "limitless_discarded": len(limit_markets_raw) - len(limit_markets),
            "already_mapped": len(self.mapped_pairs),
            "llm_verifications_sent": 0,
            "llm_matches_confirmed": 0,
            "llm_matches_rejected": 0,
            "llm_errors": 0,
            "vector_fallbacks": 0,
            "new_pairs_found": 0,
            "total_pairs_after": 0,
            "matched_pairs_detail": [],
        }
        
        if not poly_markets or not limit_markets:
            pipeline_stats["total_pairs_after"] = len(self.mapped_pairs)
            return self.mapped_pairs, pipeline_stats

        # Prepare corpuses for TF-IDF
        p_docs = [f"{m.question} {m.description}" for m in poly_markets]
        l_docs = [f"{m.get('title', '')} {m.get('description', '')}" for m in limit_markets]

        # Stage 2: Vector Embeddings (Dense Semantic Embeddings)
        from polyquant.utils.llm_client import get_llm_client
        if not config.enable_semantic_matching or get_llm_client() is None:
            reason = "DISABLED by config" if not config.enable_semantic_matching else "LLM KEY MISSING"
            logger.info(f"Stage 2/3: Semantic Matching skipped ({reason}). Returning Stage 1 results only.")
            return self.mapped_pairs

        logger.info("Stage 2: Loading Semantic Embedding Model (all-MiniLM-L6-v2)...")
        logger.info("Note: This may take several minutes if the model is being downloaded for the first time.")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            logger.warning("Proceeding with empty mappings due to model load failure.")
            return self.mapped_pairs
        
        # Encode corpuses into semantic vectors
        logger.info(f"Encoding {len(p_docs)} Polymarket and {len(l_docs)} Limitless document embeddings...")
        p_embeddings = model.encode(p_docs, convert_to_tensor=True)
        l_embeddings = model.encode(l_docs, convert_to_tensor=True)
        
        # Compute cosine similarity
        logger.info("Computing similarity matrix...")
        similarity_matrix = util.cos_sim(p_embeddings, l_embeddings).cpu().numpy()
        logger.info("Similarity matrix computed.")
        
        # Stage 3: LLM Semantic Verification
        logger.info("Stage 3: Verifying Top-N neighbors using LLM Semantic Matching (Concurrent Async)...")
        new_matches = 0
        
        # Prepare all LLM evaluation tasks
        eval_tasks = []
        task_meta = []
        
        for p_idx, p_market in enumerate(poly_markets):
            if p_market.market_id in self.mapped_pairs:
                continue # Already mapped
                
            # Get top 3 indices for this Polymarket market
            top_3_indices = np.argsort(similarity_matrix[p_idx])[-3:][::-1]
            
            # Require at least a 0.52 semantic similarity score to test with LLM
            for l_idx in top_3_indices:
                sim_score = float(similarity_matrix[p_idx][l_idx])
                if sim_score < 0.52:
                    continue
                    
                l_market = limit_markets[l_idx]
                l_id = l_market.get("id") or l_market.get("marketId")
                if not l_id:
                    continue
                    
                # Format Verify Prompt (includes resolution criteria)
                prompt = LLM_VERIFY_PROMPT.format(
                    p_q=p_market.question,
                    p_d=p_market.description,
                    p_res=p_market.resolution_source or "Not specified",
                    p_end=str(p_market.end_date or "Not specified"),
                    l_q=l_market.get('title', ''),
                    l_d=l_market.get('description', ''),
                    l_res=l_market.get('resolutionSource', '') or l_market.get('rules', '') or l_market.get('description', '')[:200] or "Not specified",
                    l_end=l_market.get('expirationDate', '') or l_market.get('expirationTimestamp', '') or "Not specified",
                )
                
                # Append to batch
                eval_tasks.append(
                    asyncio.to_thread(
                        call_llm_json,
                        prompt=prompt,
                        system_prompt="Answer JSON only.",
                        temperature=0.1
                    )
                )
                
                # Save metadata for matching back the result
                task_meta.append({
                    "p_market": p_market,
                    "sim_score": sim_score,
                    "l_id": l_id,
                    "l_title": l_market.get('title', '')
                })
                
        if not eval_tasks:
            logger.info("No new pairs met the semantic similarity threshold for LLM verification.")
            pipeline_stats["total_pairs_after"] = len(self.mapped_pairs)
            return self.mapped_pairs, pipeline_stats
            
        pipeline_stats["llm_verifications_sent"] = len(eval_tasks)
        logger.info(f"Firing {len(eval_tasks)} LLM verification requests (Rate limited to 20 RPM)...")

        # OpenRouter free tier limits to 20 requests per minute
        # We will use a semaphore of 2 and a sleep to restrict throughput
        sem = asyncio.Semaphore(2)

        async def controlled_llm_call(task):
            async with sem:
                try:
                    res = await task
                    # Add delay to stay under ~20 RPM (3 seconds per request across 2 workers)
                    await asyncio.sleep(6.0)
                    return res
                except Exception as e:
                    return e

        rate_limited_tasks = [controlled_llm_call(t) for t in eval_tasks]
        results = await asyncio.gather(*rate_limited_tasks, return_exceptions=True)
        
        # Process results, grouping by Polymarket market_id so we only map the first true match
        processed_p_market_ids = set()
        
        for meta, resp in zip(task_meta, results):
            p_market = meta["p_market"]
            
            if p_market.market_id in processed_p_market_ids:
                continue # We already mapped this Polymarket event from a different Limitless suggestion
                
            is_match = False
            match_reason = "LLM Verified"

            if isinstance(resp, Exception) or resp is None:
                # LLM FAILED (400 error or timeout) - Check for vector fallback
                pipeline_stats["llm_errors"] += 1
                if meta["sim_score"] > 0.92:
                    logger.warning(
                        "LLM FAILED - Using High-Confidence Vector Fallback (>0.92)",
                        p_question=p_market.question[:30],
                        l_title=meta['l_title'][:30],
                        score=meta['sim_score']
                    )
                    is_match = True
                    match_reason = "Vector Fallback (LLM Failed)"
                    pipeline_stats["vector_fallbacks"] += 1
                else:
                    logger.error(f"LLM evaluation failed and score ({meta['sim_score']:.2f}) too low for fallback: {resp}")
                    continue
            else:
                is_match = resp.get("is_match") is True
                match_reason = resp.get("reasoning", "LLM Verified")

            if is_match:
                logger.info(f"MATCH FOUND [{meta['sim_score']:.2f}] ({match_reason}): {p_market.question[:30]}... == LIMITLESS {meta['l_title'][:30]}...")
                self.mapped_pairs[p_market.market_id] = meta["l_id"]
                processed_p_market_ids.add(p_market.market_id)
                new_matches += 1
                pipeline_stats["llm_matches_confirmed"] += 1
                pipeline_stats["matched_pairs_detail"].append({
                    "polymarket": p_market.question[:80],
                    "limitless": meta["l_title"][:80],
                    "similarity": round(meta["sim_score"], 3),
                    "method": match_reason,
                })
            else:
                pipeline_stats["llm_matches_rejected"] += 1
                
                # Update UI Monitor with UI-friendly list
                ui_pair = {
                    "polymarket_question": p_market.question,
                    "limitless_title": meta["l_title"],
                    "polymarket_id": p_market.market_id,
                    "limitless_id": meta["l_id"],
                    "similarity": round(meta["sim_score"], 2)
                }
                
                # Add to existing list in state safely
                current_pairs = list(monitor.state.mapped_pairs)
                current_pairs.append(ui_pair)
                asyncio.create_task(monitor.update_status(mapped_pairs=current_pairs))
                    
        if new_matches > 0:
            self.save_cache()
            
        logger.info(f"Pipeline complete. Identified {new_matches} new dual-arb pairs. Total cached: {len(self.mapped_pairs)}")
        pipeline_stats["new_pairs_found"] = new_matches
        pipeline_stats["total_pairs_after"] = len(self.mapped_pairs)
        return self.mapped_pairs, pipeline_stats

    def save_cache(self):
        self._save_cache()
