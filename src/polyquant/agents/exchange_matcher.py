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

CACHE_DIR = Path(".polyquant")
CACHE_FILE = CACHE_DIR / "market_pairs.json"

LLM_VERIFY_PROMPT = """
You are a financial exchange matching engine.
Your task is to determine if two prediction markets are EXACTLY identical.
They must resolve to the exact same real-world outcome, same timeframe, and same truth source.
Pay special attention to the Resolution Source and End Date — if they differ, they are NOT a match.

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

    async def run_matching_pipeline(self) -> Dict[str, str]:
        """Runs the 3-stage funnel to find new cross-exchange arb pairs."""
        if SentenceTransformer is None:
            logger.error("sentence-transformers is not installed. Run `pip install sentence-transformers`")
            return self.mapped_pairs

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
            
        limit_markets_raw = []
        try:
            async with LimitlessClient() as l_client:
                # Fetch all active markets via pagination (larger page size for speed)
                limit_markets_raw = await l_client.get_markets(limit=100)
        except Exception as e:
            logger.warning(f"Failed to fetch Limitless markets, skipping cross-exchange matching: {e}")
            
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
        
        if not poly_markets or not limit_markets:
            return self.mapped_pairs

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
            
            # Require at least a 0.6 semantic similarity score to test with LLM
            for l_idx in top_3_indices:
                sim_score = float(similarity_matrix[p_idx][l_idx])
                if sim_score < 0.6:
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
            return self.mapped_pairs
            
        logger.info(f"Firing {len(eval_tasks)} LLM verification requests concurrently...")
        results = await asyncio.gather(*eval_tasks, return_exceptions=True)
        
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
                if meta["sim_score"] > 0.92:
                    logger.warning(
                        "LLM FAILED - Using High-Confidence Vector Fallback (>0.92)",
                        p_question=p_market.question[:30],
                        l_title=meta['l_title'][:30],
                        score=meta['sim_score']
                    )
                    is_match = True
                    match_reason = "Vector Fallback (LLM Failed)"
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
        return self.mapped_pairs

    def save_cache(self):
        self._save_cache()
