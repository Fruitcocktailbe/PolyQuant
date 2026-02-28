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
from typing import Any, Dict, List
from pathlib import Path

import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

from polyquant.data.polymarket_client import PolymarketClient
from polyquant.data.limitless_client import LimitlessClient
from polyquant.utils.llm_client import call_llm_json
from polyquant.utils import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(".polyquant")
CACHE_FILE = CACHE_DIR / "market_pairs.json"

LLM_VERIFY_PROMPT = """
You are a financial exchange matching engine.
Your task is to determine if two prediction markets are EXACTLY identical.
They must resolve to the exact same real-world outcome, same timeframe, and same truth source.

Polymarket Question: {p_q}
Polymarket Description: {p_d}

Limitless Candidate Question: {l_q}
Limitless Candidate Description: {l_d}

Respond ONLY in JSON. Return a boolean "is_match" and a short "reasoning".
{
    "is_match": true/false,
    "reasoning": "..."
}
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
        if TfidfVectorizer is None:
            logger.error("scikit-learn is not installed. Run `pip install scikit-learn`")
            return self.mapped_pairs

        # Stage 1: Liquidity Pre-filtering
        logger.info("Stage 1: Fetching and filtering active markets from both exchanges...")
        
        poly_markets = []
        async with PolymarketClient() as p_client:
            poly_markets, _ = await p_client.get_active_markets(limit=1000, min_liquidity=5000.0)
            
        limit_markets_raw = []
        async with LimitlessClient() as l_client:
            # We fetch up to 1000 active markets
            limit_markets_raw = await l_client.get_markets(limit=1000)
            
        # Filter Limitless for decent liquidity/volume (e.g., > 5000 volume)
        limit_markets = [
            m for m in limit_markets_raw 
            if float(m.get("volume", 0)) > 5000.0 or float(m.get("liquidity", 0)) > 5000.0
        ]
        
        logger.info(f"Stage 1 Complete: {len(poly_markets)} Polymarket | {len(limit_markets)} Limitless targets.")
        
        if not poly_markets or not limit_markets:
            return self.mapped_pairs

        # Prepare corpuses for TF-IDF
        p_docs = [f"{m.question} {m.description}" for m in poly_markets]
        l_docs = [f"{m.get('title', '')} {m.get('description', '')}" for m in limit_markets]

        # Stage 2: Vector Embeddings (TF-IDF Cosine Similarity)
        logger.info("Stage 2: Running TF-IDF Vector Embeddings to find Top-3 neighbors...")
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Fit on both corpuses combined, then transform
        all_docs = p_docs + l_docs
        vectorizer.fit(all_docs)
        
        p_matrix = vectorizer.transform(p_docs)
        l_matrix = vectorizer.transform(l_docs)
        
        similarity_matrix = cosine_similarity(p_matrix, l_matrix)
        
        # Stage 3: LLM Semantic Verification
        logger.info("Stage 3: Verifying Top-N neighbors using LLM Semantic Matching...")
        new_matches = 0
        
        for p_idx, p_market in enumerate(poly_markets):
            if p_market.condition_id in self.mapped_pairs:
                continue # Already mapped
                
            # Get top 3 indices for this Polymarket market
            top_3_indices = np.argsort(similarity_matrix[p_idx])[-3:][::-1]
            
            # Require at least a 0.5 similarity score to even test with LLM (saves API costs)
            for l_idx in top_3_indices:
                sim_score = similarity_matrix[p_idx][l_idx]
                if sim_score < 0.5:
                    continue
                    
                l_market = limit_markets[l_idx]
                l_id = l_market.get("id") or l_market.get("marketId")
                if not l_id:
                    continue
                    
                # Call LLM Verify
                prompt = LLM_VERIFY_PROMPT.format(
                    p_q=p_market.question,
                    p_d=p_market.description,
                    l_q=l_market.get('title', ''),
                    l_d=l_market.get('description', '')
                )
                
                resp = call_llm_json(prompt=prompt, system_prompt="Answer JSON only.", temperature=0.1)
                if resp and resp.get("is_match") is True:
                    logger.info(f"MATCH FOUND [{sim_score:.2f}]: {p_market.question[:30]}... == LIMITLESS {l_market.get('title', '')[:30]}...")
                    self.mapped_pairs[p_market.condition_id] = l_id
                    new_matches += 1
                    break # Stop looking for this market once matched
                    
        if new_matches > 0:
            self.save_cache()
            
        logger.info(f"Pipeline complete. Identified {new_matches} new dual-arb pairs. Total cached: {len(self.mapped_pairs)}")
        return self.mapped_pairs

    def save_cache(self):
        self._save_cache()
