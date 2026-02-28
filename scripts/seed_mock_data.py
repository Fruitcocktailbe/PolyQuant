import asyncio
import json
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from polyquant.data.constraint_store import (
    ConstraintStore, 
    ConstraintManifest, 
    StoredConstraint, 
    StoredDependency
)

async def seed():
    print("🌱 Seeding Mock Data for PolyQuant...")
    store = ConstraintStore()
    
    # 1. Cluster: US Election 2024 (Classic Arbitrage Example)
    manifest = ConstraintManifest(
        cluster_id="election_2024_001",
        topic="US Presidential Election 2024",
        market_ids=["mkt_pres_winner", "mkt_party_winner"],
        constraints=[
            StoredConstraint(
                constraint_id="c1",
                description="Presidential winner must be from the winning party",
                coefficients={
                    "mkt_pres_winner:trump": 1.0,
                    "mkt_party_winner:gop": -1.0
                },
                rhs=0.0,
                confidence=1.0,
                reasoning="Logical identity: Donald Trump is a Republican.",
                source_markets=["mkt_pres_winner", "mkt_party_winner"]
            ),
            StoredConstraint(
                constraint_id="c2",
                description="Total probability of party winners must be 1.0",
                coefficients={
                    "mkt_party_winner:gop": 1.0,
                    "mkt_party_winner:dem": 1.0
                },
                rhs=1.0,
                confidence=1.0,
                reasoning="Mutual exclusivity of primary parties in outcome.",
                source_markets=["mkt_party_winner"]
            )
        ],
        dependencies=[
            StoredDependency(
                source_market_id="mkt_pres_winner",
                source_outcome="trump",
                target_market_id="mkt_party_winner",
                target_outcome="gop",
                relationship="implies",
                confidence=1.0
            )
        ],
        correlations=[
            {
                "leader_id": "mkt_pres_winner",
                "laggard_id": "mkt_party_winner",
                "correlation": 0.98
            }
        ]
    )
    
    await store.save_manifest(manifest)
    
    # 2. Cluster: Federal Reserve Decision
    fed_manifest = ConstraintManifest(
        cluster_id="fed_june_2025",
        topic="Fed Interest Rate Decision (June 2025)",
        market_ids=["mkt_fed_pause", "mkt_fed_hike"],
        constraints=[
            StoredConstraint(
                constraint_id="fed_c1",
                description="Sum to 1.0",
                coefficients={
                    "mkt_fed_pause:yes": 1.0,
                    "mkt_fed_hike:yes": 1.0
                },
                rhs=1.0,
                confidence=1.0,
                reasoning="Fed will either hike or pause.",
                source_markets=["mkt_fed_pause", "mkt_fed_hike"]
            )
        ]
    )
    
    await store.save_manifest(fed_manifest)
    
    print("✅ Seed complete. Check .polyquant/constraints/")

if __name__ == "__main__":
    asyncio.run(seed())
