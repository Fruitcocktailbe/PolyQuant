import asyncio
import json
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from polyquant.agents.discovery import DiscoveryAgent, MarketCluster
from polyquant.agents.logic_architect import LogicArchitect, AnalysisResult
from polyquant.data.market_models import Market, Outcome, MarketDependency

async def test_discovery_prompt_parsing():
    print("Testing Discovery Agent Prompt Parsing...")
    agent = DiscoveryAgent()
    
    # Mock Gemini response for Event Clustering
    mock_response_text = """
    ```json
    {
        "clusters": [
            {
                "topic": "US Election 2024",
                "market_ids": ["m1", "m2"],
                "potential_dependencies": [
                    "Market m1 (Trump Win) implies Market m2 (GOP Win)"
                ],
                "confidence": 0.95
            }
        ]
    }
    ```
    """
    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=MagicMock(text=mock_response_text))
    agent._genai_model = mock_model
    
    # Mock markets
    markets = [
        Market(market_id="m1", question="Will Trump win?", volume=1000),
        Market(market_id="m2", question="Will GOP win?", volume=500),
    ]
    
    # Test _cluster_markets
    print("Running _cluster_markets...")
    try:
        clusters = await agent._cluster_markets(markets)
        print(f"Clusters found: {len(clusters)}")
        if len(clusters) != 1:
             print(f"FAILURE: Expected 1 cluster, got {len(clusters)}")
             return
             
        if clusters[0].topic != "US Election 2024":
            print(f"FAILURE: Expected topic 'US Election 2024', got '{clusters[0].topic}'")
            return
            
        if len(clusters[0].markets) != 2:
            print(f"FAILURE: Expected 2 markets in cluster, got {len(clusters[0].markets)}")
            return
            
        print("Discovery Agent Prompt Parsing: PASSED")
    except Exception as e:
        print(f"CRITICAL ERROR during Discovery parsing: {e}")
        import traceback
        traceback.print_exc()

async def test_logic_architect_prompt_parsing():
    print("Testing Logic Architect Prompt Parsing...")
    architect = LogicArchitect()
    
    # Mock Gemini response for Logic Scout with reasoning (as dict, mocking extraction)
    mock_response_dict = {
        "dependencies": [
            {
                "source_market_id": "m1",
                "source_outcome": "Yes",
                "target_market_id": "m2",
                "target_outcome": "Yes",
                "relationship": "MUTUALLY_EXCLUSIVE", 
                "confidence": 0.99,
                "reasoning": "Events cannot happen simultaneously"
            }
        ],
        "constraints": []
    }
    
    cluster = MarketCluster(
        topic="Test Cluster",
        markets=[
            Market(market_id="m1", question="Q1"),
            Market(market_id="m2", question="Q2")
        ]
    )
    
    # Test _parse_response directly with the dict
    print(f"Testing with response: {mock_response_dict}")
    try:
        result = architect._parse_response(mock_response_dict, cluster)
        print(f"Parsed Result: {result}")
        
        if len(result.dependencies) != 1:
            print(f"FAILURE: Expected 1 dependency, got {len(result.dependencies)}")
            print(f"Dependencies found: {result.dependencies}")
            return

        dep = result.dependencies[0]
        if dep.relationship != "MUTUALLY_EXCLUSIVE":
            print(f"FAILURE: Expected MUTUALLY_EXCLUSIVE, got {dep.relationship}")
            return
            
        if dep.reasoning != "Events cannot happen simultaneously":
            print(f"FAILURE: Expected specific reasoning, got '{dep.reasoning}'")
            return
            
        print("Logic Architect Prompt Parsing: PASSED")
    except Exception as e:
        print(f"CRITICAL ERROR during parsing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_discovery_prompt_parsing())
    asyncio.run(test_logic_architect_prompt_parsing())
