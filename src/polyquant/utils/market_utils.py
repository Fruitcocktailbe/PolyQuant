
"""
Market-related utility functions.
"""
from typing import Optional

def extract_market_id(outcome_id: str) -> str:
    """
    Extract the market ID from an outcome ID.
    
    Format expected: "marketID_outcomeID" or just "marketID" if no underscore.
    
    Args:
        outcome_id: The outcome ID string
        
    Returns:
        The market ID part, or empty string if invalid
    """
    if not outcome_id:
        return ""
        
    if "_" in outcome_id:
        return outcome_id.split("_")[0]
        
    # If no underscore, assume it might be just the market ID or invalid
    # For safety, we return it as is, or should we iterate?
    # Context dependent.
    return outcome_id
