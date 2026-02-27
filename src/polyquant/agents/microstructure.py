import logging
from datetime import datetime

from polyquant.data.market_models import OrderBook, MicrostructureSignal

logger = logging.getLogger(__name__)

class MicrostructureAgent:
    """
    Analyzes order book dynamics to predict short-term price moves.
    Focuses on Order Imbalance and Spread analysis.
    """
    
    def analyze(self, order_book: OrderBook) -> MicrostructureSignal:
        """
        Calculate microstructure metrics from an order book.
        
        Args:
            order_book: Snapshot of current bids/asks
            
        Returns:
            MicrostructureSignal with calculated metrics
        """
        signal = MicrostructureSignal()
        
        if not order_book.bids or not order_book.asks:
            return signal
            
        best_bid = float(order_book.bids[0].price)
        best_ask = float(order_book.asks[0].price)
        
        # 1. Spread
        signal.spread = best_ask - best_bid
        
        # 2. Imbalance (Volume Weighted)
        # We look at top 3 levels for immediate pressure
        bid_vol = sum(float(x.size) for x in order_book.bids[:3])
        ask_vol = sum(float(x.size) for x in order_book.asks[:3])
        
        if bid_vol + ask_vol > 0:
            # Range: -1 (Full Sell Pressure) to +1 (Full Buy Pressure)
            signal.imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
        # 3. Weighted Midpoint
        # If imbalance is high, wmid shifts towards the heavy side
        # Formula: (BestBid * AskVol + BestAsk * BidVol) / (BidVol + AskVol)
        if bid_vol + ask_vol > 0:
            signal.weighted_midpoint = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
        else:
            signal.weighted_midpoint = (best_bid + best_ask) / 2
            
        return signal
