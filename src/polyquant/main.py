"""
PolyQuant 2.0 Main Orchestrator

This is the main entry point that coordinates all agents in the pipeline.

THE PIPELINE:
-------------
1. Discovery Agent (GPT-4o) -> Scans markets, finds clusters
2. Logic Architect (DeepSeek-R1) -> Analyzes dependencies
3. Validator Agent (o1-preview) -> Verifies constraints
4. Solver Oracle (SCIP) -> Computes optimal trades
5. Executor (Rust) -> Executes trades [NOT YET IMPLEMENTED]

EXECUTION MODES:
----------------
- paper: Simulation mode, no real trades
- live: Real trading (requires confirmation)

USAGE:
------
    # From command line
    python -m polyquant.main
    
    # Or programmatically
    from polyquant.main import PolyQuantOrchestrator
    
    async def run():
        orchestrator = PolyQuantOrchestrator()
        async with orchestrator:
            await orchestrator.run_pipeline()
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Any

from polyquant.agents import (
    DiscoveryAgent,
    LogicArchitect,
    MarketCluster,
    ValidatorAgent,
)
from polyquant.data import PolymarketClient
from polyquant.risk import KillSwitch, PositionSizer
from polyquant.solver import ArbitrageDetector, SCIPSolver
from polyquant.utils import config, get_logger

logger = get_logger(__name__)


class PolyQuantOrchestrator:
    """
    Main orchestrator that coordinates all agents in the arbitrage pipeline.
    
    The orchestrator runs the full pipeline:
    1. Discovery: Find related markets
    2. Reasoning: Identify logical dependencies
    3. Verification: Validate constraints
    4. Optimization: Find optimal trades
    5. Execution: Submit trades (paper or live mode)
    
    Example:
        orchestrator = PolyQuantOrchestrator()
        
        async with orchestrator:
            # Run once
            await orchestrator.run_pipeline()
            
            # Or run continuously
            await orchestrator.run_continuous(interval_seconds=60)
    """
    
    def __init__(self):
        """Initialize the orchestrator with all components."""
        self.trading_mode = config.trading_mode
        
        # Initialize components (lazy - actual init happens in __aenter__)
        self._discovery: DiscoveryAgent | None = None
        self._logic_architect: LogicArchitect | None = None
        self._validator: ValidatorAgent | None = None
        self._solver: SCIPSolver | None = None
        self._arbitrage_detector: ArbitrageDetector | None = None
        self._polymarket: PolymarketClient | None = None
        
        # Risk management
        self._kill_switch: KillSwitch | None = None
        self._position_sizer: PositionSizer | None = None
        
        # State tracking
        self._is_running = False
        self._pipeline_count = 0
        self._opportunities_found = 0
        self._trades_executed = 0
        
        logger.info(
            "PolyQuantOrchestrator initialized",
            trading_mode=self.trading_mode,
        )
    
    async def __aenter__(self) -> "PolyQuantOrchestrator":
        """Initialize all components."""
        logger.info("Starting PolyQuant 2.0...")
        
        # Initialize agents
        self._discovery = DiscoveryAgent()
        await self._discovery.__aenter__()
        
        self._logic_architect = LogicArchitect()
        await self._logic_architect.__aenter__()
        
        self._validator = ValidatorAgent()
        await self._validator.__aenter__()
        
        # Initialize solver
        self._solver = SCIPSolver()
        self._arbitrage_detector = ArbitrageDetector()
        
        # Initialize Polymarket client
        self._polymarket = PolymarketClient()
        await self._polymarket.__aenter__()
        
        # Initialize risk management
        self._kill_switch = KillSwitch(
            initial_capital=10000,  # TODO: Get from config/user
            on_trigger=self._on_kill_switch_trigger,
        )
        self._position_sizer = PositionSizer(capital=10000)
        
        self._is_running = True
        
        logger.info("PolyQuant 2.0 started successfully")
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Cleanup all components."""
        logger.info("Shutting down PolyQuant 2.0...")
        
        self._is_running = False
        
        if self._discovery:
            await self._discovery.__aexit__(*args)
        if self._logic_architect:
            await self._logic_architect.__aexit__(*args)
        if self._validator:
            await self._validator.__aexit__(*args)
        if self._polymarket:
            await self._polymarket.__aexit__(*args)
        
        logger.info(
            "PolyQuant 2.0 shutdown complete",
            pipelines_run=self._pipeline_count,
            opportunities_found=self._opportunities_found,
            trades_executed=self._trades_executed,
        )
    
    async def run_pipeline(self) -> dict[str, Any]:
        """
        Run the full pipeline once.
        
        Executes all phases in sequence:
        1. Discovery -> Find market clusters
        2. Logic Architect -> Analyze dependencies
        3. Validator -> Verify constraints
        4. Solver -> Find optimal trades
        5. (Future) Executor -> Submit trades
        
        Returns:
            Dict with pipeline results and metrics
        """
        start_time = datetime.utcnow()
        self._pipeline_count += 1
        
        logger.info("Starting pipeline run", run_number=self._pipeline_count)
        
        # Check kill switch
        if self._kill_switch and not self._kill_switch.can_trade():
            logger.warning("Pipeline blocked by kill switch")
            return {"status": "blocked", "reason": "kill_switch"}
        
        results: dict[str, Any] = {
            "run_number": self._pipeline_count,
            "start_time": start_time.isoformat(),
            "status": "running",
        }
        
        try:
            # ================================================================
            # PHASE 1: DISCOVERY
            # ================================================================
            logger.info("Phase 1: Discovery - Scanning markets...")
            
            if not self._discovery:
                raise RuntimeError("Discovery agent not initialized")
            
            clusters = await self._discovery.scan_markets(
                limit=50,
                min_liquidity=1000,
            )
            
            results["discovery"] = {
                "clusters_found": len(clusters),
                "total_markets": sum(len(c.markets) for c in clusters),
            }
            
            if not clusters:
                logger.info("No market clusters found")
                results["status"] = "complete"
                results["outcome"] = "no_clusters"
                return results
            
            # ================================================================
            # PHASE 2: REASONING
            # ================================================================
            logger.info("Phase 2: Reasoning - Analyzing dependencies...")
            
            if not self._logic_architect:
                raise RuntimeError("Logic Architect not initialized")
            
            all_dependencies = []
            all_constraints = []
            
            for cluster in clusters:
                analysis = await self._logic_architect.analyze_cluster(cluster)
                all_dependencies.extend(analysis.dependencies)
                all_constraints.extend(analysis.constraints)
            
            results["reasoning"] = {
                "dependencies_found": len(all_dependencies),
                "constraints_generated": len(all_constraints),
            }
            
            if not all_constraints:
                logger.info("No constraints found")
                results["status"] = "complete"
                results["outcome"] = "no_constraints"
                return results
            
            # ================================================================
            # PHASE 3: VERIFICATION
            # ================================================================
            logger.info("Phase 3: Verification - Validating constraints...")
            
            if not self._validator:
                raise RuntimeError("Validator agent not initialized")
            
            # Validate each cluster's analysis
            validated_constraints = []
            validation_issues = []
            
            for cluster in clusters:
                analysis = await self._logic_architect.analyze_cluster(cluster)
                validated = await self._validator.validate(analysis)
                
                if validated.is_valid:
                    validated_constraints.extend(validated.validated_constraints)
                else:
                    validation_issues.extend(validated.issues)
            
            results["verification"] = {
                "valid_constraints": len(validated_constraints),
                "issues_found": len(validation_issues),
            }
            
            if not validated_constraints:
                logger.info("No constraints passed validation")
                results["status"] = "complete"
                results["outcome"] = "validation_failed"
                return results
            
            # ================================================================
            # PHASE 4: OPTIMIZATION
            # ================================================================
            logger.info("Phase 4: Optimization - Finding arbitrage...")
            
            if not self._polymarket:
                raise RuntimeError("Polymarket client not initialized")
            
            # Get order books for all markets in validated clusters
            # For now, use the first validated analysis
            opportunities = []
            
            for cluster in clusters:
                # Re-run validation for this cluster
                analysis = await self._logic_architect.analyze_cluster(cluster)
                validated = await self._validator.validate(analysis)
                
                if not validated.is_valid:
                    continue
                
                # Get order books
                order_books = {}
                for market in cluster.markets:
                    try:
                        obs = await self._polymarket.get_all_order_books(market)
                        order_books.update(obs)
                    except Exception as e:
                        logger.warning(
                            "Failed to get order book",
                            market_id=market.market_id,
                            error=str(e),
                        )
                
                if not order_books:
                    continue
                
                # Detect arbitrage
                opportunity = await self._arbitrage_detector.detect(
                    validated,
                    order_books,
                    min_profit=10.0,
                )
                
                if opportunity:
                    opportunities.append(opportunity)
            
            results["optimization"] = {
                "opportunities_found": len(opportunities),
                "total_expected_profit": sum(
                    float(o.expected_profit) for o in opportunities
                ),
            }
            
            self._opportunities_found += len(opportunities)
            
            # ================================================================
            # PHASE 5: EXECUTION (Paper/Live)
            # ================================================================
            if opportunities:
                logger.info(
                    "Phase 5: Execution",
                    mode=self.trading_mode,
                    opportunity_count=len(opportunities),
                )
                
                if self.trading_mode == "paper":
                    # Paper trading - just log what we would do
                    for opp in opportunities:
                        logger.info(
                            "[PAPER] Would execute trades",
                            expected_profit=float(opp.expected_profit),
                            trade_count=len(opp.trades),
                        )
                    results["execution"] = {
                        "mode": "paper",
                        "trades_simulated": sum(len(o.trades) for o in opportunities),
                    }
                else:
                    # Live trading - TODO: Implement Rust executor integration
                    logger.warning("Live trading not yet implemented")
                    results["execution"] = {
                        "mode": "live",
                        "status": "not_implemented",
                    }
            
            results["status"] = "complete"
            results["outcome"] = "success"
            
        except Exception as e:
            logger.error("Pipeline failed", error=str(e))
            results["status"] = "error"
            results["error"] = str(e)
        
        # Record timing
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        results["elapsed_seconds"] = elapsed
        
        logger.info(
            "Pipeline run complete",
            run_number=self._pipeline_count,
            status=results["status"],
            elapsed=elapsed,
        )
        
        return results
    
    async def run_continuous(
        self,
        interval_seconds: int = 60,
        max_runs: int | None = None,
    ) -> None:
        """
        Run the pipeline continuously.
        
        Args:
            interval_seconds: Seconds between runs
            max_runs: Maximum number of runs (None = infinite)
        """
        run_count = 0
        
        logger.info(
            "Starting continuous operation",
            interval=interval_seconds,
            max_runs=max_runs,
        )
        
        while self._is_running and (max_runs is None or run_count < max_runs):
            try:
                await self.run_pipeline()
            except Exception as e:
                logger.error("Pipeline run failed", error=str(e))
            
            run_count += 1
            
            if self._is_running and (max_runs is None or run_count < max_runs):
                logger.debug(
                    "Waiting before next run",
                    seconds=interval_seconds,
                )
                await asyncio.sleep(interval_seconds)
        
        logger.info("Continuous operation ended", total_runs=run_count)
    
    def stop(self) -> None:
        """Signal the orchestrator to stop."""
        logger.info("Stop requested")
        self._is_running = False
    
    def _on_kill_switch_trigger(self, event: Any) -> None:
        """Handle kill switch trigger."""
        logger.critical(
            "KILL SWITCH TRIGGERED - STOPPING ALL OPERATIONS",
            event=event,
        )
        self.stop()


async def main() -> None:
    """Main entry point for the application."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                     PolyQuant 2.0                             ║
    ║       Autonomous Arbitrage Extraction System                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Set up signal handlers for graceful shutdown
    orchestrator: PolyQuantOrchestrator | None = None
    
    def signal_handler(sig: int, frame: Any) -> None:
        print("\nShutdown signal received...")
        if orchestrator:
            orchestrator.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        orchestrator = PolyQuantOrchestrator()
        async with orchestrator:
            # Run a single pipeline for now
            # TODO: Add CLI arguments for continuous mode
            result = await orchestrator.run_pipeline()
            
            print("\n" + "=" * 60)
            print("Pipeline Result:")
            print("=" * 60)
            
            for key, value in result.items():
                print(f"  {key}: {value}")
            
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
