# This is the updated _detect_dutching_opportunity method with P1 and P2 improvements
# To integrate: Replace lines 840-998 in fw_solver.py with this implementation

async def _detect_dutching_opportunity(
    self,
    validated: "ValidatedResult",
    order_books: Dict[str, OrderBook]
) -> Optional[ArbitrageOpportunity]:
    """
    Specialized Dutching detection with enhanced robustness.

    Improvements in v0.4.0:
    - P1.1: VWAP slippage enforcement
    - P1.2: ROI calculation for ranking
    - P1.3: Partition size limiting
    - P2.1: Capital reservation across rings
    - P2.2: Dynamic confidence scoring
    - P2.3: Expected unwind cost modeling

    Searches constraints for partitions (e.g., Yes + No = 1) and checks for
    risk-free arbitrage opportunities where sum(best_asks) < 1.0.
    """
    logger.info("Running specialized Dutching detection...", constraint_count=len(validated.validated_constraints))

    # P2.1: Track reserved capital across rings
    reserved_capital = Decimal("0")
    available_capital = Decimal(str(self.position_sizer.capital))

    best_ring_trades = []
    best_ring_profit = Decimal("-1")
    best_ring_markets = []
    best_ring_roi = 0.0

    # Collect all potential rings for ranking
    candidate_rings = []

    # Find "Partition" constraints: sum(z_i) >= 1.0 where all coeffs are 1.0
    # In PolyQuant, a complete set of outcomes is represented as a partition.
    for constraint in validated.validated_constraints:
        logger.debug("Checking constraint for Dutching", desc=constraint.description)
        if not all(c == 1.0 for c in constraint.coefficients.values()):
            logger.debug("Skipping: not all coefficients are 1.0")
            continue
        if abs(constraint.rhs - 1.0) > 1e-6:
            logger.debug("Skipping: RHS is not 1.0", rhs=constraint.rhs)
            continue

        # We found a potential Dutching ring!
        outcome_ids = list(constraint.coefficients.keys())

        # P1.3: Partition size limit (prevent O(N²) on large markets)
        if len(outcome_ids) > config.max_partition_size:
            logger.debug(
                f"Skipping ring: {len(outcome_ids)} outcomes exceeds max_partition_size={config.max_partition_size}",
                outcomes=outcome_ids
            )
            continue

        logger.info("Potential Dutching ring found", outcomes=outcome_ids, size=len(outcome_ids))

        # Collect odds and depth for all legs
        odds_list = []
        depth_list = []
        best_ask_prices = []  # For slippage check
        skip_ring = False

        for o_id in outcome_ids:
            ob = order_books.get(o_id)
            if not ob or not ob.best_ask:
                logger.debug("Skipping ring: missing order book or best_ask", outcome_id=o_id)
                skip_ring = True
                break

            price = float(ob.best_ask)
            odds = (1.0 / price) - 1.0 if price > 0 else 0.0

            depth = 0.0
            if ob.asks and abs(float(ob.asks[0].price) - price) < 1e-6:
                depth = float(ob.asks[0].size) * price # USD liquidity
            if depth == 0:
                depth = 1000.0 # Fallback

            odds_list.append(odds)
            depth_list.append(depth)
            best_ask_prices.append(Decimal(str(price)))

        if skip_ring:
            continue

        # Calculate Dutching sizes
        sizes = self.position_sizer.calculate_dutching_sizes(odds_list, depth_list)

        # Check if any size is > 0 (meaning implied_sum < 1.0)
        if all(s.recommended_size > 0 for s in sizes):
            logger.info("Dutching arb found by sizer", total_stake=sum(s.recommended_size for s in sizes))
            ring_trades = []
            ring_total_profit = Decimal("0")
            ring_markets = set()
            vwap_slippage_detected = False

            for i, o_id in enumerate(outcome_ids):
                size_res = sizes[i]
                ob = order_books[o_id]

                # Refine to VWAP
                vwap = ob.get_vwap(OrderSide.BUY, Decimal(str(size_res.recommended_size)))
                if vwap is None:
                    logger.debug("Skipping ring: insufficient VWAP depth", outcome_id=o_id)
                    skip_ring = True
                    break

                # P1.1: VWAP Slippage Check
                best_ask = best_ask_prices[i]
                if best_ask > 0:
                    slippage_pct = abs(vwap - best_ask) / best_ask
                    if slippage_pct > Decimal(str(config.vwap_slippage_limit)):
                        logger.debug(
                            f"Dutching ring rejected: VWAP slippage {slippage_pct:.2%} exceeds limit {config.vwap_slippage_limit:.2%}",
                            outcome_id=o_id,
                            vwap=vwap,
                            best_ask=best_ask
                        )
                        vwap_slippage_detected = True
                        skip_ring = True
                        break

                market_id = extract_market_id(o_id)
                ring_markets.add(market_id)

                exchange_info = validated.market_exchanges.get(market_id, "polymarket")
                exchange_name = "polymarket"
                reason = "dutching_arb"

                if exchange_info.startswith("limitless:"):
                    exchange_name = "limitless"
                    slug = exchange_info.split(":")[1]
                    reason += f":{slug}"
                elif exchange_info == "limitless":
                    exchange_name = "limitless"

                ring_trades.append(ProposedTrade(
                    market_id=market_id,
                    outcome_id=o_id,
                    side=OrderSide.BUY,
                    size=size_res.recommended_size,
                    limit_price=vwap,
                    exchange=exchange_name,
                    reason=reason
                ))

            if skip_ring:
                if vwap_slippage_detected:
                    logger.debug("Ring rejected due to VWAP slippage")
                continue

            # Calculate profit for the ring
            # With Dutching, payout is identical regardless of which outcome wins
            # Payout = size_res.recommended_size (number of shares) * 1.0 (payout per share)
            # We just take the payout from the first leg since they should be equal
            payout = Decimal(str(sizes[0].recommended_size))

            total_cost = sum(t.size * t.limit_price for t in ring_trades)

            ring_total_profit = payout - total_cost

            # Store candidate ring for later ranking
            candidate_rings.append({
                "trades": ring_trades,
                "markets": list(ring_markets),
                "gross_profit": ring_total_profit,
                "capital_deployed": total_cost,
                "sizes": sizes,
                "depth_list": depth_list
            })

    if not candidate_rings:
        return None

    # P2.1: Rank rings by ROI and process in order
    # This ensures we pick the best rings first and reserve capital accordingly
    candidate_rings.sort(key=lambda r: float(r["gross_profit"] / r["capital_deployed"] if r["capital_deployed"] > 0 else 0), reverse=True)

    # Process top ring (or multiple if capital allows)
    for ring in candidate_rings:
        ring_trades = ring["trades"]
        ring_markets = ring["markets"]
        ring_gross_profit = ring["gross_profit"]
        capital_required = ring["capital_deployed"]

        # P2.1: Capital reservation check
        if reserved_capital + capital_required > available_capital:
            logger.debug(f"Skipping ring: insufficient remaining capital (need={capital_required}, available={available_capital - reserved_capital})")
            continue

        # Deduct Fees and Gas
        total_gas = Decimal("0")
        total_fees = Decimal("0")
        poly_gas = Decimal(str(config.polygon_gas_per_tx))
        base_gas = Decimal(str(config.base_gas_per_tx))
        poly_fee_pct = Decimal(str(config.polymarket_taker_fee_pct))
        limitless_fee_pct = Decimal(str(config.limitless_taker_fee_pct))

        for t in ring_trades:
            if t.exchange == "limitless":
                total_gas += base_gas
                total_fees += Decimal(str(t.size)) * t.limit_price * limitless_fee_pct
            else:
                total_gas += poly_gas
                total_fees += Decimal(str(t.size)) * t.limit_price * poly_fee_pct

        # P2.3: Expected unwind cost (if partial fill occurs)
        partial_fill_prob = Decimal(str(config.partial_fill_probability))
        unwind_spread = Decimal(str(config.unwind_spread_estimate))
        expected_unwind_cost = Decimal("0")

        for t in ring_trades:
            notional_value = Decimal(str(t.size)) * t.limit_price
            expected_unwind_cost += notional_value * unwind_spread * partial_fill_prob

        # Net profit after all costs
        net_profit = ring_gross_profit - total_gas - total_fees - expected_unwind_cost

        if net_profit <= 0:
            logger.debug(f"Ring rejected: net profit {net_profit} <= 0 after fees/gas/unwind_cost")
            continue

        # P1.2: Calculate ROI
        roi = float(net_profit / capital_required) if capital_required > 0 else 0.0

        # P2.2: Dynamic confidence scoring
        # Based on liquidity cushion (how much depth vs stake) and staleness
        liquidity_ratios = []
        for i, t in enumerate(ring_trades):
            stake = Decimal(str(t.size)) * t.limit_price
            depth = Decimal(str(ring["depth_list"][i]))
            if stake > 0:
                liquidity_ratios.append(float(depth / stake))

        min_liquidity_ratio = min(liquidity_ratios) if liquidity_ratios else 1.0

        # Liquidity confidence: 0.7 if tight (1.1x), 1.0 if ample (5x+)
        liquidity_confidence = min(1.0, 0.7 + (min_liquidity_ratio - 1.0) * 0.15)

        # Staleness penalty (assume 0ms staleness for now, can be enhanced with order book age)
        staleness_confidence = 1.0  # Placeholder for future enhancement

        # Combined confidence (80% liquidity, 20% staleness)
        confidence = min(1.0, liquidity_confidence * 0.8 + staleness_confidence * 0.2)

        # P1.2: Capital efficiency (profit per second, estimated)
        # Assume ~100ms execution time per leg
        estimated_execution_time_sec = len(ring_trades) * 0.1
        capital_efficiency = float(net_profit / Decimal(str(estimated_execution_time_sec)))

        # Accept this ring (only processing top ring for now, can extend to multiple)
        reserved_capital += capital_required
        best_ring_trades = ring_trades
        best_ring_profit = net_profit
        best_ring_markets = ring_markets
        best_ring_roi = roi

        logger.info(
            f"Found Dutching Arbitrage Opportunity",
            profit=float(net_profit),
            roi=f"{roi:.2%}",
            confidence=f"{confidence:.2%}",
            capital_efficiency=f"${capital_efficiency:.2f}/sec"
        )

        return ArbitrageOpportunity(
            markets=best_ring_markets,
            trades=best_ring_trades,
            expected_profit=net_profit,
            roi=roi,
            capital_efficiency=capital_efficiency,
            confidence=confidence
        )

    # No acceptable rings found
    return None
