#!/usr/bin/env python3
"""
Price Comparator for Arbitrage Detection
Advanced price comparison and analysis for forex trading platform
Identifies price discrepancies across multiple data sources and brokers

Author: Platform3 Development Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import statistics
import numpy as np
from collections import defaultdict, deque
import threading

# Import from ArbitrageEngine
from .ArbitrageEngine import PriceQuote, ArbitrageConfig

@dataclass
class PriceComparison:
    """Price comparison result"""
    symbol: str
    source_a: str
    source_b: str
    price_diff_pips: float
    price_diff_percentage: float
    spread_diff: float
    timestamp: datetime
    confidence: float
    is_significant: bool

@dataclass
class PriceStatistics:
    """Price statistics for a symbol"""
    symbol: str
    source: str
    avg_price: float
    price_volatility: float
    avg_spread: float
    spread_volatility: float
    update_frequency: float  # Updates per second
    last_update: datetime
    sample_count: int

class PriceComparator:
    """
    Advanced price comparison engine for arbitrage detection
    Analyzes price differences across multiple data sources
    """
    
    def __init__(self, config: ArbitrageConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Price data storage
        self.price_data: Dict[str, Dict[str, PriceQuote]] = defaultdict(dict)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Price statistics
        self.price_stats: Dict[str, PriceStatistics] = {}
        
        # Comparison results
        self.comparison_history: deque = deque(maxlen=10000)
        
        # Performance metrics
        self.stats = {
            "comparisons_performed": 0,
            "significant_differences": 0,
            "avg_price_diff": 0.0,
            "max_price_diff": 0.0,
            "start_time": None
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Thresholds for significant price differences
        self.significance_thresholds = {
            "min_pip_difference": 0.3,
            "min_percentage_difference": 0.005,  # 0.005%
            "min_confidence": 0.6
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("PriceComparator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def update_price(self, quote: PriceQuote):
        """Update price data and trigger comparison"""
        with self.lock:
            # Store latest price
            self.price_data[quote.symbol][quote.source] = quote
            
            # Store in history
            history_key = f"{quote.symbol}_{quote.source}"
            self.price_history[history_key].append(quote)
            
            # Update statistics
            self._update_price_statistics(quote)
            
            # Trigger comparison if we have multiple sources
            if len(self.price_data[quote.symbol]) > 1:
                self._compare_prices(quote.symbol)
    
    def _update_price_statistics(self, quote: PriceQuote):
        """Update price statistics for a source"""
        stats_key = f"{quote.symbol}_{quote.source}"
        history_key = f"{quote.symbol}_{quote.source}"
        
        if history_key in self.price_history:
            history = list(self.price_history[history_key])
            
            if len(history) >= 10:  # Need minimum samples
                # Calculate price statistics
                mid_prices = [(q.bid + q.ask) / 2 for q in history]
                spreads = [q.spread for q in history]
                
                avg_price = statistics.mean(mid_prices)
                price_volatility = statistics.stdev(mid_prices) if len(mid_prices) > 1 else 0.0
                avg_spread = statistics.mean(spreads)
                spread_volatility = statistics.stdev(spreads) if len(spreads) > 1 else 0.0
                
                # Calculate update frequency
                if len(history) > 1:
                    time_span = (history[-1].timestamp - history[0].timestamp).total_seconds()
                    update_frequency = len(history) / time_span if time_span > 0 else 0
                else:
                    update_frequency = 0
                
                self.price_stats[stats_key] = PriceStatistics(
                    symbol=quote.symbol,
                    source=quote.source,
                    avg_price=avg_price,
                    price_volatility=price_volatility,
                    avg_spread=avg_spread,
                    spread_volatility=spread_volatility,
                    update_frequency=update_frequency,
                    last_update=quote.timestamp,
                    sample_count=len(history)
                )
    
    def _compare_prices(self, symbol: str):
        """Compare prices across all sources for a symbol"""
        if symbol not in self.price_data:
            return
        
        sources = list(self.price_data[symbol].keys())
        
        # Compare all source pairs
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source_a = sources[i]
                source_b = sources[j]
                
                quote_a = self.price_data[symbol][source_a]
                quote_b = self.price_data[symbol][source_b]
                
                comparison = self._analyze_price_difference(quote_a, quote_b)
                if comparison:
                    self.comparison_history.append(comparison)
                    
                    # Update statistics
                    self.stats["comparisons_performed"] += 1
                    if comparison.is_significant:
                        self.stats["significant_differences"] += 1
                    
                    # Update running averages
                    self._update_comparison_stats(comparison)
    
    def _analyze_price_difference(self, quote_a: PriceQuote, quote_b: PriceQuote) -> Optional[PriceComparison]:
        """Analyze price difference between two quotes"""
        try:
            # Check if quotes are recent enough
            now = datetime.now(timezone.utc)
            if (now - quote_a.timestamp).total_seconds() > 10 or \
               (now - quote_b.timestamp).total_seconds() > 10:
                return None
            
            # Calculate mid prices
            mid_a = (quote_a.bid + quote_a.ask) / 2
            mid_b = (quote_b.bid + quote_b.ask) / 2
            
            # Calculate price difference
            price_diff = abs(mid_a - mid_b)
            price_diff_pips = price_diff * 10000  # Convert to pips
            price_diff_percentage = (price_diff / min(mid_a, mid_b)) * 100
            
            # Calculate spread difference
            spread_diff = abs(quote_a.spread - quote_b.spread)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_comparison_confidence(quote_a, quote_b)
            
            # Determine if difference is significant
            is_significant = (
                price_diff_pips >= self.significance_thresholds["min_pip_difference"] and
                price_diff_percentage >= self.significance_thresholds["min_percentage_difference"] and
                confidence >= self.significance_thresholds["min_confidence"]
            )
            
            comparison = PriceComparison(
                symbol=quote_a.symbol,
                source_a=quote_a.source,
                source_b=quote_b.source,
                price_diff_pips=price_diff_pips,
                price_diff_percentage=price_diff_percentage,
                spread_diff=spread_diff,
                timestamp=datetime.now(timezone.utc),
                confidence=confidence,
                is_significant=is_significant
            )
            
            if is_significant:
                self.logger.info(
                    f"Significant price difference detected: {quote_a.symbol} "
                    f"{quote_a.source} vs {quote_b.source} - "
                    f"Diff: {price_diff_pips:.2f} pips ({price_diff_percentage:.3f}%) "
                    f"Confidence: {confidence:.2f}"
                )
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error analyzing price difference: {e}")
            return None
    
    def _calculate_comparison_confidence(self, quote_a: PriceQuote, quote_b: PriceQuote) -> float:
        """Calculate confidence in price comparison"""
        confidence = 1.0
        
        # Reduce confidence based on latency
        max_latency = max(quote_a.latency_ms, quote_b.latency_ms)
        confidence -= min(max_latency / 200, 0.3)  # Max 30% reduction
        
        # Reduce confidence based on time difference
        time_diff = abs((quote_a.timestamp - quote_b.timestamp).total_seconds())
        confidence -= min(time_diff / 5, 0.2)  # Max 20% reduction
        
        # Reduce confidence based on spread width
        avg_spread = (quote_a.spread + quote_b.spread) / 2
        if avg_spread > 0.001:  # 1 pip
            confidence -= min(avg_spread * 500, 0.2)  # Max 20% reduction
        
        # Adjust confidence based on historical volatility
        stats_a_key = f"{quote_a.symbol}_{quote_a.source}"
        stats_b_key = f"{quote_b.symbol}_{quote_b.source}"
        
        if stats_a_key in self.price_stats and stats_b_key in self.price_stats:
            avg_volatility = (self.price_stats[stats_a_key].price_volatility + 
                            self.price_stats[stats_b_key].price_volatility) / 2
            
            if avg_volatility > 0.001:  # High volatility
                confidence -= min(avg_volatility * 100, 0.15)  # Max 15% reduction
        
        return max(confidence, 0.1)  # Minimum 10% confidence
    
    def _update_comparison_stats(self, comparison: PriceComparison):
        """Update comparison statistics"""
        # Update average price difference
        if self.stats["comparisons_performed"] > 0:
            current_avg = self.stats["avg_price_diff"]
            new_avg = ((current_avg * (self.stats["comparisons_performed"] - 1)) + 
                      comparison.price_diff_pips) / self.stats["comparisons_performed"]
            self.stats["avg_price_diff"] = new_avg
        else:
            self.stats["avg_price_diff"] = comparison.price_diff_pips
        
        # Update maximum price difference
        if comparison.price_diff_pips > self.stats["max_price_diff"]:
            self.stats["max_price_diff"] = comparison.price_diff_pips
    
    def get_price_statistics(self, symbol: str = None, source: str = None) -> Dict[str, Any]:
        """Get price statistics for symbol/source"""
        with self.lock:
            if symbol and source:
                stats_key = f"{symbol}_{source}"
                if stats_key in self.price_stats:
                    stats = self.price_stats[stats_key]
                    return {
                        "symbol": stats.symbol,
                        "source": stats.source,
                        "avg_price": stats.avg_price,
                        "price_volatility": stats.price_volatility,
                        "avg_spread": stats.avg_spread,
                        "spread_volatility": stats.spread_volatility,
                        "update_frequency": stats.update_frequency,
                        "last_update": stats.last_update.isoformat(),
                        "sample_count": stats.sample_count
                    }
                return {}
            else:
                # Return all statistics
                result = {}
                for key, stats in self.price_stats.items():
                    result[key] = {
                        "symbol": stats.symbol,
                        "source": stats.source,
                        "avg_price": stats.avg_price,
                        "price_volatility": stats.price_volatility,
                        "avg_spread": stats.avg_spread,
                        "spread_volatility": stats.spread_volatility,
                        "update_frequency": stats.update_frequency,
                        "last_update": stats.last_update.isoformat(),
                        "sample_count": stats.sample_count
                    }
                return result
    
    def get_recent_comparisons(self, symbol: str = None, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent price comparisons"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        recent_comparisons = []
        for comparison in self.comparison_history:
            if comparison.timestamp > cutoff_time:
                if symbol is None or comparison.symbol == symbol:
                    recent_comparisons.append({
                        "symbol": comparison.symbol,
                        "source_a": comparison.source_a,
                        "source_b": comparison.source_b,
                        "price_diff_pips": comparison.price_diff_pips,
                        "price_diff_percentage": comparison.price_diff_percentage,
                        "spread_diff": comparison.spread_diff,
                        "timestamp": comparison.timestamp.isoformat(),
                        "confidence": comparison.confidence,
                        "is_significant": comparison.is_significant
                    })
        
        # Sort by timestamp (most recent first)
        return sorted(recent_comparisons, key=lambda x: x["timestamp"], reverse=True)
    
    def get_significant_differences(self, symbol: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get significant price differences"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        significant_diffs = []
        for comparison in self.comparison_history:
            if (comparison.timestamp > cutoff_time and 
                comparison.is_significant):
                if symbol is None or comparison.symbol == symbol:
                    significant_diffs.append({
                        "symbol": comparison.symbol,
                        "source_a": comparison.source_a,
                        "source_b": comparison.source_b,
                        "price_diff_pips": comparison.price_diff_pips,
                        "price_diff_percentage": comparison.price_diff_percentage,
                        "timestamp": comparison.timestamp.isoformat(),
                        "confidence": comparison.confidence
                    })
        
        # Sort by price difference (largest first)
        return sorted(significant_diffs, key=lambda x: x["price_diff_pips"], reverse=True)
    
    def get_source_comparison_matrix(self, symbol: str) -> Dict[str, Any]:
        """Get comparison matrix for all sources of a symbol"""
        if symbol not in self.price_data:
            return {}
        
        sources = list(self.price_data[symbol].keys())
        matrix = {}
        
        for source_a in sources:
            matrix[source_a] = {}
            for source_b in sources:
                if source_a != source_b:
                    # Find recent comparison
                    recent_comparison = None
                    for comparison in reversed(self.comparison_history):
                        if (comparison.symbol == symbol and
                            ((comparison.source_a == source_a and comparison.source_b == source_b) or
                             (comparison.source_a == source_b and comparison.source_b == source_a))):
                            recent_comparison = comparison
                            break
                    
                    if recent_comparison:
                        matrix[source_a][source_b] = {
                            "price_diff_pips": recent_comparison.price_diff_pips,
                            "confidence": recent_comparison.confidence,
                            "timestamp": recent_comparison.timestamp.isoformat()
                        }
                    else:
                        matrix[source_a][source_b] = None
                else:
                    matrix[source_a][source_b] = {
                        "price_diff_pips": 0.0,
                        "confidence": 1.0,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
        
        return matrix
    
    def get_comparison_stats(self) -> Dict[str, Any]:
        """Get overall comparison statistics"""
        with self.lock:
            stats = self.stats.copy()
            
            if stats["start_time"]:
                runtime = time.time() - stats["start_time"]
                stats["runtime_hours"] = runtime / 3600
                stats["comparisons_per_hour"] = (stats["comparisons_performed"] / runtime) * 3600 if runtime > 0 else 0
            
            if stats["comparisons_performed"] > 0:
                stats["significant_percentage"] = (stats["significant_differences"] / stats["comparisons_performed"]) * 100
            else:
                stats["significant_percentage"] = 0
            
            stats["active_symbols"] = len(self.price_data)
            stats["total_sources"] = sum(len(sources) for sources in self.price_data.values())
            
            return stats
    
    def set_significance_thresholds(self, min_pip_difference: float = None, 
                                  min_percentage_difference: float = None,
                                  min_confidence: float = None):
        """Update significance thresholds"""
        if min_pip_difference is not None:
            self.significance_thresholds["min_pip_difference"] = min_pip_difference
        
        if min_percentage_difference is not None:
            self.significance_thresholds["min_percentage_difference"] = min_percentage_difference
        
        if min_confidence is not None:
            self.significance_thresholds["min_confidence"] = min_confidence
        
        self.logger.info(f"Updated significance thresholds: {self.significance_thresholds}")
    
    def start_monitoring(self):
        """Start price comparison monitoring"""
        self.stats["start_time"] = time.time()
        self.logger.info("Price comparator monitoring started")
    
    def stop_monitoring(self):
        """Stop price comparison monitoring"""
        self.logger.info("Price comparator monitoring stopped")


# Example usage
if __name__ == "__main__":
    from ArbitrageEngine import ArbitrageConfig
    
    config = ArbitrageConfig()
    comparator = PriceComparator(config)
    
    # Start monitoring
    comparator.start_monitoring()
    
    # Simulate price updates
    import random
    from datetime import datetime, timezone
    
    symbols = ["EURUSD", "GBPUSD"]
    sources = ["broker_a", "broker_b", "broker_c"]
    
    for i in range(100):
        for symbol in symbols:
            base_price = 1.1000 if symbol == "EURUSD" else 1.3000
            
            for source in sources:
                # Add random variation and source bias
                variation = random.uniform(-0.001, 0.001)
                source_bias = random.uniform(-0.0005, 0.0005) if source == "broker_b" else 0
                
                bid = base_price + variation + source_bias
                ask = bid + random.uniform(0.0001, 0.0003)
                
                quote = PriceQuote(
                    source=source,
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    timestamp=datetime.now(timezone.utc),
                    volume=random.uniform(100, 1000),
                    latency_ms=random.uniform(1, 5)
                )
                
                comparator.update_price(quote)
        
        time.sleep(0.1)  # 100ms between updates
    
    # Print results
    print("Comparison Statistics:")
    stats = comparator.get_comparison_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSignificant Differences:")
    significant = comparator.get_significant_differences()
    for diff in significant[:5]:  # Show top 5
        print(f"  {diff['symbol']} {diff['source_a']} vs {diff['source_b']}: {diff['price_diff_pips']:.2f} pips")
    
    comparator.stop_monitoring()
