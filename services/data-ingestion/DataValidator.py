#!/usr/bin/env python3
"""
Data Validator for Real-Time Market Data
Advanced validation and quality assurance for forex trading platform
Optimized for high-frequency data validation with minimal latency

Author: Platform3 Development Team
Version: 1.0.0
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
import numpy as np
from collections import deque, defaultdict
import threading
import json

# Import the TickData class from RealTimeDataProcessor
from RealTimeDataProcessor import TickData

class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    expected_range: Optional[Tuple[float, float]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class ValidationConfig:
    """Configuration for data validation"""
    # Price validation
    min_price: float = 0.0001
    max_price: float = 100.0
    max_price_change_pct: float = 5.0  # 5% max change
    
    # Spread validation
    min_spread: float = 0.0001
    max_spread: float = 0.01  # 100 pips
    max_spread_pct: float = 1.0  # 1% of price
    
    # Volume validation
    min_volume: float = 0.0
    max_volume: float = 1000000.0
    
    # Timestamp validation
    max_timestamp_delay: int = 60  # seconds
    max_future_timestamp: int = 5  # seconds
    
    # Statistical validation
    price_outlier_threshold: float = 3.0  # standard deviations
    volume_outlier_threshold: float = 3.0
    
    # Historical data window
    history_window_size: int = 1000
    min_history_for_stats: int = 100

class DataValidator:
    """
    High-performance data validator for real-time market data
    Provides comprehensive validation with statistical analysis
    """
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.logger = self._setup_logging()
        
        # Historical data for statistical validation
        self.price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.history_window_size)
        )
        self.volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.history_window_size)
        )
        self.spread_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.history_window_size)
        )
        
        # Last known values for change detection
        self.last_tick: Dict[str, TickData] = {}
        
        # Validation statistics
        self.stats = {
            "total_validations": 0,
            "valid_ticks": 0,
            "invalid_ticks": 0,
            "warnings": 0,
            "errors": 0,
            "critical_errors": 0,
            "validation_times": deque(maxlen=1000)
        }
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("DataValidator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_tick(self, tick: TickData) -> List[ValidationResult]:
        """
        Comprehensive validation of a single tick
        Returns list of validation results
        """
        start_time = time.time()
        results = []
        
        with self.lock:
            self.stats["total_validations"] += 1
            
            try:
                # Basic field validation
                results.extend(self._validate_basic_fields(tick))
                
                # Price validation
                results.extend(self._validate_prices(tick))
                
                # Spread validation
                results.extend(self._validate_spread(tick))
                
                # Volume validation
                results.extend(self._validate_volume(tick))
                
                # Timestamp validation
                results.extend(self._validate_timestamp(tick))
                
                # Statistical validation (if enough history)
                if len(self.price_history[tick.symbol]) >= self.config.min_history_for_stats:
                    results.extend(self._validate_statistical(tick))
                
                # Change validation (if previous tick exists)
                if tick.symbol in self.last_tick:
                    results.extend(self._validate_changes(tick))
                
                # Update history and last tick
                self._update_history(tick)
                self.last_tick[tick.symbol] = tick
                
                # Update statistics
                has_errors = any(r.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL] for r in results)
                has_warnings = any(r.level == ValidationLevel.WARNING for r in results)
                
                if has_errors:
                    self.stats["invalid_ticks"] += 1
                    if any(r.level == ValidationLevel.CRITICAL for r in results):
                        self.stats["critical_errors"] += 1
                    else:
                        self.stats["errors"] += 1
                else:
                    self.stats["valid_ticks"] += 1
                
                if has_warnings:
                    self.stats["warnings"] += 1
                
                # Record validation time
                validation_time = (time.time() - start_time) * 1000  # ms
                self.stats["validation_times"].append(validation_time)
                
                return results
                
            except Exception as e:
                self.logger.error(f"Validation error: {e}")
                return [ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.CRITICAL,
                    message=f"Validation exception: {str(e)}",
                    field="validation_system"
                )]
    
    def _validate_basic_fields(self, tick: TickData) -> List[ValidationResult]:
        """Validate basic field presence and types"""
        results = []
        
        # Symbol validation
        if not tick.symbol or not isinstance(tick.symbol, str):
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message="Invalid or missing symbol",
                field="symbol",
                value=tick.symbol
            ))
        elif len(tick.symbol) < 6:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Symbol too short",
                field="symbol",
                value=tick.symbol
            ))
        
        # Session validation
        valid_sessions = ["Asian", "London", "NY", "Overlap"]
        if tick.session not in valid_sessions:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Invalid session: {tick.session}",
                field="session",
                value=tick.session
            ))
        
        return results
    
    def _validate_prices(self, tick: TickData) -> List[ValidationResult]:
        """Validate bid and ask prices"""
        results = []
        
        # Bid price validation
        if tick.bid <= 0:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message="Bid price must be positive",
                field="bid",
                value=tick.bid
            ))
        elif tick.bid < self.config.min_price or tick.bid > self.config.max_price:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Bid price out of valid range",
                field="bid",
                value=tick.bid,
                expected_range=(self.config.min_price, self.config.max_price)
            ))
        
        # Ask price validation
        if tick.ask <= 0:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message="Ask price must be positive",
                field="ask",
                value=tick.ask
            ))
        elif tick.ask < self.config.min_price or tick.ask > self.config.max_price:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Ask price out of valid range",
                field="ask",
                value=tick.ask,
                expected_range=(self.config.min_price, self.config.max_price)
            ))
        
        # Bid-Ask relationship
        if tick.bid > 0 and tick.ask > 0 and tick.ask <= tick.bid:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.CRITICAL,
                message="Ask price must be greater than bid price",
                field="bid_ask_relationship",
                value=f"bid={tick.bid}, ask={tick.ask}"
            ))
        
        return results
    
    def _validate_spread(self, tick: TickData) -> List[ValidationResult]:
        """Validate spread values"""
        results = []
        
        # Calculate actual spread
        if tick.bid > 0 and tick.ask > 0:
            actual_spread = tick.ask - tick.bid
            
            # Check if reported spread matches calculated
            if abs(tick.spread - actual_spread) > 0.00001:  # 0.1 pip tolerance
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message="Reported spread doesn't match calculated spread",
                    field="spread",
                    value=f"reported={tick.spread}, calculated={actual_spread}"
                ))
            
            # Spread range validation
            if actual_spread < self.config.min_spread:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message="Spread too narrow",
                    field="spread",
                    value=actual_spread,
                    expected_range=(self.config.min_spread, self.config.max_spread)
                ))
            elif actual_spread > self.config.max_spread:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Spread too wide",
                    field="spread",
                    value=actual_spread,
                    expected_range=(self.config.min_spread, self.config.max_spread)
                ))
            
            # Spread percentage validation
            mid_price = (tick.bid + tick.ask) / 2
            spread_pct = (actual_spread / mid_price) * 100
            if spread_pct > self.config.max_spread_pct:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message="Spread percentage too high",
                    field="spread_percentage",
                    value=spread_pct
                ))
        
        return results
    
    def _validate_volume(self, tick: TickData) -> List[ValidationResult]:
        """Validate volume data"""
        results = []
        
        if tick.volume < self.config.min_volume:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="Volume below minimum",
                field="volume",
                value=tick.volume,
                expected_range=(self.config.min_volume, self.config.max_volume)
            ))
        elif tick.volume > self.config.max_volume:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Volume above maximum",
                field="volume",
                value=tick.volume,
                expected_range=(self.config.min_volume, self.config.max_volume)
            ))
        
        return results
    
    def _validate_timestamp(self, tick: TickData) -> List[ValidationResult]:
        """Validate timestamp"""
        results = []
        
        now = datetime.now(timezone.utc)
        
        # Check if timestamp is too old
        if (now - tick.timestamp).total_seconds() > self.config.max_timestamp_delay:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message="Timestamp too old",
                field="timestamp",
                value=tick.timestamp.isoformat()
            ))
        
        # Check if timestamp is in the future
        if (tick.timestamp - now).total_seconds() > self.config.max_future_timestamp:
            results.append(ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message="Timestamp in the future",
                field="timestamp",
                value=tick.timestamp.isoformat()
            ))
        
        return results
    
    def _validate_statistical(self, tick: TickData) -> List[ValidationResult]:
        """Statistical validation using historical data"""
        results = []
        
        symbol = tick.symbol
        
        # Price outlier detection
        if len(self.price_history[symbol]) >= self.config.min_history_for_stats:
            prices = list(self.price_history[symbol])
            mean_price = statistics.mean(prices)
            std_price = statistics.stdev(prices) if len(prices) > 1 else 0
            
            if std_price > 0:
                mid_price = (tick.bid + tick.ask) / 2
                z_score = abs(mid_price - mean_price) / std_price
                
                if z_score > self.config.price_outlier_threshold:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Price outlier detected (z-score: {z_score:.2f})",
                        field="price_outlier",
                        value=mid_price
                    ))
        
        # Volume outlier detection
        if len(self.volume_history[symbol]) >= self.config.min_history_for_stats:
            volumes = list(self.volume_history[symbol])
            mean_volume = statistics.mean(volumes)
            std_volume = statistics.stdev(volumes) if len(volumes) > 1 else 0
            
            if std_volume > 0:
                z_score = abs(tick.volume - mean_volume) / std_volume
                
                if z_score > self.config.volume_outlier_threshold:
                    results.append(ValidationResult(
                        is_valid=False,
                        level=ValidationLevel.WARNING,
                        message=f"Volume outlier detected (z-score: {z_score:.2f})",
                        field="volume_outlier",
                        value=tick.volume
                    ))
        
        return results
    
    def _validate_changes(self, tick: TickData) -> List[ValidationResult]:
        """Validate changes from previous tick"""
        results = []
        
        last_tick = self.last_tick[tick.symbol]
        
        # Price change validation
        last_mid = (last_tick.bid + last_tick.ask) / 2
        current_mid = (tick.bid + tick.ask) / 2
        
        if last_mid > 0:
            price_change_pct = abs(current_mid - last_mid) / last_mid * 100
            
            if price_change_pct > self.config.max_price_change_pct:
                results.append(ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Price change too large: {price_change_pct:.2f}%",
                    field="price_change",
                    value=price_change_pct
                ))
        
        return results
    
    def _update_history(self, tick: TickData):
        """Update historical data for statistical analysis"""
        symbol = tick.symbol
        mid_price = (tick.bid + tick.ask) / 2
        
        self.price_history[symbol].append(mid_price)
        self.volume_history[symbol].append(tick.volume)
        self.spread_history[symbol].append(tick.spread)
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        with self.lock:
            stats = self.stats.copy()
            
            # Calculate additional metrics
            if stats["total_validations"] > 0:
                stats["valid_percentage"] = (stats["valid_ticks"] / stats["total_validations"]) * 100
                stats["error_percentage"] = (stats["invalid_ticks"] / stats["total_validations"]) * 100
            else:
                stats["valid_percentage"] = 0
                stats["error_percentage"] = 0
            
            # Validation performance
            if stats["validation_times"]:
                stats["avg_validation_time_ms"] = statistics.mean(stats["validation_times"])
                stats["max_validation_time_ms"] = max(stats["validation_times"])
                stats["p95_validation_time_ms"] = np.percentile(stats["validation_times"], 95)
            
            return stats
    
    def reset_stats(self):
        """Reset validation statistics"""
        with self.lock:
            self.stats = {
                "total_validations": 0,
                "valid_ticks": 0,
                "invalid_ticks": 0,
                "warnings": 0,
                "errors": 0,
                "critical_errors": 0,
                "validation_times": deque(maxlen=1000)
            }
    
    def is_tick_valid(self, tick: TickData) -> bool:
        """
        Quick validation check - returns True if tick is valid
        """
        results = self.validate_tick(tick)
        return not any(r.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL] for r in results)


# Example usage
if __name__ == "__main__":
    # Create validator with custom config
    config = ValidationConfig(
        max_price_change_pct=2.0,
        max_spread=0.005
    )
    validator = DataValidator(config)
    
    # Test with sample tick
    test_tick = TickData(
        symbol="EURUSD",
        timestamp=datetime.now(timezone.utc),
        bid=1.1000,
        ask=1.1005,
        volume=100,
        spread=0.0005,
        session="London"
    )
    
    # Validate tick
    results = validator.validate_tick(test_tick)
    
    print(f"Validation results for {test_tick.symbol}:")
    for result in results:
        print(f"  {result.level.value}: {result.message}")
    
    # Print statistics
    stats = validator.get_validation_stats()
    print(f"\nValidation statistics: {json.dumps(stats, indent=2)}")
