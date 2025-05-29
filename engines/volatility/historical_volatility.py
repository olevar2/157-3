"""
Historical Volatility - Standard Deviation-based Volatility Measurement
Calculates rolling volatility using price returns over specified periods.
Essential for risk assessment, position sizing, and volatility-based trading strategies.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class HistoricalVolatilitySignal:
    """Signal output for Historical Volatility"""
    timestamp: Optional[Any] = None
    volatility: float = 0.0
    volatility_percentile: float = 50.0
    volatility_regime: str = "normal"
    volatility_momentum: float = 0.0
    
    # Analysis components
    trend_confirmation: str = "neutral"
    risk_level: str = "moderate"
    position_sizing_factor: float = 1.0
    volatility_breakout: bool = False
    volatility_expansion: bool = False
    volatility_contraction: bool = False
    
    # Signal generation
    signal_strength: float = 0.0
    signal_confidence: float = 0.0
    signal_direction: str = "hold"
    
    # Component analysis
    short_term_volatility: float = 0.0
    medium_term_volatility: float = 0.0
    long_term_volatility: float = 0.0
    volatility_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'volatility': self.volatility,
            'volatility_percentile': self.volatility_percentile,
            'volatility_regime': self.volatility_regime,
            'volatility_momentum': self.volatility_momentum,
            'trend_confirmation': self.trend_confirmation,
            'risk_level': self.risk_level,
            'position_sizing_factor': self.position_sizing_factor,
            'volatility_breakout': self.volatility_breakout,
            'volatility_expansion': self.volatility_expansion,
            'volatility_contraction': self.volatility_contraction,
            'signal_strength': self.signal_strength,
            'signal_confidence': self.signal_confidence,
            'signal_direction': self.signal_direction,
            'short_term_volatility': self.short_term_volatility,
            'medium_term_volatility': self.medium_term_volatility,
            'long_term_volatility': self.long_term_volatility,
            'volatility_ratio': self.volatility_ratio
        }


class HistoricalVolatility:
    """
    Historical Volatility Indicator Implementation
    
    Calculates market volatility using rolling standard deviation of price returns.
    
    Key Features:
    - Multiple timeframe volatility analysis (short, medium, long term)
    - Volatility regime identification (low, normal, high, extreme)
    - Volatility momentum and trend analysis
    - Risk assessment and position sizing recommendations
    - Volatility breakout and expansion/contraction detection
    - Percentile ranking for relative volatility assessment
    """
    
    def __init__(self, 
                 period: int = 20,
                 short_period: int = 10,
                 long_period: int = 50,
                 annualization_factor: float = 252.0,
                 low_threshold: float = 10.0,
                 high_threshold: float = 20.0,
                 extreme_threshold: float = 30.0):
        """
        Initialize Historical Volatility with advanced parameters
        
        Args:
            period: Main volatility calculation period (default 20)
            short_period: Short-term volatility period (default 10)
            long_period: Long-term volatility period (default 50)
            annualization_factor: Factor for annualizing volatility (default 252 trading days)
            low_threshold: Low volatility threshold in % (default 10%)
            high_threshold: High volatility threshold in % (default 20%)
            extreme_threshold: Extreme volatility threshold in % (default 30%)
        """
        self.period = period
        self.short_period = short_period
        self.long_period = long_period
        self.annualization_factor = annualization_factor
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.extreme_threshold = extreme_threshold
        
        # Data storage
        self.prices: List[float] = []
        self.returns: List[float] = []
        self.volatilities: List[float] = []
        self.short_volatilities: List[float] = []
        self.long_volatilities: List[float] = []
        self.signals: List[HistoricalVolatilitySignal] = []
        
        # Running calculations
        self.last_price = None
        
    def calculate(self, 
                  close: float, 
                  timestamp: Optional[Any] = None) -> HistoricalVolatilitySignal:
        """
        Calculate Historical Volatility with comprehensive analysis
        """
        try:
            # Store price
            self.prices.append(close)
            
            # Calculate return if we have previous price
            if self.last_price is not None:
                daily_return = math.log(close / self.last_price)
                self.returns.append(daily_return)
            
            self.last_price = close
            
            # Need minimum data for volatility calculation
            if len(self.returns) < self.period:
                return HistoricalVolatilitySignal(
                    timestamp=timestamp,
                    volatility=0.0,
                    signal_direction="hold",
                    signal_confidence=0.0
                )
            
            # Calculate main volatility (annualized percentage)
            recent_returns = self.returns[-self.period:]
            volatility = np.std(recent_returns) * math.sqrt(self.annualization_factor) * 100
            self.volatilities.append(volatility)
            
            # Calculate short-term volatility
            if len(self.returns) >= self.short_period:
                short_returns = self.returns[-self.short_period:]
                short_volatility = np.std(short_returns) * math.sqrt(self.annualization_factor) * 100
                self.short_volatilities.append(short_volatility)
            else:
                short_volatility = volatility
                self.short_volatilities.append(short_volatility)
            
            # Calculate long-term volatility
            if len(self.returns) >= self.long_period:
                long_returns = self.returns[-self.long_period:]
                long_volatility = np.std(long_returns) * math.sqrt(self.annualization_factor) * 100
                self.long_volatilities.append(long_volatility)
            else:
                long_volatility = volatility
                self.long_volatilities.append(long_volatility)
            
            # Volatility momentum (rate of change)
            volatility_momentum = 0.0
            if len(self.volatilities) >= 6:
                current_vol = volatility
                previous_vol = self.volatilities[-6]
                if previous_vol > 0:
                    volatility_momentum = (current_vol - previous_vol) / previous_vol * 100
            
            # Determine volatility regime
            volatility_regime = self._identify_volatility_regime(volatility)
            
            # Calculate volatility percentile
            volatility_percentile = self._calculate_percentile(volatility, self.volatilities)
            
            # Risk assessment
            risk_level = self._assess_risk_level(volatility)
            
            # Position sizing factor (inverse relationship with volatility)
            position_sizing_factor = max(0.1, min(2.0, self.high_threshold / max(volatility, 1.0)))
            
            # Volatility patterns
            volatility_breakout = self._detect_volatility_breakout(volatility)
            volatility_expansion = self._detect_volatility_expansion(short_volatility, long_volatility)
            volatility_contraction = self._detect_volatility_contraction(short_volatility, long_volatility)
            
            # Volatility ratio (short-term vs long-term)
            volatility_ratio = short_volatility / max(long_volatility, 0.01)
            
            # Trend confirmation based on volatility patterns
            trend_confirmation = self._analyze_trend_confirmation(
                volatility, volatility_momentum, volatility_ratio
            )
            
            # Generate signal
            signal_direction, signal_strength, signal_confidence = self._generate_signal(
                volatility, volatility_regime, volatility_momentum, volatility_breakout,
                volatility_expansion, volatility_contraction
            )
            
            # Create signal
            signal = HistoricalVolatilitySignal(
                timestamp=timestamp,
                volatility=volatility,
                volatility_percentile=volatility_percentile,
                volatility_regime=volatility_regime,
                volatility_momentum=volatility_momentum,
                trend_confirmation=trend_confirmation,
                risk_level=risk_level,
                position_sizing_factor=position_sizing_factor,
                volatility_breakout=volatility_breakout,
                volatility_expansion=volatility_expansion,
                volatility_contraction=volatility_contraction,
                signal_strength=signal_strength,
                signal_confidence=signal_confidence,
                signal_direction=signal_direction,
                short_term_volatility=short_volatility,
                medium_term_volatility=volatility,
                long_term_volatility=long_volatility,
                volatility_ratio=volatility_ratio
            )
            
            self.signals.append(signal)
            return signal
            
        except Exception as e:
            return HistoricalVolatilitySignal(
                timestamp=timestamp,
                volatility=0.0,
                signal_direction="error",
                signal_confidence=0.0
            )
    
    def _identify_volatility_regime(self, volatility: float) -> str:
        """Identify current volatility regime"""
        if volatility >= self.extreme_threshold:
            return "extreme"
        elif volatility >= self.high_threshold:
            return "high"
        elif volatility <= self.low_threshold:
            return "low"
        else:
            return "normal"
    
    def _calculate_percentile(self, current_value: float, historical_values: List[float]) -> float:
        """Calculate percentile rank of current value"""
        if len(historical_values) < 10:
            return 50.0
        
        # Use last 100 values for percentile calculation
        recent_values = historical_values[-100:] if len(historical_values) > 100 else historical_values
        rank = sum(1 for v in recent_values if v <= current_value)
        return (rank / len(recent_values)) * 100
    
    def _assess_risk_level(self, volatility: float) -> str:
        """Assess risk level based on volatility"""
        if volatility >= self.extreme_threshold:
            return "extreme"
        elif volatility >= self.high_threshold:
            return "high"
        elif volatility <= self.low_threshold:
            return "low"
        else:
            return "moderate"
    
    def _detect_volatility_breakout(self, current_volatility: float) -> bool:
        """Detect volatility breakout"""
        if len(self.volatilities) < 10:
            return False
        
        recent_avg = np.mean(self.volatilities[-10:])
        return current_volatility > recent_avg * 1.5  # 50% increase threshold
    
    def _detect_volatility_expansion(self, short_vol: float, long_vol: float) -> bool:
        """Detect volatility expansion (short-term > long-term)"""
        return short_vol > long_vol * 1.2  # 20% threshold
    
    def _detect_volatility_contraction(self, short_vol: float, long_vol: float) -> bool:
        """Detect volatility contraction (short-term < long-term)"""
        return short_vol < long_vol * 0.8  # 20% threshold
    
    def _analyze_trend_confirmation(self, volatility: float, momentum: float, ratio: float) -> str:
        """Analyze trend confirmation based on volatility patterns"""
        if volatility <= self.low_threshold and momentum < -10:
            return "strong_trend_likely"
        elif volatility >= self.high_threshold and momentum > 10:
            return "trend_weakening"
        elif ratio > 1.3:  # Short-term vol much higher than long-term
            return "trend_change_possible"
        elif ratio < 0.7:  # Short-term vol much lower than long-term
            return "trend_continuation"
        else:
            return "neutral"
    
    def _generate_signal(self, volatility: float, regime: str, momentum: float,
                        breakout: bool, expansion: bool, contraction: bool) -> Tuple[str, float, float]:
        """Generate trading signal based on volatility analysis"""
        
        # Default values
        direction = "hold"
        strength = 0.0
        confidence = 0.5
        
        # Low volatility - potential opportunity
        if regime == "low" and momentum < 0:
            direction = "buy_opportunity"
            strength = 0.6
            confidence = 0.7
        
        # High volatility - caution
        elif regime == "high" or regime == "extreme":
            direction = "caution"
            strength = 0.7 if regime == "extreme" else 0.5
            confidence = 0.8
        
        # Volatility breakout - trend change possible
        elif breakout:
            direction = "trend_change"
            strength = 0.6
            confidence = 0.7
        
        # Volatility expansion - momentum building
        elif expansion and momentum > 5:
            direction = "momentum_building"
            strength = 0.5
            confidence = 0.6
        
        # Volatility contraction - consolidation
        elif contraction and momentum < -5:
            direction = "consolidation"
            strength = 0.4
            confidence = 0.6
        
        return direction, strength, confidence
    
    def get_current_signal(self) -> Optional[HistoricalVolatilitySignal]:
        """Get the most recent signal"""
        return self.signals[-1] if self.signals else None
    
    def get_volatility_summary(self) -> Dict[str, Any]:
        """Get comprehensive volatility summary"""
        if not self.volatilities:
            return {}
        
        current_vol = self.volatilities[-1]
        avg_vol = np.mean(self.volatilities[-20:]) if len(self.volatilities) >= 20 else np.mean(self.volatilities)
        
        return {
            'current_volatility': current_vol,
            'average_volatility': avg_vol,
            'volatility_regime': self._identify_volatility_regime(current_vol),
            'risk_level': self._assess_risk_level(current_vol),
            'percentile_rank': self._calculate_percentile(current_vol, self.volatilities),
            'position_sizing_factor': max(0.1, min(2.0, self.high_threshold / max(current_vol, 1.0))),
            'data_points': len(self.volatilities)
        }


def test_historical_volatility():
    """Test Historical Volatility implementation with EURUSD-like data"""
    print("=== HISTORICAL VOLATILITY TEST ===")
    
    # Initialize indicator
    hv = HistoricalVolatility(
        period=20,
        short_period=10,
        long_period=50,
        low_threshold=8.0,
        high_threshold=15.0,
        extreme_threshold=25.0
    )
    
    # Simulate EURUSD price data with varying volatility periods
    np.random.seed(42)
    base_price = 1.1000
    results = []
    
    print("Simulating price data with varying volatility periods...")
    
    for i in range(100):
        # Create different volatility regimes
        if i < 25:
            # Low volatility period
            daily_vol = 0.003  # ~0.3% daily volatility
            regime_name = "Low Volatility"
        elif i < 50:
            # Normal volatility period  
            daily_vol = 0.008  # ~0.8% daily volatility
            regime_name = "Normal Volatility"
        elif i < 75:
            # High volatility period
            daily_vol = 0.015  # ~1.5% daily volatility
            regime_name = "High Volatility"
        else:
            # Extreme volatility period
            daily_vol = 0.025  # ~2.5% daily volatility
            regime_name = "Extreme Volatility"
        
        # Generate price with controlled volatility
        return_val = np.random.normal(0, daily_vol)
        base_price = base_price * (1 + return_val)
        
        # Calculate indicator
        signal = hv.calculate(base_price, timestamp=i)
        results.append(signal)
        
        # Print updates every 25 periods
        if (i + 1) % 25 == 0:
            print(f"\nPeriod {i+1} ({regime_name}):")
            print(f"  Price: {base_price:.4f}")
            print(f"  Volatility: {signal.volatility:.2f}%")
            print(f"  Regime: {signal.volatility_regime}")
            print(f"  Risk Level: {signal.risk_level}")
            print(f"  Signal: {signal.signal_direction}")
            print(f"  Position Size Factor: {signal.position_sizing_factor:.2f}")
    
    # Final analysis
    print(f"\n=== FINAL ANALYSIS ===")
    final_signal = results[-1]
    summary = hv.get_volatility_summary()
    
    print(f"Final Volatility: {final_signal.volatility:.2f}%")
    print(f"Volatility Regime: {final_signal.volatility_regime}")
    print(f"Volatility Percentile: {final_signal.volatility_percentile:.1f}%")
    print(f"Risk Level: {final_signal.risk_level}")
    print(f"Signal Direction: {final_signal.signal_direction}")
    print(f"Signal Strength: {final_signal.signal_strength:.2f}")
    print(f"Signal Confidence: {final_signal.signal_confidence:.2f}")
    print(f"Position Sizing Factor: {final_signal.position_sizing_factor:.2f}")
    
    print(f"\nVolatility Components:")
    print(f"  Short-term: {final_signal.short_term_volatility:.2f}%")
    print(f"  Medium-term: {final_signal.medium_term_volatility:.2f}%")
    print(f"  Long-term: {final_signal.long_term_volatility:.2f}%")
    print(f"  Volatility Ratio: {final_signal.volatility_ratio:.2f}")
    
    print(f"\nVolatility Patterns:")
    print(f"  Breakout: {final_signal.volatility_breakout}")
    print(f"  Expansion: {final_signal.volatility_expansion}")
    print(f"  Contraction: {final_signal.volatility_contraction}")
    print(f"  Trend Confirmation: {final_signal.trend_confirmation}")
    
    # Test edge cases
    print(f"\n=== EDGE CASE TESTS ===")
    
    # Test with insufficient data
    hv_new = HistoricalVolatility()
    early_signal = hv_new.calculate(1.1000)
    print(f"Insufficient data signal: {early_signal.signal_direction}")
    
    # Test with extreme price movement
    extreme_signal = hv.calculate(base_price * 1.1)  # 10% jump
    print(f"Extreme movement volatility: {extreme_signal.volatility:.2f}%")
    print(f"Extreme movement regime: {extreme_signal.volatility_regime}")
    
    print(f"\n=== TEST COMPLETED SUCCESSFULLY ===")
    return True


if __name__ == "__main__":
    test_historical_volatility()
