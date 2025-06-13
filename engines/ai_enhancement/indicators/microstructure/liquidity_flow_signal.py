"""
Liquidity Flow Signal Indicator

Analyzes market liquidity flow patterns by tracking bid-ask spreads, market depth changes,
and liquidity absorption patterns to identify liquidity providers and takers.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..base_indicator import StandardIndicatorInterface, IndicatorMetadata, IndicatorValidationError


@dataclass
class LiquidityFlowResult:
    """Result structure for Liquidity Flow analysis"""
    liquidity_state: str  # "abundant", "scarce", "normal"
    flow_direction: str  # "providing", "taking", "neutral"
    depth_change: float  # Change in market depth
    spread_ratio: float  # Current spread vs average
    absorption_rate: float  # Rate of liquidity absorption
    provider_strength: float  # Strength of liquidity providers
    taker_aggression: float  # Aggression level of liquidity takers
    imbalance_score: float  # Buy/sell imbalance


class LiquidityFlowSignal(StandardIndicatorInterface):
    """
    Liquidity Flow Analysis
    
    Monitors liquidity flow through:
    - Bid-ask spread analysis
    - Market depth estimation
    - Liquidity absorption patterns
    - Provider vs taker dynamics
    - Order book imbalance detection
    """
    
    CATEGORY = "microstructure"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"
    
    def __init__(self, lookback_period: int = 20, spread_sensitivity: float = 2.0,
                 depth_threshold: float = 1.5, imbalance_window: int = 5, **kwargs):
        """
        Initialize Liquidity Flow indicator
        
        Args:
            lookback_period: Period for liquidity analysis
            spread_sensitivity: Sensitivity for spread detection
            depth_threshold: Threshold for depth changes
            imbalance_window: Window for imbalance calculation
        """
        self.lookback_period = lookback_period
        self.spread_sensitivity = spread_sensitivity
        self.depth_threshold = depth_threshold
        self.imbalance_window = imbalance_window
        super().__init__(**kwargs)
    
    def _setup_defaults(self):
        """Setup default parameters"""
        if not hasattr(self, 'lookback_period'):
            self.lookback_period = 20
        if not hasattr(self, 'spread_sensitivity'):
            self.spread_sensitivity = 2.0
        if not hasattr(self, 'depth_threshold'):
            self.depth_threshold = 1.5
        if not hasattr(self, 'imbalance_window'):
            self.imbalance_window = 5
    
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        if self.lookback_period < 5:
            raise IndicatorValidationError("Lookback period must be at least 5")
        if self.spread_sensitivity <= 0:
            raise IndicatorValidationError("Spread sensitivity must be positive")
        if self.depth_threshold <= 0:
            raise IndicatorValidationError("Depth threshold must be positive")
        if self.imbalance_window < 2:
            raise IndicatorValidationError("Imbalance window must be at least 2")
        return True
    
    def _get_required_columns(self) -> List[str]:
        """Required data columns"""
        return ["high", "low", "close", "volume"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points required"""
        return self.lookback_period + 10
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate liquidity flow signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with liquidity flow analysis
        """
        try:
            self.validate_input_data(data)
            
            # Estimate bid-ask spreads from OHLC data
            spreads = self._estimate_spreads(data)
            
            # Calculate market depth proxy
            market_depth = self._estimate_market_depth(data)
            
            # Calculate liquidity absorption
            absorption = self._calculate_absorption_rate(data)
            
            # Calculate order imbalance
            imbalance = self._calculate_order_imbalance(data)
            
            # Analyze liquidity flow
            results = []
            for i in range(len(data)):
                if i < self.lookback_period:
                    result = {
                        'liquidity_state': 'normal',
                        'flow_direction': 'neutral',
                        'depth_change': 0.0,
                        'spread_ratio': 1.0,
                        'absorption_rate': 0.0,
                        'provider_strength': 0.5,
                        'taker_aggression': 0.5,
                        'imbalance_score': 0.0
                    }
                else:
                    analysis = self._analyze_liquidity_flow(
                        i, data, spreads, market_depth, absorption, imbalance
                    )
                    
                    result = {
                        'liquidity_state': analysis.liquidity_state,
                        'flow_direction': analysis.flow_direction,
                        'depth_change': analysis.depth_change,
                        'spread_ratio': analysis.spread_ratio,
                        'absorption_rate': analysis.absorption_rate,
                        'provider_strength': analysis.provider_strength,
                        'taker_aggression': analysis.taker_aggression,
                        'imbalance_score': analysis.imbalance_score
                    }
                
                results.append(result)
            
            result_df = pd.DataFrame(results, index=data.index)
            self._last_calculation = result_df
            
            return result_df
            
        except Exception as e:
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _estimate_spreads(self, data: pd.DataFrame) -> pd.Series:
        """Estimate bid-ask spreads from OHLC data"""
        # Use high-low range as spread proxy
        daily_range = data['high'] - data['low']
        
        # Normalize by price
        spread_estimate = daily_range / data['close']
        
        # Smooth the spread estimate
        spread_sma = spread_estimate.rolling(window=5).mean()
        
        return spread_sma.fillna(spread_estimate)
    
    def _estimate_market_depth(self, data: pd.DataFrame) -> pd.Series:
        """Estimate market depth from volume and volatility"""
        # Calculate volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.lookback_period).std()
        
        # Depth proxy: volume / volatility
        depth_proxy = data['volume'] / (volatility * data['close'])
        
        # Normalize and smooth
        depth_normalized = depth_proxy.rolling(window=self.lookback_period).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 0.0001)
        )
        
        return depth_normalized.fillna(0)
    
    def _calculate_absorption_rate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate liquidity absorption rate"""
        # Price impact per unit volume
        price_change = abs(data['close'] - data['close'].shift(1))
        volume_norm = data['volume'] / data['volume'].rolling(window=self.lookback_period).mean()
        
        # Absorption rate: price impact relative to volume
        absorption = price_change / (volume_norm + 0.001)
        absorption_smooth = absorption.rolling(window=5).mean()
        
        return absorption_smooth.fillna(0)
    
    def _calculate_order_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance"""
        # Use typical price and volume to estimate buy/sell pressure
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Money flow multiplier
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.fillna(0)
        
        # Calculate imbalance over window
        imbalance = mfm.rolling(window=self.imbalance_window).mean()
        
        return imbalance.fillna(0)
    
    def _analyze_liquidity_flow(self, index: int, data: pd.DataFrame,
                              spreads: pd.Series, market_depth: pd.Series,
                              absorption: pd.Series, imbalance: pd.Series) -> LiquidityFlowResult:
        """Analyze liquidity flow at specific index"""
        
        # Current metrics
        current_spread = spreads.iloc[index]
        current_depth = market_depth.iloc[index]
        current_absorption = absorption.iloc[index]
        current_imbalance = imbalance.iloc[index]
        
        # Historical averages
        window_start = max(0, index - self.lookback_period)
        
        avg_spread = spreads.iloc[window_start:index].mean()
        avg_depth = market_depth.iloc[window_start:index].mean()
        avg_absorption = absorption.iloc[window_start:index].mean()
        
        # Calculate ratios
        spread_ratio = current_spread / (avg_spread + 0.0001)
        depth_change = current_depth - avg_depth
        
        # Determine liquidity state
        liquidity_state = self._determine_liquidity_state(spread_ratio, depth_change)
        
        # Determine flow direction
        flow_direction = self._determine_flow_direction(
            current_absorption, avg_absorption, current_imbalance
        )
        
        # Calculate provider/taker dynamics
        provider_strength = self._calculate_provider_strength(
            spread_ratio, depth_change, current_absorption
        )
        
        taker_aggression = self._calculate_taker_aggression(
            current_absorption, avg_absorption, current_imbalance
        )
        
        return LiquidityFlowResult(
            liquidity_state=liquidity_state,
            flow_direction=flow_direction,
            depth_change=depth_change,
            spread_ratio=spread_ratio,
            absorption_rate=current_absorption,
            provider_strength=provider_strength,
            taker_aggression=taker_aggression,
            imbalance_score=current_imbalance
        )
    
    def _determine_liquidity_state(self, spread_ratio: float, depth_change: float) -> str:
        """Determine current liquidity state"""
        # High spreads or low depth = scarce liquidity
        if spread_ratio > self.spread_sensitivity or depth_change < -self.depth_threshold:
            return 'scarce'
        # Low spreads and high depth = abundant liquidity
        elif spread_ratio < 1/self.spread_sensitivity and depth_change > self.depth_threshold:
            return 'abundant'
        else:
            return 'normal'
    
    def _determine_flow_direction(self, current_absorption: float, 
                                avg_absorption: float, imbalance: float) -> str:
        """Determine liquidity flow direction"""
        # High absorption with positive imbalance = taking liquidity
        if current_absorption > avg_absorption * 1.5 and abs(imbalance) > 0.3:
            return 'taking'
        # Low absorption with balanced flow = providing liquidity
        elif current_absorption < avg_absorption * 0.7 and abs(imbalance) < 0.2:
            return 'providing'
        else:
            return 'neutral'
    
    def _calculate_provider_strength(self, spread_ratio: float, 
                                   depth_change: float, absorption: float) -> float:
        """Calculate liquidity provider strength"""
        strength = 0.5  # Base strength
        
        # Tight spreads indicate strong providers
        if spread_ratio < 1.0:
            strength += (1.0 - spread_ratio) * 0.3
        
        # Increased depth indicates providers
        if depth_change > 0:
            strength += min(depth_change / 2, 0.3)
        
        # Low absorption indicates providers
        if absorption < 0.1:
            strength += 0.2
        
        return min(max(strength, 0.0), 1.0)
    
    def _calculate_taker_aggression(self, current_absorption: float,
                                  avg_absorption: float, imbalance: float) -> float:
        """Calculate liquidity taker aggression"""
        aggression = 0.5  # Base aggression
        
        # High absorption indicates aggressive taking
        if current_absorption > avg_absorption:
            aggression += min((current_absorption / avg_absorption - 1) * 0.3, 0.3)
        
        # Strong imbalance indicates aggression
        aggression += abs(imbalance) * 0.2
        
        return min(max(aggression, 0.0), 1.0)
    
    def get_metadata(self) -> IndicatorMetadata:
        """Get indicator metadata"""
        return IndicatorMetadata(
            name="Liquidity Flow Signal",
            category=self.CATEGORY,
            description="Analyzes market liquidity flow and provider/taker dynamics",
            parameters={
                "lookback_period": self.lookback_period,
                "spread_sensitivity": self.spread_sensitivity,
                "depth_threshold": self.depth_threshold,
                "imbalance_window": self.imbalance_window
            },
            input_requirements=self._get_required_columns(),
            output_type="DataFrame",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self._get_minimum_data_points()
        )
    
    def get_display_name(self) -> str:
        """Get display name for the indicator"""
        return f"Liquidity Flow ({self.lookback_period})"
    
    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return {
            "lookback_period": self.lookback_period,
            "spread_sensitivity": self.spread_sensitivity,
            "depth_threshold": self.depth_threshold,
            "imbalance_window": self.imbalance_window
        }