"""
CompositeSignal Indicator - Multi-Signal Aggregation Engine
Platform3 Trading Framework
Version: 1.0.0

This indicator combines multiple trading signals using advanced weighting
and aggregation techniques to generate composite trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import logging
from datetime import datetime

import sys
import os

from engines.ai_enhancement.indicators.base_indicator import StandardIndicatorInterface
from engines.ai_enhancement.indicators.base_indicator import IndicatorValidationError


@dataclass
class CompositeSignalConfig:
    """Configuration for CompositeSignal indicator"""
    period: int = 20
    signal_sources: int = 5
    weight_adaptation: bool = True
    confidence_weighting: bool = True
    momentum_factor: float = 0.3
    trend_factor: float = 0.4
    volatility_factor: float = 0.2
    volume_factor: float = 0.1
    decay_factor: float = 0.95
    threshold: float = 0.6


class CompositeSignalIndicator(StandardIndicatorInterface):
    """
    CompositeSignal Indicator v1.0.0
    
    A sophisticated signal aggregation system that combines multiple trading
    signals using weighted averaging, confidence scoring, and adaptive weighting.
    
    Features:
    - Multi-signal aggregation with dynamic weighting
    - Confidence-based signal filtering
    - Adaptive weight adjustment based on performance
    - Signal strength normalization
    - Trend and momentum consideration
    
    Mathematical Foundation:
    The composite signal is calculated as:
    CS(t) = Σ(wi * si * ci) / Σ(wi * ci)
    
    Where:
    - wi = weight of signal i
    - si = normalized signal i
    - ci = confidence of signal i
    
    Weights are updated using:
    wi(t+1) = wi(t) * α + (1-α) * performance_factor
    """
    
    # Class-level metadata
    name = "CompositeSignal"
    version = "1.0.0"
    category = "ml_advanced"
    description = "Multi-signal aggregation engine with adaptive weighting"
    
    def __init__(self, **params):
        """Initialize CompositeSignal indicator"""
        # Extract parameters with defaults
        self.parameters = params
        self.config = CompositeSignalConfig(
            period=self.parameters.get('period', 20),
            signal_sources=self.parameters.get('signal_sources', 5),
            weight_adaptation=self.parameters.get('weight_adaptation', True),
            confidence_weighting=self.parameters.get('confidence_weighting', True),
            momentum_factor=self.parameters.get('momentum_factor', 0.3),
            trend_factor=self.parameters.get('trend_factor', 0.4),
            volatility_factor=self.parameters.get('volatility_factor', 0.2),
            volume_factor=self.parameters.get('volume_factor', 0.1),
            decay_factor=self.parameters.get('decay_factor', 0.95),
            threshold=self.parameters.get('threshold', 0.6)
        )
        
        # Initialize state
        self.reset()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def reset(self):
        """Reset indicator state"""
        self.signal_weights = np.ones(self.config.signal_sources) / self.config.signal_sources
        self.signal_performance = np.ones(self.config.signal_sources) * 0.5
        self.signal_history = []
        self.composite_history = []
        self.confidence_history = []
        
    def calculate(self, data: Union[pd.DataFrame, Dict[str, List], np.ndarray]) -> np.ndarray:
        """
        Calculate CompositeSignal values
        
        Args:
            data: Price data (OHLCV format)
            
        Returns:
            np.ndarray: Composite signals with confidence scores
        """
        try:
            # Input validation
            if data is None or len(data) == 0:
                raise ValidationError("Input data cannot be empty")
                
            # Convert data to DataFrame if needed
            df = self._prepare_data(data)
            
            if len(df) < self.config.period:
                return np.full((len(df), 3), np.nan)  # composite_signal, confidence, final_signal
                
            # Generate individual signals
            individual_signals = self._generate_individual_signals(df)
            
            # Calculate composite signals
            composite_signals = self._calculate_composite_signals(individual_signals)
            
            # Update weights if adaptation is enabled
            if self.config.weight_adaptation:
                self._update_weights(individual_signals, df)
                
            return composite_signals
            
        except Exception as e:
            self.logger.error(f"Error in CompositeSignal calculation: {str(e)}")
            raise CalculationError(f"CompositeSignal calculation failed: {str(e)}")
            
    def _prepare_data(self, data: Any) -> pd.DataFrame:
        """Prepare and validate input data"""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame({'close': data})
            else:
                columns = ['open', 'high', 'low', 'close', 'volume'][:data.shape[1]]
                df = pd.DataFrame(data, columns=columns)
        else:
            raise ValidationError("Unsupported data format")
            
        # Ensure required columns
        if 'close' not in df.columns:
            raise ValidationError("Close price is required")
            
        # Fill missing columns with close price
        for col in ['open', 'high', 'low']:
            if col not in df.columns:
                df[col] = df['close']
                
        if 'volume' not in df.columns:
            df['volume'] = 1.0
            
        return df.dropna()
        
    def _generate_individual_signals(self, df: pd.DataFrame) -> np.ndarray:
        """Generate individual trading signals"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        n_points = len(close)
        signals = np.zeros((n_points, self.config.signal_sources, 2))  # signal, confidence
        
        # Signal 1: Moving Average Crossover
        sma_fast = pd.Series(close).rolling(10).mean().values
        sma_slow = pd.Series(close).rolling(20).mean().values
        ma_signal = np.where(sma_fast > sma_slow, 1, -1)
        ma_confidence = np.abs(sma_fast - sma_slow) / sma_slow
        ma_confidence = np.clip(ma_confidence, 0, 1)
        
        signals[:, 0, 0] = ma_signal
        signals[:, 0, 1] = ma_confidence
        
        # Signal 2: RSI Momentum
        rsi = self._calculate_rsi(close, 14)
        rsi_signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
        rsi_confidence = np.where(rsi < 30, (30 - rsi) / 30, 
                                 np.where(rsi > 70, (rsi - 70) / 30, 0))
        
        signals[:, 1, 0] = rsi_signal
        signals[:, 1, 1] = rsi_confidence
        
        # Signal 3: Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        bb_signal = np.where(bb_position < 0.2, 1, np.where(bb_position > 0.8, -1, 0))
        bb_confidence = np.where(bb_position < 0.2, 0.2 - bb_position,
                                np.where(bb_position > 0.8, bb_position - 0.8, 0))
        bb_confidence = np.clip(bb_confidence * 5, 0, 1)  # Scale confidence
        
        signals[:, 2, 0] = bb_signal
        signals[:, 2, 1] = bb_confidence
        
        # Signal 4: MACD
        macd, macd_signal_line = self._calculate_macd(close)
        macd_crossover = np.where(macd > macd_signal_line, 1, -1)
        macd_strength = np.abs(macd - macd_signal_line)
        macd_confidence = np.clip(macd_strength / np.nanstd(macd_strength), 0, 1)
        
        signals[:, 3, 0] = macd_crossover
        signals[:, 3, 1] = macd_confidence
        
        # Signal 5: Volume-Price Trend
        vpt = self._calculate_vpt(close, volume)
        vpt_signal = np.where(np.diff(vpt, prepend=vpt[0]) > 0, 1, -1)
        vpt_momentum = np.abs(np.diff(vpt, prepend=vpt[0]))
        vpt_confidence = np.clip(vpt_momentum / np.nanstd(vpt_momentum), 0, 1)
        
        signals[:, 4, 0] = vpt_signal
        signals[:, 4, 1] = vpt_confidence
        
        # Clean up NaN values
        signals = np.nan_to_num(signals, 0)
        
        return signals
        
    def _calculate_composite_signals(self, individual_signals: np.ndarray) -> np.ndarray:
        """Calculate composite signals from individual signals"""
        n_points = individual_signals.shape[0]
        composite_results = np.zeros((n_points, 3))  # composite, confidence, final_signal
        
        for i in range(n_points):
            signals = individual_signals[i, :, 0]  # Extract signals
            confidences = individual_signals[i, :, 1]  # Extract confidences
            
            # Apply confidence weighting if enabled
            if self.config.confidence_weighting:
                weights = self.signal_weights * confidences
            else:
                weights = self.signal_weights.copy()
                
            # Normalize weights
            total_weight = np.sum(weights)
            if total_weight > 0:
                weights = weights / total_weight
            else:
                weights = np.ones_like(weights) / len(weights)
                
            # Calculate weighted composite signal
            composite_signal = np.sum(weights * signals)
            
            # Calculate overall confidence
            overall_confidence = np.sum(weights * confidences)
            
            # Generate final signal based on threshold
            if overall_confidence > self.config.threshold:
                if composite_signal > 0.3:
                    final_signal = 1
                elif composite_signal < -0.3:
                    final_signal = -1
                else:
                    final_signal = 0
            else:
                final_signal = 0
                
            composite_results[i] = [composite_signal, overall_confidence, final_signal]
            
        return composite_results
        
    def _update_weights(self, individual_signals: np.ndarray, df: pd.DataFrame):
        """Update signal weights based on performance"""
        if len(individual_signals) < self.config.period:
            return
            
        # Simple performance tracking based on signal accuracy
        recent_signals = individual_signals[-self.config.period:]
        
        # Calculate performance based on signal direction vs price movement
        close_prices = df['close'].values[-self.config.period:]
        price_direction = np.sign(np.diff(close_prices))
        
        for j in range(self.config.signal_sources):
            signal_direction = recent_signals[1:, j, 0]  # Skip first point
            
            # Calculate agreement with price direction
            agreement = np.mean(signal_direction * price_direction)
            
            # Update performance with decay
            self.signal_performance[j] = (self.config.decay_factor * self.signal_performance[j] + 
                                        (1 - self.config.decay_factor) * (agreement + 1) / 2)
            
        # Update weights based on performance
        self.signal_weights = self.signal_performance / np.sum(self.signal_performance)
        
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(period).mean().values
        avg_loss = pd.Series(losses).rolling(period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50], rsi])  # Prepend neutral value
        
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int):
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(period).mean().values
        std = pd.Series(prices).rolling(period).std().values
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        return upper, sma, lower
        
    def _calculate_macd(self, prices: np.ndarray):
        """Calculate MACD"""
        ema12 = pd.Series(prices).ewm(span=12).mean().values
        ema26 = pd.Series(prices).ewm(span=26).mean().values
        macd = ema12 - ema26
        signal_line = pd.Series(macd).ewm(span=9).mean().values
        
        return macd, signal_line
        
    def _calculate_vpt(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate Volume Price Trend"""
        price_change = np.diff(prices, prepend=prices[0])
        vpt = np.cumsum(volumes * (price_change / prices))
        return vpt
        
    def get_signal(self, data: Any) -> int:
        """Get current signal from the indicator"""
        result = self.calculate(data)
        if len(result) == 0:
            return 0
        return int(result[-1, 2])  # Return latest final signal
        
    def get_current_value(self, data: Any) -> float:
        """Get current indicator value"""
        result = self.calculate(data)
        if len(result) == 0:
            return 0.0
        return float(result[-1, 0])  # Return latest composite signal
        
    def validate_parameters(self) -> bool:
        """Validate indicator parameters"""
        try:
            period = self.parameters.get('period', 20)
            if not isinstance(period, (int, float)) or period <= 0:
                return False
                
            signal_sources = self.parameters.get('signal_sources', 5)
            if not isinstance(signal_sources, int) or signal_sources <= 0:
                return False
                
            threshold = self.parameters.get('threshold', 0.6)
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                return False
                
            return True
        except Exception:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return CompositeSignal metadata as dictionary for compatibility"""
        return {
            "name": "CompositeSignal",
            "category": self.CATEGORY,
            "description": "Composite Signal Generator combining multiple signal sources using advanced aggregation",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "Dict",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self._get_minimum_data_points(),
        }

    def _get_required_columns(self) -> List[str]:
        """CompositeSignal can work with OHLCV data"""
        return ["open", "high", "low", "close", "volume"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for composite signal calculation"""
        return self.parameters.get("period", 20)


def get_composite_signal_indicator(**params) -> CompositeSignalIndicator:
    """
    Factory function to create CompositeSignal indicator
    
    Args:
        **params: Indicator parameters
        
    Returns:
        CompositeSignalIndicator: Configured indicator instance
    """
    return CompositeSignalIndicator(**params)


# Export for registry discovery
__all__ = ['CompositeSignalIndicator', 'get_composite_signal_indicator']