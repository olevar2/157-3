#!/usr/bin/env python3
"""
TimeframeConfig - Multi-Timeframe Configuration and Analysis Indicator

A comprehensive timeframe management and analysis indicator that handles
multiple timeframe configurations for technical analysis synchronization.

Author: Platform3 AI Enhancement Engine
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import sys
import os
from datetime import datetime, timedelta
from enum import Enum

# Add parent directories to path for imports

try:
    from engines.ai_enhancement.indicators.base_indicator import BaseIndicator
except ImportError:
    # For direct script execution, create a minimal base class
    class BaseIndicator:
        def __init__(self, name: str):
            self.name = name


class TimeframeType(Enum):
    """Supported timeframe types."""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class TimeframeConfig(BaseIndicator):
    """
    Multi-Timeframe Configuration Indicator
    
    Manages and analyzes multiple timeframe configurations for synchronized
    technical analysis. Provides timeframe alignment, conversion utilities,
    and multi-timeframe signal correlation.
    
    The indicator provides:
    - Timeframe hierarchy management
    - Data alignment across timeframes
    - Timeframe strength analysis
    - Signal synchronization
    - Trend consistency analysis
    """
    
    def __init__(self, 
                 primary_timeframe: str = "1H",
                 secondary_timeframes: List[str] = None,
                 alignment_tolerance: float = 0.1,
                 min_data_points: int = 50):
        """
        Initialize TimeframeConfig indicator.
        
        Args:
            primary_timeframe: Primary timeframe for analysis (default: "1H")
            secondary_timeframes: List of secondary timeframes
            alignment_tolerance: Tolerance for timeframe alignment (default: 0.1)
            min_data_points: Minimum data points required (default: 50)
        """
        super().__init__("TimeframeConfig")
        self.primary_timeframe = primary_timeframe
        
        if secondary_timeframes is None:
            self.secondary_timeframes = ["15T", "4H", "1D"]
        else:
            self.secondary_timeframes = secondary_timeframes
        
        self.alignment_tolerance = max(0.01, alignment_tolerance)
        self.min_data_points = max(10, min_data_points)
        
        # Timeframe hierarchy (in minutes)
        self.timeframe_hierarchy = {
            "1T": 1, "5T": 5, "15T": 15, "30T": 30,
            "1H": 60, "2H": 120, "4H": 240, "8H": 480,
            "1D": 1440, "1W": 10080, "1M": 43200
        }
        
        # State variables
        self.timeframe_data = {}
        self.alignment_status = {}
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate timeframe configuration signals.
        
        Args:
            data: DataFrame with OHLCV columns and datetime index
            
        Returns:
            Dictionary containing:
            - timeframe_strength: Strength across different timeframes
            - alignment_score: Timeframe alignment quality
            - trend_consistency: Trend consistency across timeframes
            - volatility_profile: Volatility across timeframes
            - signal_quality: Overall signal quality score
            - primary_signal: Main trading signal (-1 to 1)
        """
        try:
            if len(data) < self.min_data_points:
                return self._empty_result(len(data))
            
            # Analyze primary timeframe
            primary_analysis = self._analyze_timeframe(data, self.primary_timeframe)
            
            # Analyze secondary timeframes (simulated)
            secondary_analyses = {}
            for tf in self.secondary_timeframes:
                secondary_analyses[tf] = self._simulate_timeframe_analysis(data, tf)
            
            # Calculate timeframe strength
            timeframe_strength = self._calculate_timeframe_strength(
                primary_analysis, secondary_analyses)
            
            # Calculate alignment score
            alignment_score = self._calculate_alignment_score(
                primary_analysis, secondary_analyses)
            
            # Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(
                primary_analysis, secondary_analyses)
            
            # Calculate volatility profile
            volatility_profile = self._calculate_volatility_profile(
                primary_analysis, secondary_analyses)
            
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(
                timeframe_strength, alignment_score, trend_consistency)
            
            # Generate primary signal
            primary_signal = self._generate_primary_signal(
                primary_analysis, signal_quality, trend_consistency)
            
            return {
                'timeframe_strength': timeframe_strength,
                'alignment_score': alignment_score,
                'trend_consistency': trend_consistency,
                'volatility_profile': volatility_profile,
                'signal_quality': signal_quality,
                'primary_signal': primary_signal
            }
            
        except Exception as e:
            warnings.warn(f"Error in TimeframeConfig calculation: {str(e)}")
            return self._empty_result(len(data))
    
    def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, np.ndarray]:
        """Analyze a specific timeframe."""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        volume = data['volume'].values
        
        # Calculate basic indicators for this timeframe
        returns = np.diff(np.log(close))
        returns = np.concatenate([[0], returns])
        
        # Trend analysis
        window = min(20, len(close) // 4)
        if window > 1:
            trend = pd.Series(close).rolling(window=window).mean().values
            trend_direction = np.diff(trend)
            trend_direction = np.concatenate([[0], trend_direction])
        else:
            trend = close.copy()
            trend_direction = np.zeros_like(close)
        
        # Volatility analysis
        if len(returns) > 1:
            volatility = pd.Series(returns).rolling(
                window=min(10, len(returns))).std().values
        else:
            volatility = np.zeros_like(close)
        
        # Volume analysis
        if len(volume) > 1:
            volume_ma = pd.Series(volume).rolling(
                window=min(10, len(volume))).mean().values
            volume_ratio = volume / np.maximum(volume_ma, 1)
        else:
            volume_ratio = np.ones_like(volume)
        
        return {
            'trend': trend,
            'trend_direction': trend_direction,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'returns': returns
        }
    
    def _simulate_timeframe_analysis(self, data: pd.DataFrame, 
                                   timeframe: str) -> Dict[str, np.ndarray]:
        """Simulate analysis for different timeframes."""
        # Get timeframe multiplier
        primary_minutes = self.timeframe_hierarchy.get(self.primary_timeframe, 60)
        target_minutes = self.timeframe_hierarchy.get(timeframe, 60)
        
        ratio = target_minutes / primary_minutes
        
        # Resample data conceptually
        if ratio > 1:
            # Higher timeframe - aggregate data
            window = int(ratio)
            window = max(1, min(window, len(data) // 2))
        else:
            # Lower timeframe - interpolate data
            window = max(1, int(1 / ratio))
        
        # Simulate the analysis with adjusted parameters
        analysis = self._analyze_timeframe(data, timeframe)
        
        # Adjust for timeframe characteristics
        if ratio > 1:
            # Higher timeframe - smoother signals
            for key in analysis:
                if len(analysis[key]) > window:
                    analysis[key] = pd.Series(analysis[key]).rolling(
                        window=window, min_periods=1).mean().values
        else:
            # Lower timeframe - more volatile signals
            for key in analysis:
                if key == 'volatility':
                    analysis[key] = analysis[key] * (1 + (1 / ratio - 1) * 0.5)
        
        return analysis
    
    def _calculate_timeframe_strength(self, primary_analysis: Dict[str, np.ndarray],
                                    secondary_analyses: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Calculate strength across timeframes."""
        length = len(primary_analysis['trend'])
        
        # Primary timeframe strength
        primary_trend_strength = np.abs(primary_analysis['trend_direction'])
        primary_trend_strength = primary_trend_strength / (
            np.max(primary_trend_strength) + 1e-8)
        
        # Secondary timeframe strengths
        secondary_strengths = []
        for tf, analysis in secondary_analyses.items():
            if len(analysis['trend_direction']) == length:
                strength = np.abs(analysis['trend_direction'])
                strength = strength / (np.max(strength) + 1e-8)
                secondary_strengths.append(strength)
        
        # Combine strengths
        if secondary_strengths:
            combined_secondary = np.mean(secondary_strengths, axis=0)
            timeframe_strength = 0.6 * primary_trend_strength + 0.4 * combined_secondary
        else:
            timeframe_strength = primary_trend_strength
        
        return timeframe_strength
    
    def _calculate_alignment_score(self, primary_analysis: Dict[str, np.ndarray],
                                 secondary_analyses: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Calculate timeframe alignment score."""
        length = len(primary_analysis['trend_direction'])
        alignment_scores = []
        
        primary_direction = np.sign(primary_analysis['trend_direction'])
        
        for tf, analysis in secondary_analyses.items():
            if len(analysis['trend_direction']) == length:
                secondary_direction = np.sign(analysis['trend_direction'])
                
                # Calculate agreement
                agreement = (primary_direction == secondary_direction).astype(float)
                
                # Weight by timeframe importance
                tf_minutes = self.timeframe_hierarchy.get(tf, 60)
                primary_minutes = self.timeframe_hierarchy.get(self.primary_timeframe, 60)
                
                if tf_minutes > primary_minutes:
                    weight = 1.5  # Higher timeframes more important
                elif tf_minutes < primary_minutes:
                    weight = 0.8  # Lower timeframes less important
                else:
                    weight = 1.0
                
                weighted_agreement = agreement * weight
                alignment_scores.append(weighted_agreement)
        
        if alignment_scores:
            alignment_score = np.mean(alignment_scores, axis=0)
        else:
            alignment_score = np.ones(length) * 0.5
        
        return alignment_score
    
    def _calculate_trend_consistency(self, primary_analysis: Dict[str, np.ndarray],
                                   secondary_analyses: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Calculate trend consistency across timeframes."""
        length = len(primary_analysis['trend_direction'])
        
        # Calculate trend strength in primary timeframe
        primary_strength = np.abs(primary_analysis['trend_direction'])
        
        # Calculate consistency with secondary timeframes
        consistency_scores = []
        
        for tf, analysis in secondary_analyses.items():
            if len(analysis['trend_direction']) == length:
                secondary_strength = np.abs(analysis['trend_direction'])
                
                # Normalize strengths
                primary_norm = primary_strength / (np.max(primary_strength) + 1e-8)
                secondary_norm = secondary_strength / (np.max(secondary_strength) + 1e-8)
                
                # Calculate consistency as correlation-like measure
                consistency = 1 - np.abs(primary_norm - secondary_norm)
                consistency_scores.append(consistency)
        
        if consistency_scores:
            trend_consistency = np.mean(consistency_scores, axis=0)
        else:
            trend_consistency = np.ones(length) * 0.5
        
        return trend_consistency
    
    def _calculate_volatility_profile(self, primary_analysis: Dict[str, np.ndarray],
                                    secondary_analyses: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Calculate volatility profile across timeframes."""
        length = len(primary_analysis['volatility'])
        
        primary_vol = primary_analysis['volatility']
        volatilities = [primary_vol]
        
        for tf, analysis in secondary_analyses.items():
            if len(analysis['volatility']) == length:
                volatilities.append(analysis['volatility'])
        
        if len(volatilities) > 1:
            # Calculate relative volatility strength
            vol_array = np.array(volatilities)
            volatility_profile = np.mean(vol_array, axis=0)
            
            # Normalize
            volatility_profile = volatility_profile / (
                np.max(volatility_profile) + 1e-8)
        else:
            volatility_profile = primary_vol / (np.max(primary_vol) + 1e-8)
        
        return volatility_profile
    
    def _calculate_signal_quality(self, timeframe_strength: np.ndarray,
                                alignment_score: np.ndarray,
                                trend_consistency: np.ndarray) -> np.ndarray:
        """Calculate overall signal quality."""
        # Combine factors with weights
        signal_quality = (0.4 * timeframe_strength + 
                         0.35 * alignment_score + 
                         0.25 * trend_consistency)
        
        return np.clip(signal_quality, 0, 1)
    
    def _generate_primary_signal(self, primary_analysis: Dict[str, np.ndarray],
                               signal_quality: np.ndarray,
                               trend_consistency: np.ndarray) -> np.ndarray:
        """Generate primary trading signal."""
        trend_direction = primary_analysis['trend_direction']
        
        # Normalize trend direction
        max_trend = np.max(np.abs(trend_direction)) + 1e-8
        normalized_trend = trend_direction / max_trend
        
        # Apply quality and consistency filters
        signal = normalized_trend * signal_quality * trend_consistency
        
        # Apply smoothing
        if len(signal) >= 3:
            signal = pd.Series(signal).rolling(
                window=3, min_periods=1).mean().values
        
        return np.clip(signal, -1.0, 1.0)
    
    def _empty_result(self, length: int) -> Dict[str, np.ndarray]:
        """Return empty result arrays."""
        return {
            'timeframe_strength': np.full(length, 0.5),
            'alignment_score': np.full(length, 0.5),
            'trend_consistency': np.full(length, 0.5),
            'volatility_profile': np.full(length, 0.5),
            'signal_quality': np.full(length, 0.5),
            'primary_signal': np.zeros(length)
        }
    
    def get_timeframe_info(self) -> Dict[str, Any]:
        """Get information about configured timeframes."""
        return {
            'primary_timeframe': self.primary_timeframe,
            'secondary_timeframes': self.secondary_timeframes,
            'timeframe_hierarchy': self.timeframe_hierarchy,
            'total_timeframes': len(self.secondary_timeframes) + 1
        }
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get trading signals as a DataFrame.
        
        Args:
            data: Input OHLCV data
            
        Returns:
            DataFrame with signal columns
        """
        result = self.calculate(data)
        
        signals_df = pd.DataFrame(index=data.index)
        for key, value in result.items():
            signals_df[f'timeframe_{key}'] = value
        
        return signals_df
    
    def analyze_timeframe_hierarchy(self) -> Dict[str, Union[str, int]]:
        """Analyze the timeframe hierarchy setup."""
        all_timeframes = [self.primary_timeframe] + self.secondary_timeframes
        timeframe_minutes = []
        
        for tf in all_timeframes:
            minutes = self.timeframe_hierarchy.get(tf, 0)
            timeframe_minutes.append((tf, minutes))
        
        # Sort by timeframe length
        timeframe_minutes.sort(key=lambda x: x[1])
        
        return {
            'shortest_timeframe': timeframe_minutes[0][0],
            'longest_timeframe': timeframe_minutes[-1][0],
            'timeframe_range_ratio': timeframe_minutes[-1][1] / max(timeframe_minutes[0][1], 1),
            'total_range_minutes': timeframe_minutes[-1][1] - timeframe_minutes[0][1]
        }


def demonstrate_timeframe_config():
    """Demonstrate TimeframeConfig indicator usage."""
    print("=" * 50)
    print("TimeframeConfig Indicator Demonstration")
    print("=" * 50)
    
    # Generate sample data with datetime index
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='H')
    
    # Create realistic OHLCV data
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Add multiple timeframe effects
        daily_trend = 0.001 * np.sin(i * 2 * np.pi / 24)  # Daily cycle
        weekly_trend = 0.005 * np.sin(i * 2 * np.pi / (24 * 7))  # Weekly cycle
        noise = np.random.normal(0, 0.01)
        
        price_change = daily_trend + weekly_trend + noise
        
        if i == 0:
            price = base_price
        else:
            price = prices[-1] * (1 + price_change)
        
        # Create OHLC from base price
        volatility = abs(price_change) * 2
        high = price * (1 + volatility * np.random.uniform(0, 1))
        low = price * (1 - volatility * np.random.uniform(0, 1))
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        close = low + (high - low) * np.random.uniform(0.2, 0.8)
        
        prices.append(close)
        
        # Volume with timeframe patterns
        base_volume = 1000
        daily_volume_cycle = 200 * abs(np.sin(i * 2 * np.pi / 24))
        random_volume = np.random.exponential(500)
        
        volume = int(base_volume + daily_volume_cycle + random_volume)
        volumes.append(volume)
    
    data = pd.DataFrame({
        'open': [prices[0]] + [p * 0.999 for p in prices[:-1]],
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Test different configurations
    configs = [
        {
            "primary_timeframe": "1H",
            "secondary_timeframes": ["15T", "4H"],
            "name": "Intraday Focus"
        },
        {
            "primary_timeframe": "4H",
            "secondary_timeframes": ["1H", "1D"],
            "name": "Swing Trading"
        },
        {
            "primary_timeframe": "1D",
            "secondary_timeframes": ["4H", "1W"],
            "name": "Position Trading"
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print(f"Primary: {config['primary_timeframe']}")
        print(f"Secondary: {config['secondary_timeframes']}")
        
        # Create and calculate indicator
        indicator = TimeframeConfig(
            primary_timeframe=config['primary_timeframe'],
            secondary_timeframes=config['secondary_timeframes']
        )
        
        # Show timeframe hierarchy
        hierarchy = indicator.analyze_timeframe_hierarchy()
        print(f"\nTimeframe Hierarchy:")
        print(f"  Shortest: {hierarchy['shortest_timeframe']}")
        print(f"  Longest: {hierarchy['longest_timeframe']}")
        print(f"  Range Ratio: {hierarchy['timeframe_range_ratio']:.1f}x")
        
        result = indicator.calculate(data)
        
        # Display statistics
        print(f"\nResults Summary:")
        for key, values in result.items():
            if len(values) > 0 and not np.all(np.isnan(values)):
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    print(f"{key}:")
                    print(f"  Range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
                    print(f"  Mean: {valid_values.mean():.4f}")
                    print(f"  Std: {valid_values.std():.4f}")
        
        # Show recent signals
        print(f"\nRecent Analysis (last 10 periods):")
        recent_signals = result['primary_signal'][-10:]
        recent_quality = result['signal_quality'][-10:]
        recent_alignment = result['alignment_score'][-10:]
        
        for i in range(len(recent_signals)):
            date = data.index[-10 + i].strftime('%Y-%m-%d %H:%M')
            signal = recent_signals[i]
            quality = recent_quality[i]
            alignment = recent_alignment[i]
            print(f"  {date}: Signal={signal:.3f}, Quality={quality:.3f}, Alignment={alignment:.3f}")
        
        # Current analysis
        current_signal = result['primary_signal'][-1]
        current_quality = result['signal_quality'][-1]
        current_alignment = result['alignment_score'][-1]
        current_consistency = result['trend_consistency'][-1]
        
        print(f"\nCurrent Multi-Timeframe Analysis:")
        print(f"  Primary Signal: {current_signal:.4f}")
        print(f"  Signal Quality: {current_quality:.4f}")
        print(f"  Timeframe Alignment: {current_alignment:.4f}")
        print(f"  Trend Consistency: {current_consistency:.4f}")
        
        if current_signal > 0.3 and current_quality > 0.6:
            print("  -> Strong Buy (High quality, aligned timeframes)")
        elif current_signal > 0.1 and current_quality > 0.4:
            print("  -> Buy (Moderate quality)")
        elif current_signal < -0.3 and current_quality > 0.6:
            print("  -> Strong Sell (High quality, aligned timeframes)")
        elif current_signal < -0.1 and current_quality > 0.4:
            print("  -> Sell (Moderate quality)")
        else:
            print("  -> Neutral/Hold (Mixed or low quality signals)")
    
    # Test edge cases
    print(f"\n" + "="*50)
    print("Edge Case Testing:")
    print("="*50)
    
    # Test with minimal data
    small_data = data.head(20)
    indicator = TimeframeConfig()
    result = indicator.calculate(small_data)
    print(f"Small dataset test - Signal range: [{result['primary_signal'].min():.3f}, {result['primary_signal'].max():.3f}]")
    
    # Test with single timeframe
    single_tf_indicator = TimeframeConfig(
        primary_timeframe="1H",
        secondary_timeframes=[]
    )
    result = single_tf_indicator.calculate(data)
    print(f"Single timeframe test - Signal range: [{result['primary_signal'].min():.3f}, {result['primary_signal'].max():.3f}]")
    
    print(f"\n" + "="*50)
    print("TimeframeConfig demonstration completed!")
    print("="*50)


if __name__ == "__main__":
    demonstrate_timeframe_config()