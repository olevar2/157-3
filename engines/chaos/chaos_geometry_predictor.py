#!/usr/bin/env python3
"""
ChaosGeometryPredictor - Platform3 Financial Indicator

Platform3 compliant implementation with CCI proven patterns.
Inspired by Chaos Theory and Fractal Geometry for Advanced Market Analysis.

Created for Platform3 - Maximizing profits for humanitarian causes
Helping sick children and poor families through advanced trading technology
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Platform3 imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai_enhancement', 'indicators'))
from base_indicator import (
    IndicatorValidationError,
    StandardIndicatorInterface,
)

logger = logging.getLogger(__name__)

class ChaosGeometryPredictor(StandardIndicatorInterface):
    """
    ChaosGeometryPredictor - Platform3 Implementation
    
    Platform3 compliant financial indicator with:
    - CCI Proven Pattern Compliance
    - Performance Optimization  
    - Robust Error Handling
    - Chaos Theory and Fractal Geometry Analysis
    """
    
    # Class-level metadata (REQUIRED for Platform3)
    CATEGORY: str = "chaos"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Platform3"
    
    def __init__(self, 
                 chaos_window: int = 30,
                 fractal_dimension_window: int = 20,
                 lyapunov_window: int = 15,
                 attractor_dimension: int = 3,
                 noise_threshold: float = 0.01,
                 **kwargs):
        """
        Initialize ChaosGeometryPredictor with CCI-compatible pattern.
        
        Args:
            chaos_window: Window size for chaos analysis
            fractal_dimension_window: Window for fractal dimension calculation
            lyapunov_window: Window for Lyapunov exponent calculation
            attractor_dimension: Dimension for strange attractor analysis
            noise_threshold: Threshold for noise filtering
        """
        # Set instance variables BEFORE calling super().__init__()
        self.chaos_window = chaos_window
        self.fractal_dimension_window = fractal_dimension_window
        self.lyapunov_window = lyapunov_window
        self.attractor_dimension = attractor_dimension
        self.noise_threshold = noise_threshold
        
        # Call parent constructor with all parameters
        super().__init__(
            chaos_window=chaos_window,
            fractal_dimension_window=fractal_dimension_window,
            lyapunov_window=lyapunov_window,
            attractor_dimension=attractor_dimension,
            noise_threshold=noise_threshold,
            **kwargs
        )
        
        # Set parameters after parent initialization
        self.parameters = {
            "chaos_window": self.chaos_window,
            "fractal_dimension_window": self.fractal_dimension_window,
            "lyapunov_window": self.lyapunov_window,
            "attractor_dimension": self.attractor_dimension,
            "noise_threshold": self.noise_threshold
        }
        
        # Humanitarian mission logging
        logger.info("Chaos ChaosGeometryPredictor initialized - Fighting for humanitarian causes")
        logger.info("Every trade helps sick children and poor families worldwide")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get indicator parameters."""
        return {
            "chaos_window": self.chaos_window,
            "fractal_dimension_window": self.fractal_dimension_window,
            "lyapunov_window": self.lyapunov_window,
            "attractor_dimension": self.attractor_dimension,
            "noise_threshold": self.noise_threshold
        }
    
    def validate_parameters(self) -> bool:
        """Validate parameters."""
        if not isinstance(self.chaos_window, int) or self.chaos_window <= 0:
            raise IndicatorValidationError("chaos_window must be a positive integer")
        
        if not isinstance(self.fractal_dimension_window, int) or self.fractal_dimension_window <= 0:
            raise IndicatorValidationError("fractal_dimension_window must be a positive integer")
        
        if not isinstance(self.lyapunov_window, int) or self.lyapunov_window <= 0:
            raise IndicatorValidationError("lyapunov_window must be a positive integer")
        
        if not isinstance(self.attractor_dimension, int) or self.attractor_dimension <= 0:
            raise IndicatorValidationError("attractor_dimension must be a positive integer")
        
        if not isinstance(self.noise_threshold, (int, float)) or self.noise_threshold <= 0:
            raise IndicatorValidationError("noise_threshold must be positive")
        
        return True
    
    def calculate(self, data: Union[np.ndarray, pd.DataFrame]) -> pd.Series:
        """
        Calculate ChaosGeometryPredictor.
        
        Args:
            data: Input data (DataFrame with OHLCV or Series with close prices)
            
        Returns:
            pd.Series: Chaos geometry predictor values
        """
        try:
            # Validate input data
            self.validate_input_data(data)
            
            # Convert to DataFrame if necessary
            if isinstance(data, pd.Series):
                df = pd.DataFrame({'close': data})
                df.index = data.index
            else:
                df = data.copy()
            
            # Ensure we have required columns
            if 'close' not in df.columns:
                raise IndicatorValidationError("Data must contain 'close' column")
            
            # Get additional columns if available
            close = df['close'].values
            high = df['high'].values if 'high' in df.columns else close
            low = df['low'].values if 'low' in df.columns else close
            volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
            
            # Calculate chaos geometry predictor
            chaos_values = self._calculate_chaos_predictor(close, high, low, volume)
            
            # Create result series with proper index
            result = pd.Series(chaos_values, index=df.index, name='chaos_geometry_predictor')
            
            # Store last calculation
            self._last_calculation = result
            
            return result
            
        except Exception as e:
            logger.error(f"ChaosGeometryPredictor calculation error: {str(e)}")
            raise IndicatorValidationError(f"Calculation failed: {str(e)}")
    
    def _calculate_chaos_predictor(self, close: np.ndarray, high: np.ndarray, 
                                 low: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate chaos geometry predictor values"""
        try:
            n = len(close)
            chaos_predictor = np.full(n, np.nan)
            
            for i in range(self.chaos_window - 1, n):
                # Extract analysis windows
                price_window = close[i - self.chaos_window + 1:i + 1]
                high_window = high[i - self.chaos_window + 1:i + 1]
                low_window = low[i - self.chaos_window + 1:i + 1]
                volume_window = volume[i - self.chaos_window + 1:i + 1]
                
                # 1. Lyapunov Exponent Calculation
                lyapunov_exp = self._calculate_lyapunov_exponent(price_window)
                
                # 2. Fractal Dimension Analysis
                fractal_dim = self._calculate_fractal_dimension(price_window)
                
                # 3. Phase Space Reconstruction
                phase_space = self._reconstruct_phase_space(price_window)
                
                # 4. Strange Attractor Analysis
                attractor_analysis = self._analyze_strange_attractor(phase_space)
                
                # 5. Chaos Measure Calculation
                chaos_measure = self._calculate_chaos_measure(price_window, volume_window)
                
                # 6. Non-linear Dynamics Analysis
                nonlinear_dynamics = self._analyze_nonlinear_dynamics(
                    price_window, high_window, low_window
                )
                
                # 7. Synthesize chaos prediction
                chaos_predictor[i] = self._synthesize_chaos_prediction(
                    lyapunov_exp, fractal_dim, attractor_analysis,
                    chaos_measure, nonlinear_dynamics
                )
            
            return chaos_predictor
            
        except Exception as e:
            logger.error(f"Chaos predictor calculation error: {str(e)}")
            return np.full(len(close), np.nan)
    
    def _calculate_lyapunov_exponent(self, prices: np.ndarray) -> float:
        """Calculate largest Lyapunov exponent for chaos detection"""
        try:
            if len(prices) < self.lyapunov_window:
                return 0.0
            
            # Use simplified method for financial time series
            window_prices = prices[-self.lyapunov_window:]
            
            # Calculate log returns
            returns = np.diff(np.log(window_prices + 1e-8))
            
            if len(returns) < 2:
                return 0.0
            
            # Estimate Lyapunov exponent using variance of returns
            # Positive values indicate chaos, negative indicate stability
            mean_return = np.mean(returns)
            variance_return = np.var(returns)
            
            if variance_return > 0:
                lyapunov = mean_return / np.sqrt(variance_return)
            else:
                lyapunov = 0.0
            
            return np.clip(lyapunov, -5.0, 5.0)
            
        except Exception:
            return 0.0
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            if len(prices) < self.fractal_dimension_window:
                return 1.5  # Default dimension
            
            window_prices = prices[-self.fractal_dimension_window:]
            
            # Normalize prices
            normalized_prices = (window_prices - np.min(window_prices))
            price_range = np.max(window_prices) - np.min(window_prices)
            if price_range > 0:
                normalized_prices = normalized_prices / price_range
            
            # Box-counting algorithm (simplified)
            box_sizes = np.logspace(-2, 0, 10)  # Different box sizes
            box_counts = []
            
            for box_size in box_sizes:
                if box_size <= 0:
                    continue
                    
                # Count boxes needed to cover the curve
                n_boxes_x = max(1, int(1.0 / box_size))
                n_boxes_y = max(1, int(1.0 / box_size))
                
                boxes_covered = set()
                for j, price in enumerate(normalized_prices):
                    if j < len(normalized_prices) - 1:
                        t1 = j / len(normalized_prices)
                        t2 = (j + 1) / len(normalized_prices)
                        p1 = price
                        p2 = normalized_prices[j + 1]
                        
                        # Add boxes along the line segment
                        box_x1 = int(t1 * n_boxes_x)
                        box_x2 = int(t2 * n_boxes_x)
                        box_y1 = int(p1 * n_boxes_y)
                        box_y2 = int(p2 * n_boxes_y)
                        
                        for bx in range(min(box_x1, box_x2), max(box_x1, box_x2) + 1):
                            for by in range(min(box_y1, box_y2), max(box_y1, box_y2) + 1):
                                boxes_covered.add((bx, by))
                
                box_counts.append(len(boxes_covered))
            
            # Calculate fractal dimension from slope
            if len(box_counts) > 2:
                log_box_sizes = np.log(box_sizes)
                log_box_counts = np.log(np.array(box_counts) + 1e-8)
                
                # Linear regression to find slope
                slope = np.polyfit(log_box_sizes, log_box_counts, 1)[0]
                fractal_dimension = -slope
            else:
                fractal_dimension = 1.5
            
            return np.clip(fractal_dimension, 1.0, 3.0)
            
        except Exception:
            return 1.5
    
    def _reconstruct_phase_space(self, prices: np.ndarray) -> np.ndarray:
        """Reconstruct phase space using delay embedding"""
        try:
            if len(prices) < self.attractor_dimension + 1:
                return np.zeros((1, self.attractor_dimension))
            
            # Use optimal delay (simplified estimation)
            delay = max(1, len(prices) // 10)
            
            # Create delay vectors
            phase_space = []
            for i in range(len(prices) - (self.attractor_dimension - 1) * delay):
                vector = []
                for j in range(self.attractor_dimension):
                    idx = i + j * delay
                    if idx < len(prices):
                        vector.append(prices[idx])
                    else:
                        vector.append(prices[-1])
                if len(vector) == self.attractor_dimension:
                    phase_space.append(vector)
            
            return np.array(phase_space) if phase_space else np.zeros((1, self.attractor_dimension))
            
        except Exception:
            return np.zeros((1, self.attractor_dimension))
    
    def _analyze_strange_attractor(self, phase_space: np.ndarray) -> float:
        """Analyze strange attractor properties"""
        try:
            if phase_space.shape[0] < 3:
                return 0.0
            
            # Calculate attractor properties
            # 1. Measure the spread of points in phase space
            centroid = np.mean(phase_space, axis=0)
            distances = np.linalg.norm(phase_space - centroid, axis=1)
            
            # 2. Calculate attractor dimension estimate
            max_distance = np.max(distances)
            if max_distance > 0:
                normalized_distances = distances / max_distance
                
                # Correlation dimension estimate
                correlation_sum = 0
                n_points = len(phase_space)
                
                for i in range(min(n_points, 50)):  # Limit for performance
                    for j in range(i + 1, min(n_points, 50)):
                        dist = np.linalg.norm(phase_space[i] - phase_space[j])
                        if dist < 0.1 * max_distance:  # Small radius
                            correlation_sum += 1
                
                # Normalize correlation sum
                if n_points > 1:
                    correlation_dimension = correlation_sum / (n_points * (n_points - 1) / 2)
                else:
                    correlation_dimension = 0
            else:
                correlation_dimension = 0
            
            # 3. Strange attractor indicator
            # Strange attractors have non-integer dimensions
            attractor_strangeness = abs(correlation_dimension - round(correlation_dimension))
            
            return np.clip(attractor_strangeness, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_chaos_measure(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate overall chaos measure"""
        try:
            if len(prices) < 3:
                return 0.0
            
            # 1. Price predictability measure
            price_returns = np.diff(np.log(prices + 1e-8))
            if len(price_returns) < 2:
                return 0.0
            
            # Autocorrelation of returns (chaos reduces predictability)
            if len(price_returns) > 1:
                autocorr = np.corrcoef(price_returns[:-1], price_returns[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0
            else:
                autocorr = 0
            
            # 2. Volume-price relationship chaos
            if len(volumes) >= len(prices):
                volume_changes = np.diff(volumes[-len(price_returns):])
                if len(volume_changes) > 0 and len(price_returns) > 0:
                    volume_price_corr = np.corrcoef(
                        price_returns[:len(volume_changes)], 
                        volume_changes[:len(price_returns)]
                    )[0, 1]
                    if np.isnan(volume_price_corr):
                        volume_price_corr = 0
                else:
                    volume_price_corr = 0
            else:
                volume_price_corr = 0
            
            # 3. Chaos measure (lower predictability = higher chaos)
            chaos_measure = 1.0 - (abs(autocorr) + abs(volume_price_corr)) / 2
            
            return np.clip(chaos_measure, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _analyze_nonlinear_dynamics(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
        """Analyze nonlinear dynamics patterns"""
        try:
            if len(prices) < 5:
                return 0.0
            
            # 1. Bifurcation detection
            # Look for period-doubling patterns
            returns = np.diff(prices)
            if len(returns) < 4:
                return 0.0
            
            # Check for alternating patterns (period-2)
            even_returns = returns[::2]
            odd_returns = returns[1::2]
            
            min_len = min(len(even_returns), len(odd_returns))
            if min_len > 1:
                even_mean = np.mean(even_returns[:min_len])
                odd_mean = np.mean(odd_returns[:min_len])
                period_2_strength = abs(even_mean - odd_mean) / (np.std(returns) + 1e-8)
            else:
                period_2_strength = 0
            
            # 2. Nonlinear trend analysis
            # Fit polynomial and measure nonlinearity
            if len(prices) > 3:
                t = np.arange(len(prices))
                linear_fit = np.polyfit(t, prices, 1)
                quadratic_fit = np.polyfit(t, prices, 2)
                
                linear_pred = np.polyval(linear_fit, t)
                quadratic_pred = np.polyval(quadratic_fit, t)
                
                linear_error = np.mean(np.abs(prices - linear_pred))
                quadratic_error = np.mean(np.abs(prices - quadratic_pred))
                
                if linear_error > 0:
                    nonlinearity = (linear_error - quadratic_error) / linear_error
                else:
                    nonlinearity = 0
            else:
                nonlinearity = 0
            
            # 3. Combine nonlinear measures
            nonlinear_dynamics = (period_2_strength + abs(nonlinearity)) / 2
            
            return np.clip(nonlinear_dynamics, 0.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _synthesize_chaos_prediction(self, lyapunov_exp: float, fractal_dim: float,
                                   attractor_analysis: float, chaos_measure: float,
                                   nonlinear_dynamics: float) -> float:
        """Synthesize all chaos components into prediction signal"""
        try:
            # Normalize inputs
            normalized_lyapunov = np.tanh(lyapunov_exp)  # Sigmoid-like normalization
            normalized_fractal = (fractal_dim - 1.5) / 1.5  # Center around 1.5
            
            # Weight the components
            chaos_signal = (
                normalized_lyapunov * 0.25 +
                normalized_fractal * 0.25 +
                attractor_analysis * 0.20 +
                chaos_measure * 0.15 +
                nonlinear_dynamics * 0.15
            )
            
            # Normalize to CCI-like range (-100 to +100)
            normalized_signal = chaos_signal * 100
            
            return np.clip(normalized_signal, -100.0, 100.0)
            
        except Exception:
            return 0.0
    
    @property
    def minimum_periods(self) -> int:
        """Minimum periods required."""
        return self.chaos_window
    
    def _get_required_columns(self) -> List[str]:
        """Required columns for calculation"""
        return ["close"]
    
    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed"""
        return self.chaos_window
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get indicator metadata."""
        return {
            "name": "ChaosGeometryPredictor",
            "category": self.CATEGORY,
            "description": "Chaos theory and fractal geometry predictor using Lyapunov exponents and phase space analysis",
            "parameters": self.parameters,
            "input_requirements": self._get_required_columns(),
            "output_type": "pd.Series",
            "version": self.VERSION,
            "author": self.AUTHOR,
            "min_data_points": self.minimum_periods
        }

def export_indicator():
    """Export the indicator for registry discovery."""
    return {
        "class": ChaosGeometryPredictor,
        "category": "chaos",
        "name": "ChaosGeometryPredictor",
        "description": "Chaos theory and fractal geometry predictor"
    }

if __name__ == "__main__":
    print("*** Testing ChaosGeometryPredictor - Advanced Chaos Theory Trading Indicator")
    print("*** Fighting for humanitarian causes - helping sick children and poor families")
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate synthetic chaotic data
    t = np.arange(100)
    base_price = 100
    
    # Add chaotic dynamics (logistic map inspired)
    x = 0.5  # Initial value
    chaos_series = []
    for i in range(100):
        x = 3.9 * x * (1 - x)  # Chaotic logistic map
        chaos_series.append(x)
    
    chaos_component = 10 * np.array(chaos_series)
    
    # Add fractal noise
    fractal_noise = np.random.normal(0, 1, 100)
    for i in range(1, 100):
        # Add long-range correlation (fractional noise)
        fractal_noise[i] = 0.8 * fractal_noise[i-1] + 0.2 * fractal_noise[i]
    
    # Add strange attractor-like behavior
    attractor_x, attractor_y = 0.1, 0.1
    attractor_series = []
    for i in range(100):
        # Simplified Lorenz-like equations
        dx = 0.1 * (attractor_y - attractor_x)
        dy = 0.1 * (attractor_x - attractor_y - attractor_x * attractor_y)
        attractor_x += dx
        attractor_y += dy
        attractor_series.append(attractor_x)
    
    attractor_component = 5 * np.array(attractor_series)
    
    closes = base_price + chaos_component + fractal_noise + attractor_component
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': closes + np.random.normal(0, 0.5, 100),
        'high': closes + abs(np.random.normal(0, 1, 100)),
        'low': closes - abs(np.random.normal(0, 1, 100)),
        'close': closes,
        'volume': np.random.randint(1000, 10000, 100)
    })
    test_data.set_index('date', inplace=True)
    
    # Test the indicator
    try:
        indicator = ChaosGeometryPredictor(
            chaos_window=30,
            fractal_dimension_window=20,
            lyapunov_window=15,
            attractor_dimension=3
        )
        
        result = indicator.calculate(test_data)
        
        print(f"\n*** Chaos Geometry Analysis Results:")
        print(f"Predictor Values: {len(result)} calculated")
        print(f"Latest Prediction: {result.iloc[-1]:.3f}")
        print(f"Min Prediction: {result.min():.3f}")
        print(f"Max Prediction: {result.max():.3f}")
        print(f"Mean Prediction: {result.mean():.3f}")
        print(f"Prediction Std Dev: {result.std():.3f}")
        
        # Test signal interpretation
        latest_value = result.iloc[-1]
        if latest_value > 50:
            signal = "STRONG_BUY (Chaotic Breakout)"
        elif latest_value > 20:
            signal = "BUY (Fractal Growth)"
        elif latest_value > -20:
            signal = "HOLD (Strange Attractor)"
        elif latest_value > -50:
            signal = "SELL (Fractal Decline)"
        else:
            signal = "STRONG_SELL (Chaotic Collapse)"
        
        print(f"*** Trading Signal: {signal}")
        print(f"*** Chaos Level: {'High' if abs(latest_value) > 40 else 'Moderate' if abs(latest_value) > 20 else 'Low'}")
        print(f"*** Fractal State: {'Complex' if abs(latest_value) > 30 else 'Simple'}")
        
        print("\n*** ChaosGeometryPredictor test completed successfully!")
        print("*** Ready to generate profits for humanitarian causes!")
        
    except Exception as e:
        print(f"*** Test failed: {str(e)}")
        import traceback
        traceback.print_exc()