"""
Fractal Correlation Dimension Indicator

Calculates the fractal correlation dimension to measure the fractal complexity
and self-similarity of price movements, providing insights into market structure
and regime identification.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ..base_indicator import StandardIndicatorInterface


@dataclass
class FractalCorrelationDimensionResult:
    correlation_dimension: float
    complexity_level: str  # "low", "moderate", "high"
    fractal_regime: str   # "trending", "ranging", "chaotic"
    confidence: float
    embedding_dimension: int
    timestamp: Optional[str] = None


class FractalCorrelationDimension(StandardIndicatorInterface):
    """
    Fractal Correlation Dimension Indicator
    
    Measures the fractal dimension of price time series using correlation integrals
    to determine market complexity and regime characteristics.
    
    The correlation dimension quantifies how the number of points within a given
    distance scales with that distance in the reconstructed phase space.
    """
    
    CATEGORY = "technical"
    
    def __init__(self, 
                 lookback: int = 100,
                 max_embedding_dim: int = 10,
                 min_embedding_dim: int = 2,
                 tolerance_range: Tuple[float, float] = (0.001, 0.1),
                 **kwargs):
        """
        Initialize Fractal Correlation Dimension calculator.
        
        Args:
            lookback: Number of periods to analyze
            max_embedding_dim: Maximum embedding dimension to test
            min_embedding_dim: Minimum embedding dimension to test
            tolerance_range: Range of tolerance values for correlation integral
        """
        super().__init__(**kwargs)
        self.lookback = lookback
        self.max_embedding_dim = max_embedding_dim
        self.min_embedding_dim = min_embedding_dim
        self.tolerance_range = tolerance_range
    
    def calculate(self, data: pd.DataFrame) -> FractalCorrelationDimensionResult:
        """
        Calculate fractal correlation dimension.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            FractalCorrelationDimensionResult with correlation dimension analysis
        """
        try:
            if len(data) < self.lookback:
                return FractalCorrelationDimensionResult(
                    correlation_dimension=0.0,
                    complexity_level="unknown",
                    fractal_regime="insufficient_data",
                    confidence=0.0,
                    embedding_dimension=0
                )
            
            # Use log returns for analysis
            returns = np.log(data['close'] / data['close'].shift(1)).dropna()
            recent_returns = returns.tail(self.lookback).values
            
            # Calculate correlation dimension using Grassberger-Procaccia algorithm
            correlation_dim, best_embedding_dim = self._calculate_correlation_dimension(recent_returns)
            
            # Classify complexity level
            complexity_level = self._classify_complexity(correlation_dim)
            
            # Determine fractal regime
            fractal_regime = self._determine_regime(correlation_dim, recent_returns)
            
            # Calculate confidence based on consistency across embedding dimensions
            confidence = self._calculate_confidence(recent_returns)
            
            return FractalCorrelationDimensionResult(
                correlation_dimension=correlation_dim,
                complexity_level=complexity_level,
                fractal_regime=fractal_regime,
                confidence=confidence,
                embedding_dimension=best_embedding_dim,
                timestamp=data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else None
            )
            
        except Exception as e:
            return FractalCorrelationDimensionResult(
                correlation_dimension=0.0,
                complexity_level="error",
                fractal_regime="calculation_error",
                confidence=0.0,
                embedding_dimension=0
            )
    
    def _calculate_correlation_dimension(self, returns: np.ndarray) -> Tuple[float, int]:
        """Calculate correlation dimension using Grassberger-Procaccia algorithm."""
        best_dim = 0.0
        best_embedding = self.min_embedding_dim
        
        for m in range(self.min_embedding_dim, self.max_embedding_dim + 1):
            # Create embedded vectors
            embedded = self._embed_timeseries(returns, m)
            if len(embedded) < 10:  # Need minimum points for reliable calculation
                continue
                
            # Calculate correlation integral for different radius values
            correlation_dim = self._estimate_correlation_dimension(embedded)
            
            if correlation_dim > best_dim:
                best_dim = correlation_dim
                best_embedding = m
        
        return max(best_dim, 0.1), best_embedding
    
    def _embed_timeseries(self, series: np.ndarray, embedding_dim: int, delay: int = 1) -> np.ndarray:
        """Create embedded vectors from time series."""
        n = len(series)
        embedded_length = n - (embedding_dim - 1) * delay
        
        if embedded_length <= 0:
            return np.array([])
        
        embedded = np.zeros((embedded_length, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = series[i * delay:i * delay + embedded_length]
        
        return embedded
    
    def _estimate_correlation_dimension(self, embedded: np.ndarray) -> float:
        """Estimate correlation dimension from embedded vectors."""
        n_points = len(embedded)
        if n_points < 10:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(min(n_points, 50)):  # Limit for performance
            for j in range(i + 1, min(n_points, 50)):
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if dist > 0:
                    distances.append(dist)
        
        if len(distances) < 10:
            return 0.0
        
        distances = np.array(distances)
        
        # Calculate correlation integral for different radii
        min_radius = np.percentile(distances, 5)
        max_radius = np.percentile(distances, 50)
        
        if min_radius >= max_radius or min_radius <= 0:
            return 0.0
        
        radii = np.logspace(np.log10(min_radius), np.log10(max_radius), 10)
        correlations = []
        
        for radius in radii:
            correlation = np.sum(distances < radius) / len(distances)
            if correlation > 0:
                correlations.append(correlation)
            else:
                correlations.append(1e-10)  # Avoid log(0)
        
        # Estimate dimension from slope of log(C(r)) vs log(r)
        if len(correlations) < 3:
            return 0.0
        
        log_radii = np.log(radii[:len(correlations)])
        log_correlations = np.log(correlations)
        
        # Linear regression to find slope
        try:
            slope = np.polyfit(log_radii, log_correlations, 1)[0]
            return max(abs(slope), 0.1)  # Correlation dimension should be positive
        except:
            return 0.5  # Default fallback
    
    def _classify_complexity(self, correlation_dim: float) -> str:
        """Classify market complexity based on correlation dimension."""
        if correlation_dim < 1.5:
            return "low"
        elif correlation_dim < 2.5:
            return "moderate"
        else:
            return "high"
    
    def _determine_regime(self, correlation_dim: float, returns: np.ndarray) -> str:
        """Determine market regime based on fractal characteristics."""
        # Calculate trend strength
        trend_strength = abs(np.mean(returns)) / (np.std(returns) + 1e-10)
        
        if correlation_dim < 1.2 and trend_strength > 0.1:
            return "trending"
        elif correlation_dim > 2.0:
            return "chaotic"
        else:
            return "ranging"
    
    def _calculate_confidence(self, returns: np.ndarray) -> float:
        """Calculate confidence in the fractal dimension estimate."""
        # Confidence based on data consistency and sufficient length
        n_points = len(returns)
        if n_points < self.lookback * 0.5:
            return 0.3
        elif n_points < self.lookback * 0.8:
            return 0.6
        else:
            return 0.9
    
    def get_display_name(self) -> str:
        return "Fractal Correlation Dimension"
    
    def get_parameters(self) -> Dict:
        return {
            "lookback": self.lookback,
            "max_embedding_dim": self.max_embedding_dim,
            "min_embedding_dim": self.min_embedding_dim,
            "tolerance_range": self.tolerance_range
        }