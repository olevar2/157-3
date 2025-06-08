from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

class IndicatorPerformance:
    """Track performance metrics for adaptive indicators"""
    def __init__(self):
        self.accuracy_scores: Dict[str, float] = {}
        self.computation_times: Dict[str, List[float]] = {}
        self.usage_counts: Dict[str, int] = {}
        self.error_rates: Dict[str, float] = {}
        
    def update_performance(self, indicator_name: str, accuracy: float, compute_time: float):
        """Update performance metrics for an indicator"""
        if indicator_name not in self.accuracy_scores:
            self.accuracy_scores[indicator_name] = accuracy
            self.computation_times[indicator_name] = []
            self.usage_counts[indicator_name] = 0
            self.error_rates[indicator_name] = 0.0
        else:
            # Exponential moving average for accuracy
            self.accuracy_scores[indicator_name] = 0.7 * self.accuracy_scores[indicator_name] + 0.3 * accuracy
        
        self.computation_times[indicator_name].append(compute_time)
        self.usage_counts[indicator_name] += 1
        
        # Keep only last 100 computation times
        if len(self.computation_times[indicator_name]) > 100:
            self.computation_times[indicator_name] = self.computation_times[indicator_name][-100:]

class AdaptiveIndicators:
    """
    Adaptive indicator system that learns and optimizes indicator selection
    based on market conditions and performance
    """
    
    def __init__(self):
        self.performance_tracker = IndicatorPerformance()
        self.market_regime_cache: Dict[str, Tuple[str, datetime]] = {}
        self.indicator_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl_seconds = 60  # 1 minute cache
        
        # Indicator registry with metadata
        self.indicator_registry = {
            'rsi': {'type': 'momentum', 'complexity': 'low', 'reliability': 0.7},
            'macd': {'type': 'trend', 'complexity': 'medium', 'reliability': 0.75},
            'bollinger_bands': {'type': 'volatility', 'complexity': 'medium', 'reliability': 0.8},
            'atr': {'type': 'volatility', 'complexity': 'low', 'reliability': 0.85},
            'stochastic': {'type': 'momentum', 'complexity': 'low', 'reliability': 0.7},
            'ichimoku': {'type': 'trend', 'complexity': 'high', 'reliability': 0.8},
            'volume_profile': {'type': 'volume', 'complexity': 'high', 'reliability': 0.75},
            'correlation_matrix': {'type': 'correlation', 'complexity': 'high', 'reliability': 0.9},
        }
        
        # Market regime to indicator mapping
        self.regime_indicator_map = {
            'trending_up': ['macd', 'ichimoku', 'rsi'],
            'trending_down': ['macd', 'ichimoku', 'rsi'],
            'volatile': ['atr', 'bollinger_bands', 'stochastic'],
            'ranging': ['bollinger_bands', 'stochastic', 'volume_profile'],
            'unknown': ['rsi', 'macd', 'atr']
        }
    
    async def select_optimal_indicators(self, 
                                      market_data: Dict[str, Any],
                                      agent_requirements: Dict[str, Any],
                                      max_indicators: int = 5) -> List[str]:
        """
        Select optimal indicators based on market conditions, agent requirements,
        and historical performance
        """
        # Detect market regime
        market_regime = await self._detect_market_regime(market_data)
        
        # Get base indicators for regime
        candidate_indicators = self.regime_indicator_map.get(market_regime, [])
        
        # Add agent-specific requirements
        if 'required_types' in agent_requirements:
            for ind_name, ind_info in self.indicator_registry.items():
                if ind_info['type'] in agent_requirements['required_types']:
                    candidate_indicators.append(ind_name)
        
        # Remove duplicates
        candidate_indicators = list(set(candidate_indicators))
        
        # Score and rank indicators
        scored_indicators = []
        for indicator in candidate_indicators:
            score = self._calculate_indicator_score(indicator, market_regime, agent_requirements)
            scored_indicators.append((indicator, score))
        
        # Sort by score and return top N
        scored_indicators.sort(key=lambda x: x[1], reverse=True)
        return [ind[0] for ind in scored_indicators[:max_indicators]]
    
    def _calculate_indicator_score(self, 
                                 indicator_name: str,
                                 market_regime: str,
                                 agent_requirements: Dict[str, Any]) -> float:
        """Calculate composite score for indicator selection"""
        score = 0.0
        
        # Base reliability score
        ind_info = self.indicator_registry.get(indicator_name, {})
        score += ind_info.get('reliability', 0.5) * 0.3
        
        # Performance history score
        if indicator_name in self.performance_tracker.accuracy_scores:
            score += self.performance_tracker.accuracy_scores[indicator_name] * 0.4
        
        # Regime appropriateness score
        if indicator_name in self.regime_indicator_map.get(market_regime, []):
            score += 0.2
        
        # Agent preference score
        if ind_info.get('type') in agent_requirements.get('preferred_types', []):
            score += 0.1
        
        return score
    
    async def _detect_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect current market regime with caching"""
        # Check cache first
        cache_key = str(hash(str(market_data.get('close', [])[-20:])))
        if cache_key in self.market_regime_cache:
            regime, timestamp = self.market_regime_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl_seconds:
                return regime
        
        # Calculate regime
        try:
            if 'close' not in market_data or len(market_data['close']) < 20:
                return 'unknown'
            
            prices = np.array(market_data['close'])
            
            # Calculate indicators for regime detection
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            trend = np.polyfit(range(len(prices)), prices, 1)[0]
            
            # Classify regime
            if volatility > np.percentile(returns, 80):
                regime = 'volatile'
            elif abs(trend) > np.percentile(np.abs(returns), 70):
                regime = 'trending_up' if trend > 0 else 'trending_down'
            else:
                regime = 'ranging'
            
            # Cache result
            self.market_regime_cache[cache_key] = (regime, datetime.now())
            return regime
            
        except Exception as e:
            return 'unknown'
    
    def update_indicator_performance(self, 
                                   indicator_name: str,
                                   prediction_accuracy: float,
                                   computation_time: float):
        """Update performance metrics for adaptive learning"""
        self.performance_tracker.update_performance(
            indicator_name,
            prediction_accuracy,
            computation_time
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all indicators"""
        report = {
            'accuracy_scores': self.performance_tracker.accuracy_scores,
            'average_computation_times': {},
            'usage_counts': self.performance_tracker.usage_counts,
            'error_rates': self.performance_tracker.error_rates
        }
        
        # Calculate average computation times
        for ind_name, times in self.performance_tracker.computation_times.items():
            if times:
                report['average_computation_times'][ind_name] = np.mean(times)
        
        return report
    
    async def calculate_adaptive_parameters(self,
                                          indicator_name: str,
                                          market_data: Dict[str, Any],
                                          base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate adaptive parameters for indicators based on market conditions
        """
        market_regime = await self._detect_market_regime(market_data)
        adapted_params = base_params.copy()
        
        # Adaptive rules based on indicator and regime
        if indicator_name == 'rsi' and market_regime == 'volatile':
            adapted_params['period'] = min(base_params.get('period', 14) + 7, 21)
            adapted_params['overbought'] = 75
            adapted_params['oversold'] = 25
            
        elif indicator_name == 'bollinger_bands':
            if market_regime == 'volatile':
                adapted_params['std_dev'] = 2.5
            elif market_regime in ['trending_up', 'trending_down']:
                adapted_params['std_dev'] = 2.0
                adapted_params['period'] = 20
        
        elif indicator_name == 'macd':
            if market_regime in ['trending_up', 'trending_down']:
                # Faster MACD for trending markets
                adapted_params['fast_period'] = 8
                adapted_params['slow_period'] = 17
                adapted_params['signal_period'] = 7
        
        return adapted_params

# Create singleton instance
adaptive_indicators = AdaptiveIndicators()
