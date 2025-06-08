"""
Indicator Coordinator for AI Agents
Manages and coordinates all 115+ indicators for AI decision making
"""

from typing import Dict, List, Any, Optional
import numpy as np
from engines.indicator_registry import indicator_registry

class IndicatorCoordinator:
    """Coordinates indicator usage for AI agents"""
    
    def __init__(self):
        self.registry = indicator_registry
        self.active_indicators = {}
        self.indicator_results = {}
        
    def analyze_with_all_indicators(self, market_data: Dict) -> Dict:
        """Run all relevant indicators and compile results"""
        results = {
            'momentum': self._analyze_momentum(market_data),
            'trend': self._analyze_trend(market_data),
            'volume': self._analyze_volume(market_data),
            'volatility': self._analyze_volatility(market_data),
            'patterns': self._analyze_patterns(market_data),
            'composite_score': 0.0,
            'signals': [],
            'timestamp': market_data.get('timestamp')
        }
        
        # Calculate composite score
        results['composite_score'] = self._calculate_composite_score(results)
        
        # Generate trading signals
        results['signals'] = self._generate_signals(results)
        
        return results
    
    def _analyze_momentum(self, market_data: Dict) -> Dict:
        """Analyze momentum using relevant indicators"""
        momentum_indicators = [
            'rsi', 'macd', 'stochastic', 'cci', 'tsi', 'williams_r'
        ]
        
        results = {}
        for indicator_name in momentum_indicators:
            try:
                indicator = self.registry.create_indicator(indicator_name)
                result = indicator.calculate(
                    market_data['high'],
                    market_data['low'],
                    market_data['close'],
                    market_data.get('volume')
                )
                results[indicator_name] = {
                    'value': result.value,
                    'signal': result.signal,
                    'strength': result.strength
                }
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")
                
        return results
    
    def _analyze_trend(self, market_data: Dict) -> Dict:
        """Analyze trend using relevant indicators"""
        trend_indicators = [
            'ema', 'sma', 'adx', 'aroon', 'supertrend', 'ichimoku'
        ]
        
        results = {}
        for indicator_name in trend_indicators:
            try:
                indicator = self.registry.create_indicator(indicator_name)
                result = indicator.calculate(
                    market_data['high'],
                    market_data['low'],
                    market_data['close'],
                    market_data.get('volume')
                )
                results[indicator_name] = {
                    'value': result.value,
                    'signal': result.signal,
                    'direction': result.metadata.get('direction', 'neutral')
                }
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")
                
        return results
    
    def _analyze_volume(self, market_data: Dict) -> Dict:
        """Analyze volume using relevant indicators"""
        if not market_data.get('volume'):
            return {}
            
        volume_indicators = [
            'obv', 'mfi', 'vwap', 'chaikin_money_flow', 'volume_profile'
        ]
        
        results = {}
        for indicator_name in volume_indicators:
            try:
                indicator = self.registry.create_indicator(indicator_name)
                result = indicator.calculate(
                    market_data['high'],
                    market_data['low'],
                    market_data['close'],
                    market_data['volume']
                )
                results[indicator_name] = {
                    'value': result.value,
                    'signal': result.signal,
                    'flow': result.metadata.get('flow_direction', 'neutral')
                }
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")
                
        return results
    
    def _analyze_volatility(self, market_data: Dict) -> Dict:
        """Analyze volatility using relevant indicators"""
        volatility_indicators = [
            'bollinger_bands', 'atr', 'keltner_channel', 'standard_deviation'
        ]
        
        results = {}
        for indicator_name in volatility_indicators:
            try:
                indicator = self.registry.create_indicator(indicator_name)
                result = indicator.calculate(
                    market_data['high'],
                    market_data['low'],
                    market_data['close'],
                    market_data.get('volume')
                )
                results[indicator_name] = {
                    'value': result.value,
                    'bands': result.metadata.get('bands', {}),
                    'volatility_level': result.metadata.get('volatility_level', 'normal')
                }
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")
                
        return results
    
    def _analyze_patterns(self, market_data: Dict) -> Dict:
        """Analyze chart patterns using pattern recognition indicators"""
        # Pattern analysis implementation
        return {}
    
    def _calculate_composite_score(self, results: Dict) -> float:
        """Calculate overall market score from all indicators"""
        scores = []
        
        # Momentum score
        momentum_signals = results.get('momentum', {})
        if momentum_signals:
            bullish = sum(1 for ind in momentum_signals.values() if ind.get('signal') == 'BULLISH')
            bearish = sum(1 for ind in momentum_signals.values() if ind.get('signal') == 'BEARISH')
            momentum_score = (bullish - bearish) / len(momentum_signals)
            scores.append(momentum_score)
        
        # Trend score
        trend_signals = results.get('trend', {})
        if trend_signals:
            uptrend = sum(1 for ind in trend_signals.values() if ind.get('direction') == 'up')
            downtrend = sum(1 for ind in trend_signals.values() if ind.get('direction') == 'down')
            trend_score = (uptrend - downtrend) / len(trend_signals)
            scores.append(trend_score)
        
        # Volume score
        volume_signals = results.get('volume', {})
        if volume_signals:
            accumulation = sum(1 for ind in volume_signals.values() if ind.get('flow') == 'accumulation')
            distribution = sum(1 for ind in volume_signals.values() if ind.get('flow') == 'distribution')
            volume_score = (accumulation - distribution) / len(volume_signals)
            scores.append(volume_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_signals(self, results: Dict) -> List[Dict]:
        """Generate trading signals based on indicator analysis"""
        signals = []
        composite_score = results.get('composite_score', 0)
        
        if composite_score > 0.5:
            signals.append({
                'action': 'BUY',
                'strength': composite_score,
                'confidence': self._calculate_confidence(results),
                'indicators_agree': self._count_agreeing_indicators(results, 'bullish')
            })
        elif composite_score < -0.5:
            signals.append({
                'action': 'SELL',
                'strength': abs(composite_score),
                'confidence': self._calculate_confidence(results),
                'indicators_agree': self._count_agreeing_indicators(results, 'bearish')
            })
        else:
            signals.append({
                'action': 'HOLD',
                'strength': 0.0,
                'confidence': 0.5,
                'reason': 'Mixed signals from indicators'
            })
            
        return signals
    
    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate confidence based on indicator agreement"""
        # Implementation for confidence calculation
        return 0.75
    
    def _count_agreeing_indicators(self, results: Dict, direction: str) -> int:
        """Count how many indicators agree on direction"""
        count = 0
        for category_results in results.values():
            if isinstance(category_results, dict):
                for indicator_result in category_results.values():
                    if isinstance(indicator_result, dict):
                        signal = indicator_result.get('signal', '').lower()
                        if direction in signal:
                            count += 1
        return count

# Usage example for AI agents
indicator_coordinator = IndicatorCoordinator()
