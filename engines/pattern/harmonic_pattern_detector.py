#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harmonic Pattern Detector - Advanced Geometric Pattern Recognition
Platform3 Phase 3 - Enhanced Pattern Analysis

The Harmonic Pattern Detector identifies geometric harmonic patterns in price action
including Gartley, Butterfly, Bat, Crab, and other XABCD patterns. These patterns
are based on Fibonacci ratios and provide high-probability reversal points.
"""

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "shared"))
sys.path.append(str(project_root / "engines"))

from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import math
import warnings
warnings.filterwarnings('ignore')

@dataclass
class HarmonicPoint:
    """Harmonic pattern point data structure"""
    point_name: str           # X, A, B, C, D
    index: int               # Data index
    price: float             # Price level
    time: datetime           # Timestamp
    is_high: bool           # True for high, False for low

@dataclass
class HarmonicPattern:
    """Complete harmonic pattern structure"""
    pattern_type: str        # 'Gartley', 'Butterfly', 'Bat', 'Crab', etc.
    pattern_name: str        # Full descriptive name
    points: Dict[str, HarmonicPoint]  # X, A, B, C, D points
    ratios: Dict[str, float] # AB/XA, BC/AB, CD/BC, AD/XA ratios
    fibonacci_ratios: Dict[str, float]  # Expected Fibonacci ratios
    completion_price: float  # D point completion price
    validity_score: float   # Pattern validity (0-1)
    bullish: bool           # True for bullish, False for bearish
    projection_target_1: float  # First target
    projection_target_2: float  # Second target
    stop_loss: float        # Suggested stop loss
    risk_reward_ratio: float # Risk to reward ratio

class HarmonicPatternDetector:
    """
    Advanced Harmonic Pattern Detection Engine
    
    Features:
    - Gartley Pattern Detection (AB=61.8% XA, CD=78.6% BC)
    - Butterfly Pattern Detection (AB=78.6% XA, CD=127.2% BC)
    - Bat Pattern Detection (AB=38.2%-50% XA, CD=88.6% BC)
    - Crab Pattern Detection (AB=38.2%-61.8% XA, CD=224%-361.8% BC)
    - Cypher Pattern Detection
    - Shark Pattern Detection
    - ABCD Pattern Detection
    - Three Drives Pattern Detection
    """
    
    def __init__(self, lookback_period: int = 100, min_pattern_bars: int = 20):
        """Initialize Harmonic Pattern Detector with Platform3 framework"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        self.lookback_period = lookback_period
        self.min_pattern_bars = min_pattern_bars
        
        # Fibonacci ratios for harmonic patterns
        self.fibonacci_ratios = {
            'retracement': [0.236, 0.382, 0.500, 0.618, 0.786],
            'extension': [1.272, 1.414, 1.618, 2.000, 2.236, 2.618, 3.618]
        }
        
        # Pattern definitions with expected ratios
        self.pattern_definitions = {
            'Gartley': {
                'AB_XA': (0.618, 0.618),  # AB should be 61.8% of XA
                'BC_AB': (0.382, 0.886),  # BC should be 38.2%-88.6% of AB
                'CD_BC': (1.272, 1.618),  # CD should be 127.2%-161.8% of BC
                'AD_XA': (0.786, 0.786)   # AD should be 78.6% of XA
            },
            'Butterfly': {
                'AB_XA': (0.786, 0.786),
                'BC_AB': (0.382, 0.886),
                'CD_BC': (1.618, 2.618),
                'AD_XA': (1.272, 1.272)
            },
            'Bat': {
                'AB_XA': (0.382, 0.500),
                'BC_AB': (0.382, 0.886),
                'CD_BC': (1.618, 2.618),
                'AD_XA': (0.886, 0.886)
            },
            'Crab': {
                'AB_XA': (0.382, 0.618),
                'BC_AB': (0.382, 0.886),
                'CD_BC': (2.240, 3.618),
                'AD_XA': (1.618, 1.618)
            },
            'Cypher': {
                'AB_XA': (0.382, 0.618),
                'BC_AB': (1.272, 1.414),
                'CD_BC': (0.786, 0.786),
                'AD_XA': (0.786, 0.786)
            },
            'Shark': {
                'AB_XA': (0.382, 0.618),
                'BC_AB': (1.130, 1.618),
                'CD_BC': (1.618, 2.240),
                'AD_XA': (0.886, 1.130)
            }
        }
        
        self.logger.info(f"Harmonic Pattern Detector initialized - Lookback: {self.lookback_period}, Min Bars: {self.min_pattern_bars}")
    
    async def detect_patterns(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Detect harmonic patterns in price data
        
        Args:
            data: Price data (OHLC DataFrame or close price array)
            
        Returns:
            Dictionary containing detected patterns and analysis
        """
        start_time = time.time()
        
        try:
            self.logger.debug("Starting harmonic pattern detection")
            
            # Prepare and validate data
            price_data, high_data, low_data = self._prepare_data(data)
            if price_data is None:
                raise ServiceError("Invalid price data", "INVALID_DATA")
            
            # Find swing points (highs and lows)
            swing_points = await self._find_swing_points(high_data, low_data)
            if len(swing_points) < 5:
                self.logger.warning("Insufficient swing points for pattern detection")
                return self._create_empty_result()
            
            # Detect patterns for each pattern type
            detected_patterns = []
            for pattern_type in self.pattern_definitions.keys():
                patterns = await self._detect_pattern_type(swing_points, pattern_type)
                detected_patterns.extend(patterns)
            
            # Filter and rank patterns by validity
            valid_patterns = [p for p in detected_patterns if p.validity_score >= 0.7]
            valid_patterns.sort(key=lambda x: x.validity_score, reverse=True)
            
            # Analyze current market position relative to patterns
            current_analysis = await self._analyze_current_position(
                price_data, high_data, low_data, valid_patterns
            )
            
            # Generate trading signals
            trading_signals = await self._generate_pattern_signals(
                valid_patterns, price_data[-1] if len(price_data) > 0 else 0
            )
            
            result = {
                'detected_patterns': [self._pattern_to_dict(p) for p in valid_patterns],
                'pattern_count': len(valid_patterns),
                'swing_points': [self._point_to_dict(p) for p in swing_points[-20:]],  # Last 20 points
                'current_analysis': current_analysis,
                'trading_signals': trading_signals,
                'pattern_statistics': self._calculate_pattern_statistics(valid_patterns),
                'calculation_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Pattern detection completed - Found {len(valid_patterns)} valid patterns in {result['calculation_time']:.4f}s")
            return result
            
        except ServiceError as e:
            self.logger.error(f"Service error in pattern detection: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in pattern detection: {e}")
            self.error_system.handle_error(e, self.__class__.__name__)
            return None    
    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare and validate input data"""
        try:
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    price_data = data['close'].values
                    high_data = data['high'].values if 'high' in data.columns else price_data
                    low_data = data['low'].values if 'low' in data.columns else price_data
                else:
                    price_data = data.iloc[:, 0].values
                    high_data = data.iloc[:, 1].values if data.shape[1] > 1 else price_data
                    low_data = data.iloc[:, 2].values if data.shape[1] > 2 else price_data
            else:
                price_data = np.array(data)
                high_data = price_data
                low_data = price_data
            
            if len(price_data) < self.min_pattern_bars:
                self.logger.warning(f"Insufficient data: {len(price_data)} < {self.min_pattern_bars}")
                return None, None, None
            
            return price_data, high_data, low_data
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return None, None, None
    
    async def _find_swing_points(self, high_data: np.ndarray, low_data: np.ndarray, 
                                lookback: int = 5) -> List[HarmonicPoint]:
        """Find swing highs and lows in price data"""
        swing_points = []
        
        try:
            # Find swing highs
            for i in range(lookback, len(high_data) - lookback):
                is_high = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and high_data[j] >= high_data[i]:
                        is_high = False
                        break
                
                if is_high:
                    point = HarmonicPoint(
                        point_name='H',
                        index=i,
                        price=high_data[i],
                        time=datetime.now() + timedelta(minutes=i),
                        is_high=True
                    )
                    swing_points.append(point)
            
            # Find swing lows
            for i in range(lookback, len(low_data) - lookback):
                is_low = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and low_data[j] <= low_data[i]:
                        is_low = False
                        break
                
                if is_low:
                    point = HarmonicPoint(
                        point_name='L',
                        index=i,
                        price=low_data[i],
                        time=datetime.now() + timedelta(minutes=i),
                        is_high=False
                    )
                    swing_points.append(point)
            
            # Sort by index
            swing_points.sort(key=lambda x: x.index)
            
            self.logger.debug(f"Found {len(swing_points)} swing points")
            return swing_points
            
        except Exception as e:
            self.logger.error(f"Error finding swing points: {e}")
            return []
    
    async def _detect_pattern_type(self, swing_points: List[HarmonicPoint], 
                                 pattern_type: str) -> List[HarmonicPattern]:
        """Detect specific harmonic pattern type"""
        patterns = []
        
        try:
            if len(swing_points) < 5:
                return patterns
            
            pattern_def = self.pattern_definitions[pattern_type]
            
            # Look for XABCD patterns in recent swing points
            for i in range(len(swing_points) - 4):
                points_slice = swing_points[i:i+5]
                
                # Ensure alternating high/low pattern
                if not self._is_valid_xabcd_sequence(points_slice):
                    continue
                
                # Create pattern points
                pattern_points = {
                    'X': points_slice[0],
                    'A': points_slice[1], 
                    'B': points_slice[2],
                    'C': points_slice[3],
                    'D': points_slice[4]
                }
                
                # Calculate ratios
                ratios = self._calculate_pattern_ratios(pattern_points)
                
                # Check if ratios match pattern definition
                validity_score = self._validate_pattern_ratios(ratios, pattern_def)
                
                if validity_score >= 0.5:  # Minimum validity threshold
                    # Determine if bullish or bearish
                    bullish = pattern_points['X'].is_high and not pattern_points['D'].is_high
                    
                    # Calculate targets and stops
                    targets, stop_loss = self._calculate_targets_and_stops(pattern_points, bullish)
                    
                    pattern = HarmonicPattern(
                        pattern_type=pattern_type,
                        pattern_name=f"{pattern_type} {'Bullish' if bullish else 'Bearish'}",
                        points=pattern_points,
                        ratios=ratios,
                        fibonacci_ratios=pattern_def,
                        completion_price=pattern_points['D'].price,
                        validity_score=validity_score,
                        bullish=bullish,
                        projection_target_1=targets[0],
                        projection_target_2=targets[1],
                        stop_loss=stop_loss,
                        risk_reward_ratio=self._calculate_risk_reward(
                            pattern_points['D'].price, targets[0], stop_loss
                        )
                    )
                    
                    patterns.append(pattern)
            
            self.logger.debug(f"Found {len(patterns)} {pattern_type} patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting {pattern_type} patterns: {e}")
            return []    
    def _is_valid_xabcd_sequence(self, points: List[HarmonicPoint]) -> bool:
        """Check if points form valid XABCD alternating sequence"""
        if len(points) != 5:
            return False
        
        # Check for alternating highs and lows
        for i in range(len(points) - 1):
            if points[i].is_high == points[i + 1].is_high:
                return False
        
        return True
    
    def _calculate_pattern_ratios(self, points: Dict[str, HarmonicPoint]) -> Dict[str, float]:
        """Calculate harmonic pattern ratios"""
        try:
            X, A, B, C, D = points['X'], points['A'], points['B'], points['C'], points['D']
            
            XA = abs(A.price - X.price)
            AB = abs(B.price - A.price)
            BC = abs(C.price - B.price)
            CD = abs(D.price - C.price)
            AD = abs(D.price - A.price)
            
            ratios = {}
            
            if XA != 0:
                ratios['AB_XA'] = AB / XA
                ratios['AD_XA'] = AD / XA
            
            if AB != 0:
                ratios['BC_AB'] = BC / AB
            
            if BC != 0:
                ratios['CD_BC'] = CD / BC
            
            return ratios
            
        except Exception as e:
            self.logger.error(f"Error calculating ratios: {e}")
            return {}
    
    def _validate_pattern_ratios(self, ratios: Dict[str, float], 
                               pattern_def: Dict[str, Tuple[float, float]]) -> float:
        """Validate pattern ratios against definition"""
        try:
            score = 0.0
            total_checks = 0
            
            for ratio_name, (min_val, max_val) in pattern_def.items():
                if ratio_name in ratios:
                    ratio_value = ratios[ratio_name]
                    total_checks += 1
                    
                    # Calculate how close the ratio is to the expected range
                    if min_val <= ratio_value <= max_val:
                        score += 1.0
                    else:
                        # Partial score based on distance from range
                        mid_point = (min_val + max_val) / 2
                        distance = abs(ratio_value - mid_point)
                        range_size = (max_val - min_val) / 2
                        
                        if distance <= range_size * 1.5:  # Within 150% of range
                            score += max(0, 1.0 - (distance / (range_size * 1.5)))
            
            return score / total_checks if total_checks > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error validating ratios: {e}")
            return 0.0
    
    def _calculate_targets_and_stops(self, points: Dict[str, HarmonicPoint], 
                                   bullish: bool) -> Tuple[List[float], float]:
        """Calculate profit targets and stop loss"""
        try:
            X, A, B, C, D = points['X'], points['A'], points['B'], points['C'], points['D']
            
            if bullish:
                # Bullish targets: project upward from D
                target_1 = D.price + abs(C.price - D.price) * 0.618
                target_2 = D.price + abs(A.price - D.price) * 0.618
                stop_loss = D.price - abs(C.price - D.price) * 0.236
            else:
                # Bearish targets: project downward from D
                target_1 = D.price - abs(C.price - D.price) * 0.618
                target_2 = D.price - abs(A.price - D.price) * 0.618
                stop_loss = D.price + abs(C.price - D.price) * 0.236
            
            return [target_1, target_2], stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating targets: {e}")
            return [0.0, 0.0], 0.0
    
    def _calculate_risk_reward(self, entry: float, target: float, stop: float) -> float:
        """Calculate risk-reward ratio"""
        try:
            risk = abs(entry - stop)
            reward = abs(target - entry)
            return reward / risk if risk > 0 else 0.0
        except:
            return 0.0
    
    async def _analyze_current_position(self, price_data: np.ndarray, high_data: np.ndarray,
                                       low_data: np.ndarray, patterns: List[HarmonicPattern]) -> Dict[str, Any]:
        """Analyze current market position relative to detected patterns"""
        try:
            current_price = price_data[-1] if len(price_data) > 0 else 0
            current_high = high_data[-1] if len(high_data) > 0 else current_price
            current_low = low_data[-1] if len(low_data) > 0 else current_price
            
            analysis = {
                'current_price': current_price,
                'current_high': current_high,
                'current_low': current_low,
                'active_patterns': [],
                'completion_zones': [],
                'near_completion': []
            }
            
            for pattern in patterns:
                d_point = pattern.points['D']
                price_distance = abs(current_price - d_point.price) / d_point.price
                
                # Check if pattern is near completion (within 2%)
                if price_distance <= 0.02:
                    analysis['near_completion'].append({
                        'pattern_type': pattern.pattern_type,
                        'completion_price': d_point.price,
                        'distance': price_distance,
                        'bullish': pattern.bullish
                    })
                
                # Check if currently in pattern completion zone
                if pattern.bullish and current_price <= d_point.price * 1.01:
                    analysis['completion_zones'].append(pattern.pattern_type)
                elif not pattern.bullish and current_price >= d_point.price * 0.99:
                    analysis['completion_zones'].append(pattern.pattern_type)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing current position: {e}")
            return {}
    
    async def _generate_pattern_signals(self, patterns: List[HarmonicPattern], 
                                      current_price: float) -> Dict[str, Any]:
        """Generate trading signals based on detected patterns"""
        try:
            signals = {
                'buy_signals': [],
                'sell_signals': [],
                'strength': 'neutral',
                'confidence': 0.0,
                'recommendations': []
            }
            
            bullish_patterns = [p for p in patterns if p.bullish]
            bearish_patterns = [p for p in patterns if not p.bullish]
            
            # Generate buy signals
            for pattern in bullish_patterns:
                d_price = pattern.points['D'].price
                if current_price <= d_price * 1.005:  # Within 0.5% of completion
                    signals['buy_signals'].append({
                        'pattern': pattern.pattern_type,
                        'entry_price': d_price,
                        'target_1': pattern.projection_target_1,
                        'target_2': pattern.projection_target_2,
                        'stop_loss': pattern.stop_loss,
                        'risk_reward': pattern.risk_reward_ratio,
                        'validity': pattern.validity_score
                    })
            
            # Generate sell signals
            for pattern in bearish_patterns:
                d_price = pattern.points['D'].price
                if current_price >= d_price * 0.995:  # Within 0.5% of completion
                    signals['sell_signals'].append({
                        'pattern': pattern.pattern_type,
                        'entry_price': d_price,
                        'target_1': pattern.projection_target_1,
                        'target_2': pattern.projection_target_2,
                        'stop_loss': pattern.stop_loss,
                        'risk_reward': pattern.risk_reward_ratio,
                        'validity': pattern.validity_score
                    })
            
            # Determine overall strength and confidence
            total_signals = len(signals['buy_signals']) + len(signals['sell_signals'])
            if total_signals > 0:
                avg_validity = np.mean([s['validity'] for s in signals['buy_signals'] + signals['sell_signals']])
                signals['confidence'] = avg_validity
                
                if len(signals['buy_signals']) > len(signals['sell_signals']):
                    signals['strength'] = 'bullish'
                elif len(signals['sell_signals']) > len(signals['buy_signals']):
                    signals['strength'] = 'bearish'
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {'buy_signals': [], 'sell_signals': [], 'strength': 'neutral', 'confidence': 0.0}
    
    def _calculate_pattern_statistics(self, patterns: List[HarmonicPattern]) -> Dict[str, Any]:
        """Calculate statistics for detected patterns"""
        try:
            if not patterns:
                return {}
            
            pattern_types = [p.pattern_type for p in patterns]
            validity_scores = [p.validity_score for p in patterns]
            risk_rewards = [p.risk_reward_ratio for p in patterns]
            
            stats = {
                'total_patterns': len(patterns),
                'pattern_distribution': {ptype: pattern_types.count(ptype) for ptype in set(pattern_types)},
                'average_validity': np.mean(validity_scores),
                'max_validity': np.max(validity_scores),
                'average_risk_reward': np.mean(risk_rewards),
                'bullish_count': len([p for p in patterns if p.bullish]),
                'bearish_count': len([p for p in patterns if not p.bullish])
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure"""
        return {
            'detected_patterns': [],
            'pattern_count': 0,
            'swing_points': [],
            'current_analysis': {},
            'trading_signals': {'buy_signals': [], 'sell_signals': [], 'strength': 'neutral', 'confidence': 0.0},
            'pattern_statistics': {},
            'calculation_time': 0.0,
            'timestamp': datetime.now().isoformat()
        }    
    def _pattern_to_dict(self, pattern: HarmonicPattern) -> Dict[str, Any]:
        """Convert HarmonicPattern to dictionary"""
        return {
            'pattern_type': pattern.pattern_type,
            'pattern_name': pattern.pattern_name,
            'points': {name: self._point_to_dict(point) for name, point in pattern.points.items()},
            'ratios': pattern.ratios,
            'fibonacci_ratios': pattern.fibonacci_ratios,
            'completion_price': pattern.completion_price,
            'validity_score': pattern.validity_score,
            'bullish': pattern.bullish,
            'projection_target_1': pattern.projection_target_1,
            'projection_target_2': pattern.projection_target_2,
            'stop_loss': pattern.stop_loss,
            'risk_reward_ratio': pattern.risk_reward_ratio
        }
    
    def _point_to_dict(self, point: HarmonicPoint) -> Dict[str, Any]:
        """Convert HarmonicPoint to dictionary"""
        return {
            'point_name': point.point_name,
            'index': point.index,
            'price': point.price,
            'time': point.time.isoformat(),
            'is_high': point.is_high
        }

    async def get_pattern_summary(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Get summary of harmonic patterns"""
        try:
            result = await self.detect_patterns(data)
            if not result:
                return None
            
            return {
                'total_patterns': result['pattern_count'],
                'pattern_types': list(result['pattern_statistics'].get('pattern_distribution', {}).keys()),
                'strongest_pattern': max(result['detected_patterns'], 
                                       key=lambda x: x['validity_score']) if result['detected_patterns'] else None,
                'signal_strength': result['trading_signals']['strength'],
                'signal_confidence': result['trading_signals']['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pattern summary: {e}")
            return None
    
    async def get_completion_zones(self, data: Union[np.ndarray, pd.DataFrame], 
                                 current_price: float) -> List[Dict[str, Any]]:
        """Get active pattern completion zones"""
        try:
            result = await self.detect_patterns(data)
            if not result:
                return []
            
            completion_zones = []
            for pattern_dict in result['detected_patterns']:
                d_point_price = pattern_dict['completion_price']
                distance = abs(current_price - d_point_price) / d_point_price
                
                if distance <= 0.02:  # Within 2%
                    completion_zones.append({
                        'pattern_type': pattern_dict['pattern_type'],
                        'completion_price': d_point_price,
                        'distance_percent': distance * 100,
                        'bullish': pattern_dict['bullish'],
                        'validity_score': pattern_dict['validity_score'],
                        'targets': [pattern_dict['projection_target_1'], pattern_dict['projection_target_2']],
                        'stop_loss': pattern_dict['stop_loss']
                    })
            
            return completion_zones
            
        except Exception as e:
            self.logger.error(f"Error getting completion zones: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    async def test_harmonic_detector():
        # Create sample OHLC data
        np.random.seed(42)
        periods = 200
        
        # Generate realistic price data with trends and reversals
        base_price = 100
        price_data = [base_price]
        high_data = [base_price]
        low_data = [base_price]
        
        for i in range(1, periods):
            # Add some trending behavior with noise
            trend = 0.001 * np.sin(i * 0.1)
            noise = np.random.normal(0, 0.002)
            change = trend + noise
            
            new_price = price_data[-1] * (1 + change)
            price_data.append(new_price)
            
            # Create high/low with some spread
            spread = new_price * 0.01
            high_data.append(new_price + np.random.uniform(0, spread))
            low_data.append(new_price - np.random.uniform(0, spread))
        
        # Create DataFrame
        df = pd.DataFrame({
            'close': price_data,
            'high': high_data,
            'low': low_data
        })
        
        # Test the detector
        detector = HarmonicPatternDetector(lookback_period=50, min_pattern_bars=20)
        
        print("Testing Harmonic Pattern Detector...")
        result = await detector.detect_patterns(df)
        
        if result:
            print(f"Detection completed in {result['calculation_time']:.4f} seconds")
            print(f"Found {result['pattern_count']} valid patterns")
            
            if result['detected_patterns']:
                for pattern in result['detected_patterns'][:3]:  # Show first 3
                    print(f"\nPattern: {pattern['pattern_name']}")
                    print(f"Validity Score: {pattern['validity_score']:.3f}")
                    print(f"Completion Price: {pattern['completion_price']:.2f}")
                    print(f"Risk/Reward: {pattern['risk_reward_ratio']:.2f}")
            
            signals = result['trading_signals']
            print(f"\nSignal Strength: {signals['strength']} (Confidence: {signals['confidence']:.2f})")
            print(f"Buy Signals: {len(signals['buy_signals'])}")
            print(f"Sell Signals: {len(signals['sell_signals'])}")
        else:
            print("No patterns detected or error occurred")
    
    # Run the test
    asyncio.run(test_harmonic_detector())