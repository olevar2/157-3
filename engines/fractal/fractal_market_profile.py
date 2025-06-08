"""
Fractal Market Profile
======================

Advanced market profile analysis using fractal geometry to create
multi-dimensional price distribution profiles. Identifies value areas,
point of control, and market structure using fractal decomposition.

The indicator creates:
- Multi-timeframe market profiles
- Fractal value areas
- Volume-weighted price distributions
- Market structure identification

Author: Platform3 AI System
Created: December 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Fix import - use absolute import with fallback
try:
    from engines.indicator_base import IndicatorBase
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator_base import IndicatorBase


class FractalMarketProfile(IndicatorBase):
    """
    Fractal Market Profile for advanced price distribution analysis.
    
    Creates market profiles using:
    - Fractal price level decomposition
    - Multi-scale volume analysis
    - Dynamic value area calculation
    - Market structure recognition
    """
    
    def __init__(self,
                 profile_period: int = 20,
                 price_levels: int = 50,
                 value_area_pct: float = 0.70,
                 fractal_scales: List[int] = None,
                 min_level_threshold: float = 0.01):
        """
        Initialize Fractal Market Profile.
        
        Args:
            profile_period: Period for profile calculation
            price_levels: Number of price levels in profile
            value_area_pct: Percentage for value area (0.70 = 70%)
            fractal_scales: Scales for fractal analysis [1, 5, 20]
            min_level_threshold: Minimum volume threshold for level
        """
        super().__init__()
        
        self.profile_period = profile_period
        self.price_levels = price_levels
        self.value_area_pct = value_area_pct
        self.fractal_scales = fractal_scales or [1, 5, 20]
        self.min_level_threshold = min_level_threshold
        
        # Profile calculation parameters
        self.smooth_factor = 3
        
        # Validation
        if profile_period < 5:
            raise ValueError("Profile period must be at least 5")
        if not 0 < value_area_pct < 1:
            raise ValueError("Value area percentage must be between 0 and 1")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate Fractal Market Profile analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing market profile analysis
        """
        try:
            # Validate input data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            self._validate_data(data, required_columns)
            
            if len(data) < self.profile_period:
                raise ValueError(f"Insufficient data: need at least {self.profile_period} periods")
            
            # Extract data
            opens = data['open'].values
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values
            
            # Build multi-scale profiles
            profiles = self._build_fractal_profiles(opens, highs, lows, closes, volumes)
            
            # Calculate value areas for each profile
            value_areas = self._calculate_value_areas(profiles)
            
            # Identify market structure
            market_structure = self._identify_market_structure(profiles, value_areas)
            
            # Calculate profile statistics
            profile_stats = self._calculate_profile_statistics(profiles)
            
            # Generate profile-based signals
            signals = self._generate_profile_signals(profiles, value_areas, closes)
            
            # Calculate fractal dimensions of profiles
            fractal_metrics = self._calculate_fractal_metrics(profiles)
            
            # Identify key levels
            key_levels = self._identify_key_levels(profiles, value_areas)
            
            return {
                'profiles': self._serialize_profiles(profiles),
                'value_areas': value_areas,
                'market_structure': market_structure,
                'profile_statistics': profile_stats,
                'signals': signals,
                'fractal_metrics': fractal_metrics,
                'key_levels': key_levels,
                'interpretation': self._interpret_profile_state(
                    market_structure,
                    profile_stats,
                    signals[-1] if len(signals) > 0 else 0
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Fractal Market Profile: {e}")
            raise
    
    def _build_fractal_profiles(self, opens: np.ndarray, highs: np.ndarray,
                               lows: np.ndarray, closes: np.ndarray,
                               volumes: np.ndarray) -> Dict[int, Dict]:
        """Build market profiles at multiple fractal scales."""
        profiles = {}
        
        for scale in self.fractal_scales:
            # Create profile for this scale
            profile = self._create_profile(
                opens, highs, lows, closes, volumes, scale
            )
            profiles[scale] = profile
        
        return profiles
    
    def _create_profile(self, opens: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, closes: np.ndarray,
                       volumes: np.ndarray, scale: int) -> Dict:
        """Create a single market profile."""
        profile = {
            'scale': scale,
            'price_levels': {},
            'total_volume': 0,
            'sessions': []
        }
        
        # Process data in windows based on scale
        for i in range(scale * self.profile_period - 1, len(closes), scale):
            start_idx = max(0, i - scale * self.profile_period + 1)
            
            # Get window data
            window_highs = highs[start_idx:i + 1]
            window_lows = lows[start_idx:i + 1]
            window_closes = closes[start_idx:i + 1]
            window_volumes = volumes[start_idx:i + 1]
            
            if len(window_highs) == 0:
                continue
            
            # Calculate price range
            range_high = np.max(window_highs)
            range_low = np.min(window_lows)
            price_range = range_high - range_low
            
            if price_range <= 0:
                continue
            
            # Create price level buckets
            level_size = price_range / self.price_levels
            
            # Distribute volume across price levels
            session_profile = defaultdict(float)
            
            for j in range(len(window_highs)):
                # Estimate volume distribution (simplified)
                # In practice, would use tick data or more sophisticated methods
                
                # Distribute volume equally across the bar's range
                bar_levels = int((window_highs[j] - window_lows[j]) / level_size) + 1
                volume_per_level = window_volumes[j] / bar_levels
                
                # Add volume to each level the bar touches
                for level_idx in range(self.price_levels):
                    level_price = range_low + level_idx * level_size
                    
                    if window_lows[j] <= level_price <= window_highs[j]:
                        session_profile[level_idx] += volume_per_level
                        
                        # Add to overall profile
                        if level_idx not in profile['price_levels']:
                            profile['price_levels'][level_idx] = {
                                'price': level_price,
                                'volume': 0,
                                'tpo_count': 0  # Time Price Opportunities
                            }
                        
                        profile['price_levels'][level_idx]['volume'] += volume_per_level
                        profile['price_levels'][level_idx]['tpo_count'] += 1
                        profile['total_volume'] += volume_per_level
            
            # Store session info
            profile['sessions'].append({
                'start_idx': start_idx,
                'end_idx': i,
                'range_high': range_high,
                'range_low': range_low,
                'session_volume': np.sum(window_volumes),
                'close': window_closes[-1]
            })
        
        return profile
    
    def _calculate_value_areas(self, profiles: Dict[int, Dict]) -> Dict[int, Dict]:
        """Calculate value areas for each profile."""
        value_areas = {}
        
        for scale, profile in profiles.items():
            if not profile['price_levels']:
                value_areas[scale] = {
                    'poc': None,  # Point of Control
                    'vah': None,  # Value Area High
                    'val': None,  # Value Area Low
                    'value_area_volume': 0
                }
                continue
            
            # Sort levels by volume
            sorted_levels = sorted(
                profile['price_levels'].items(),
                key=lambda x: x[1]['volume'],
                reverse=True
            )
            
            if not sorted_levels:
                continue
            
            # Point of Control (highest volume level)
            poc_idx, poc_data = sorted_levels[0]
            poc_price = poc_data['price']
            
            # Calculate value area
            target_volume = profile['total_volume'] * self.value_area_pct
            accumulated_volume = 0
            value_area_levels = []
            
            for level_idx, level_data in sorted_levels:
                accumulated_volume += level_data['volume']
                value_area_levels.append((level_idx, level_data['price']))
                
                if accumulated_volume >= target_volume:
                    break
            
            # Determine VAH and VAL
            if value_area_levels:
                prices = [price for _, price in value_area_levels]
                vah = max(prices)
                val = min(prices)
            else:
                vah = val = poc_price
            
            value_areas[scale] = {
                'poc': poc_price,
                'poc_volume': poc_data['volume'],
                'vah': vah,
                'val': val,
                'value_area_volume': accumulated_volume,
                'value_area_pct': accumulated_volume / profile['total_volume'] if profile['total_volume'] > 0 else 0
            }
        
        return value_areas
    
    def _identify_market_structure(self, profiles: Dict[int, Dict], 
                                  value_areas: Dict[int, Dict]) -> Dict:
        """Identify market structure from profiles."""
        structure = {
            'profile_shape': 'undefined',
            'trend_direction': 'neutral',
            'balance_state': 'balanced',
            'structure_strength': 0.0
        }
        
        # Analyze the main scale profile
        main_scale = self.fractal_scales[0]
        if main_scale not in profiles or not profiles[main_scale]['sessions']:
            return structure
        
        profile = profiles[main_scale]
        va = value_areas[main_scale]
        
        if not va['poc']:
            return structure
        
        # Analyze profile shape
        shape = self._analyze_profile_shape(profile, va)
        structure['profile_shape'] = shape
        
        # Determine trend from session progression
        sessions = profile['sessions']
        if len(sessions) >= 2:
            first_close = sessions[0]['close']
            last_close = sessions[-1]['close']
            
            trend_strength = (last_close - first_close) / first_close
            
            if trend_strength > 0.02:
                structure['trend_direction'] = 'up'
            elif trend_strength < -0.02:
                structure['trend_direction'] = 'down'
            
            structure['structure_strength'] = abs(trend_strength)
        
        # Analyze balance state
        if len(sessions) >= 3:
            recent_sessions = sessions[-3:]
            poc_stability = self._calculate_poc_stability(recent_sessions, profiles, value_areas)
            
            if poc_stability > 0.8:
                structure['balance_state'] = 'balanced'
            elif poc_stability > 0.5:
                structure['balance_state'] = 'rotating'
            else:
                structure['balance_state'] = 'trending'
        
        return structure
    
    def _analyze_profile_shape(self, profile: Dict, value_area: Dict) -> str:
        """Analyze the shape of the market profile."""
        if not profile['price_levels']:
            return 'undefined'
        
        # Get volume distribution
        volumes = [level['volume'] for level in profile['price_levels'].values()]
        prices = [level['price'] for level in profile['price_levels'].values()]
        
        if not volumes:
            return 'undefined'
        
        # Find POC position
        poc_price = value_area['poc']
        price_range = max(prices) - min(prices)
        
        if price_range == 0:
            return 'undefined'
        
        poc_position = (poc_price - min(prices)) / price_range
        
        # Analyze distribution shape
        volume_std = np.std(volumes)
        volume_mean = np.mean(volumes)
        
        if volume_std / volume_mean < 0.5:
            shape = 'uniform'
        elif poc_position < 0.3:
            shape = 'p_shaped'  # Selling
        elif poc_position > 0.7:
            shape = 'b_shaped'  # Buying
        elif 0.4 <= poc_position <= 0.6:
            shape = 'd_shaped'  # Balanced
        else:
            shape = 'irregular'
        
        return shape
    
    def _calculate_poc_stability(self, sessions: List[Dict], profiles: Dict[int, Dict],
                                value_areas: Dict[int, Dict]) -> float:
        """Calculate POC stability across sessions."""
        # Simplified stability calculation
        # In practice, would track POC movement across sessions
        
        if len(sessions) < 2:
            return 0.5
        
        # Use price range stability as proxy
        ranges = [(s['range_high'] - s['range_low']) for s in sessions]
        range_std = np.std(ranges)
        range_mean = np.mean(ranges)
        
        if range_mean > 0:
            stability = 1 - min(1, range_std / range_mean)
        else:
            stability = 0.5
        
        return stability
    
    def _calculate_profile_statistics(self, profiles: Dict[int, Dict]) -> Dict:
        """Calculate statistics for market profiles."""
        stats = {}
        
        for scale, profile in profiles.items():
            if not profile['price_levels']:
                stats[scale] = {
                    'levels_count': 0,
                    'volume_concentration': 0,
                    'price_efficiency': 0
                }
                continue
            
            volumes = [level['volume'] for level in profile['price_levels'].values()]
            tpo_counts = [level['tpo_count'] for level in profile['price_levels'].values()]
            
            # Volume concentration (Gini coefficient)
            concentration = self._calculate_gini_coefficient(volumes)
            
            # Price efficiency (volume-weighted price movement)
            if profile['sessions']:
                total_movement = sum(abs(s['range_high'] - s['range_low']) for s in profile['sessions'])
                total_volume = profile['total_volume']
                
                if total_volume > 0:
                    efficiency = total_movement / (total_volume ** 0.5)
                else:
                    efficiency = 0
            else:
                efficiency = 0
            
            stats[scale] = {
                'levels_count': len(profile['price_levels']),
                'volume_concentration': concentration,
                'price_efficiency': efficiency,
                'avg_tpo': np.mean(tpo_counts) if tpo_counts else 0,
                'max_tpo': max(tpo_counts) if tpo_counts else 0
            }
        
        return stats
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for concentration measurement."""
        if not values or sum(values) == 0:
            return 0.0
        
        # Sort values
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini
        cumsum = 0
        for i, value in enumerate(sorted_values):
            cumsum += value * (n - i)
        
        total = sum(sorted_values)
        gini = (n + 1 - 2 * cumsum / total) / n
        
        return max(0, min(1, gini))
    
    def _generate_profile_signals(self, profiles: Dict[int, Dict],
                                 value_areas: Dict[int, Dict],
                                 closes: np.ndarray) -> np.ndarray:
        """Generate trading signals from market profile."""
        signals = np.zeros(len(closes))
        
        # Use main scale for signals
        main_scale = self.fractal_scales[0]
        if main_scale not in profiles:
            return signals
        
        profile = profiles[main_scale]
        va = value_areas[main_scale]
        
        if not va['poc'] or not profile['sessions']:
            return signals
        
        # Generate signals for each session
        for session in profile['sessions']:
            if session['end_idx'] >= len(closes):
                continue
            
            current_price = closes[session['end_idx']]
            
            # Value area signals
            if va['vah'] and va['val']:
                if current_price > va['vah']:
                    # Price above value area - potential short
                    signals[session['end_idx']] = -0.5
                elif current_price < va['val']:
                    # Price below value area - potential long
                    signals[session['end_idx']] = 0.5
                elif abs(current_price - va['poc']) / va['poc'] < 0.001:
                    # Price at POC - neutral/balanced
                    signals[session['end_idx']] = 0
            
            # Profile shape signals
            shape = self._analyze_profile_shape(profile, va)
            
            if shape == 'p_shaped':
                # P-shaped profile - selling pressure
                signals[session['end_idx']] -= 0.25
            elif shape == 'b_shaped':
                # B-shaped profile - buying pressure
                signals[session['end_idx']] += 0.25
        
        # Clip signals to [-1, 1]
        signals = np.clip(signals, -1, 1)
        
        return signals
    
    def _calculate_fractal_metrics(self, profiles: Dict[int, Dict]) -> Dict:
        """Calculate fractal metrics of market profiles."""
        metrics = {}
        
        for scale, profile in profiles.items():
            if not profile['price_levels']:
                metrics[scale] = {
                    'fractal_dimension': 0,
                    'self_similarity': 0,
                    'scaling_exponent': 0
                }
                continue
            
            # Extract volume distribution
            volumes = [level['volume'] for level in profile['price_levels'].values()]
            
            if len(volumes) < 5:
                continue
            
            # Calculate fractal dimension of volume distribution
            fractal_dim = self._calculate_distribution_fractal_dimension(volumes)
            
            # Calculate self-similarity across scales
            self_similarity = 0.5  # Placeholder - would compare across scales
            
            # Calculate scaling exponent
            scaling_exp = self._calculate_scaling_exponent(volumes)
            
            metrics[scale] = {
                'fractal_dimension': fractal_dim,
                'self_similarity': self_similarity,
                'scaling_exponent': scaling_exp
            }
        
        return metrics
    
    def _calculate_distribution_fractal_dimension(self, distribution: List[float]) -> float:
        """Calculate fractal dimension of distribution."""
        if len(distribution) < 5:
            return 0.0
        
        # Use box-counting method on distribution
        # Normalize distribution
        total = sum(distribution)
        if total == 0:
            return 0.0
        
        normalized = [v / total for v in distribution]
        
        # Simple fractal dimension estimate
        # Count non-zero boxes at different scales
        scales = [1, 2, 4, 8]
        counts = []
        
        for scale in scales:
            if scale > len(normalized):
                continue
            
            # Group into boxes
            box_count = 0
            for i in range(0, len(normalized), scale):
                box_sum = sum(normalized[i:i+scale])
                if box_sum > 0.001:  # Threshold
                    box_count += 1
            
            if box_count > 0:
                counts.append((scale, box_count))
        
        # Calculate dimension from scaling
        if len(counts) >= 2:
            log_scales = [np.log(s) for s, _ in counts]
            log_counts = [np.log(c) for _, c in counts]
            
            # Linear regression
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return max(0, min(2, -slope))
        
        return 1.0
    
    def _calculate_scaling_exponent(self, values: List[float]) -> float:
        """Calculate scaling exponent of distribution."""
        if len(values) < 10:
            return 0.0
        
        # Sort values in descending order
        sorted_values = sorted(values, reverse=True)
        
        # Fit power law to tail
        x = np.arange(1, len(sorted_values) + 1)
        y = sorted_values
        
        # Use log-log regression on non-zero values
        valid_mask = np.array(y) > 0
        if np.sum(valid_mask) < 2:
            return 0.0
        
        log_x = np.log(x[valid_mask])
        log_y = np.log(np.array(y)[valid_mask])
        
        # Linear regression in log-log space
        slope = np.polyfit(log_x, log_y, 1)[0]
        
        return -slope  # Negative slope is the scaling exponent
    
    def _identify_key_levels(self, profiles: Dict[int, Dict],
                            value_areas: Dict[int, Dict]) -> List[Dict]:
        """Identify key price levels from profiles."""
        key_levels = []
        
        # Collect POCs, VAHs, and VALs from all scales
        for scale in self.fractal_scales:
            if scale not in value_areas:
                continue
            
            va = value_areas[scale]
            
            if va['poc']:
                key_levels.append({
                    'level': va['poc'],
                    'type': 'poc',
                    'scale': scale,
                    'strength': va['poc_volume'] / profiles[scale]['total_volume'] if profiles[scale]['total_volume'] > 0 else 0
                })
            
            if va['vah']:
                key_levels.append({
                    'level': va['vah'],
                    'type': 'vah',
                    'scale': scale,
                    'strength': 0.5  # Fixed strength for VAH
                })
            
            if va['val']:
                key_levels.append({
                    'level': va['val'],
                    'type': 'val',
                    'scale': scale,
                    'strength': 0.5  # Fixed strength for VAL
                })
        
        # Find high volume nodes (HVN) and low volume nodes (LVN)
        main_profile = profiles[self.fractal_scales[0]]
        if main_profile['price_levels']:
            volumes = [(level['price'], level['volume']) for level in main_profile['price_levels'].values()]
            volumes.sort(key=lambda x: x[1], reverse=True)
            
            # Top 3 HVNs
            for i in range(min(3, len(volumes))):
                price, volume = volumes[i]
                key_levels.append({
                    'level': price,
                    'type': 'hvn',
                    'scale': self.fractal_scales[0],
                    'strength': volume / main_profile['total_volume'] if main_profile['total_volume'] > 0 else 0
                })
            
            # Bottom 3 LVNs (if they meet minimum threshold)
            lvn_candidates = [v for v in volumes if v[1] > main_profile['total_volume'] * self.min_level_threshold]
            for i in range(min(3, len(lvn_candidates))):
                price, volume = lvn_candidates[-(i+1)]
                key_levels.append({
                    'level': price,
                    'type': 'lvn',
                    'scale': self.fractal_scales[0],
                    'strength': 1 - (volume / main_profile['total_volume']) if main_profile['total_volume'] > 0 else 0
                })
        
        # Sort by price level
        key_levels.sort(key=lambda x: x['level'])
        
        return key_levels
    
    def _serialize_profiles(self, profiles: Dict[int, Dict]) -> Dict[int, Dict]:
        """Serialize profiles for output."""
        serialized = {}
        
        for scale, profile in profiles.items():
            # Convert price levels to list format
            levels_list = []
            for level_idx, level_data in profile['price_levels'].items():
                levels_list.append({
                    'index': level_idx,
                    'price': level_data['price'],
                    'volume': level_data['volume'],
                    'tpo_count': level_data['tpo_count']
                })
            
            serialized[scale] = {
                'scale': scale,
                'levels': sorted(levels_list, key=lambda x: x['price']),
                'total_volume': profile['total_volume'],
                'sessions_count': len(profile['sessions'])
            }
        
        return serialized
    
    def _interpret_profile_state(self, market_structure: Dict,
                                profile_stats: Dict, current_signal: float) -> Dict:
        """Interpret current market profile state."""
        interpretation = {
            'profile_type': market_structure['profile_shape'],
            'market_condition': '',
            'trading_environment': '',
            'key_observations': [],
            'recommendations': []
        }
        
        # Market condition based on structure
        if market_structure['balance_state'] == 'balanced':
            interpretation['market_condition'] = 'Balanced market - range-bound trading'
            interpretation['trading_environment'] = 'Mean reversion favorable'
            interpretation['recommendations'].append('Trade from extremes to POC')
        elif market_structure['balance_state'] == 'trending':
            direction = market_structure['trend_direction']
            interpretation['market_condition'] = f'Trending market - {direction} bias'
            interpretation['trading_environment'] = 'Trend following favorable'
            interpretation['recommendations'].append(f'Follow {direction} trend with value area as support')
        else:
            interpretation['market_condition'] = 'Rotating market - transitional phase'
            interpretation['trading_environment'] = 'Mixed strategies'
            interpretation['recommendations'].append('Wait for clearer structure')
        
        # Profile shape insights
        shape_insights = {
            'd_shaped': 'Normal distribution - balanced two-way trade',
            'p_shaped': 'Selling pressure - look for shorts at upper levels',
            'b_shaped': 'Buying pressure - look for longs at lower levels',
            'uniform': 'Low conviction - wait for structure development',
            'irregular': 'Complex structure - use multiple timeframes'
        }
        
        if market_structure['profile_shape'] in shape_insights:
            interpretation['key_observations'].append(
                shape_insights[market_structure['profile_shape']]
            )
        
        # Volume concentration insights
        main_stats = profile_stats.get(self.fractal_scales[0], {})
        concentration = main_stats.get('volume_concentration', 0)
        
        if concentration > 0.7:
            interpretation['key_observations'].append('High volume concentration - strong agreement on value')
        elif concentration < 0.3:
            interpretation['key_observations'].append('Low volume concentration - price discovery mode')
        
        # Current signal interpretation
        if current_signal > 0.5:
            interpretation['recommendations'].append('Bullish signal - consider long positions')
        elif current_signal < -0.5:
            interpretation['recommendations'].append('Bearish signal - consider short positions')
        
        return interpretation


def create_fractal_market_profile(profile_period: int = 20, **kwargs) -> FractalMarketProfile:
    """Factory function to create Fractal Market Profile."""
    return FractalMarketProfile(profile_period=profile_period, **kwargs)
