"""
Volume Indicators Module for Platform3 Forex Trading
Comprehensive volume analysis indicators for market confirmation and trend analysis

This module provides advanced volume-based indicators for forex trading:
- OBV (On-Balance Volume): Trend confirmation through volume analysis
- MFI (Money Flow Index): Buying/selling pressure oscillator
- VFI (Volume Flow Indicator): Directional volume flow analysis
- AdvanceDecline: Market breadth and sentiment analysis

Features:
- Real-time volume analysis for M1-H4 timeframes
- Session-aware calculations for forex markets
- Divergence detection for trend reversals
- Volume strength and momentum scoring
- Multi-timeframe signal generation
- Performance tracking and optimization
"""

import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime

# Import volume indicators
try:
    from .OBV import OBV, OBVSignal, OBVSignalType, OBVTrend
    from .MFI import MFI, MFISignal, MFISignalType, MFITrend
    from .VFI import VFI, VFISignal, VFISignalType, VFITrend
    from .AdvanceDecline import AdvanceDecline, ADSignal, ADSignalType, ADTrend
except ImportError:
    # Fallback for direct execution
    from OBV import OBV, OBVSignal, OBVSignalType, OBVTrend
    from MFI import MFI, MFISignal, MFISignalType, MFITrend
    from VFI import VFI, VFISignal, VFISignalType, VFITrend
    from AdvanceDecline import AdvanceDecline, ADSignal, ADSignalType, ADTrend

# Configure logging
logger = logging.getLogger(__name__)

class VolumeIndicatorSuite:
    """
    Comprehensive volume indicator suite for forex trading
    Combines multiple volume indicators for enhanced analysis
    """

    def __init__(self,
                 obv_config: Optional[Dict] = None,
                 mfi_config: Optional[Dict] = None,
                 vfi_config: Optional[Dict] = None,
                 ad_config: Optional[Dict] = None):
        """
        Initialize volume indicator suite

        Args:
            obv_config: Configuration for OBV indicator
            mfi_config: Configuration for MFI indicator
            vfi_config: Configuration for VFI indicator
            ad_config: Configuration for Advance/Decline indicator
        """
        # Initialize indicators with default or custom configurations
        self.obv = OBV(**(obv_config or {}))
        self.mfi = MFI(**(mfi_config or {}))
        self.vfi = VFI(**(vfi_config or {}))
        self.advance_decline = AdvanceDecline(**(ad_config or {}))

        # Consensus analysis settings
        self.consensus_threshold = 0.6
        self.strong_consensus_threshold = 0.8

        # Performance tracking
        self.suite_performance = {
            'total_analyses': 0,
            'consensus_signals': 0,
            'strong_consensus_signals': 0,
            'accuracy': 0.0,
            'last_updated': datetime.now()
        }

        logger.info("Volume Indicator Suite initialized with OBV, MFI, VFI, and Advance/Decline")

    def analyze_volume_consensus(self,
                               high: Union[pd.Series, np.ndarray],
                               low: Union[pd.Series, np.ndarray],
                               close: Union[pd.Series, np.ndarray],
                               volume: Union[pd.Series, np.ndarray],
                               timestamps: Optional[pd.Series] = None,
                               timeframe: str = 'M15') -> Dict:
        """
        Perform comprehensive volume analysis using all indicators

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            timestamps: Optional timestamps
            timeframe: Current timeframe

        Returns:
            Dictionary containing consensus analysis results
        """
        try:
            # Calculate all volume indicators
            obv_result = self.obv.calculate_obv(close, volume, timestamps)
            mfi_result = self.mfi.calculate_mfi(high, low, close, volume, timestamps)
            vfi_result = self.vfi.calculate_vfi(high, low, close, volume, timestamps)
            ad_result = self.advance_decline.calculate_advance_decline(high, low, close, volume, timestamps)

            # Generate signals from all indicators
            obv_signals = self.obv.generate_signals(close, volume, timestamps, timeframe)
            mfi_signals = self.mfi.generate_signals(high, low, close, volume, timestamps, timeframe)
            vfi_signals = self.vfi.generate_signals(high, low, close, volume, timestamps, timeframe)
            ad_signals = self.advance_decline.generate_signals(high, low, close, volume, timestamps, timeframe)

            # Perform consensus analysis
            consensus_analysis = self._analyze_consensus(
                obv_result, mfi_result, vfi_result, ad_result,
                obv_signals, mfi_signals, vfi_signals, ad_signals
            )

            # Calculate volume strength score
            volume_strength_score = self._calculate_volume_strength_score(
                obv_result, mfi_result, vfi_result, ad_result
            )

            # Determine overall volume trend
            volume_trend = self._determine_volume_trend(
                obv_result, mfi_result, vfi_result, ad_result
            )

            result = {
                'consensus_analysis': consensus_analysis,
                'volume_strength_score': volume_strength_score,
                'volume_trend': volume_trend,
                'individual_results': {
                    'obv': obv_result,
                    'mfi': mfi_result,
                    'vfi': vfi_result,
                    'advance_decline': ad_result
                },
                'individual_signals': {
                    'obv': obv_signals,
                    'mfi': mfi_signals,
                    'vfi': vfi_signals,
                    'advance_decline': ad_signals
                },
                'timeframe': timeframe,
                'timestamp': datetime.now()
            }

            # Update performance tracking
            self.suite_performance['total_analyses'] += 1
            if consensus_analysis['consensus_strength'] >= self.consensus_threshold:
                self.suite_performance['consensus_signals'] += 1
            if consensus_analysis['consensus_strength'] >= self.strong_consensus_threshold:
                self.suite_performance['strong_consensus_signals'] += 1

            logger.debug(f"Volume consensus analysis completed: "
                        f"consensus_strength={consensus_analysis['consensus_strength']:.2f}, "
                        f"volume_trend={volume_trend}")

            return result

        except Exception as e:
            logger.error(f"Error in volume consensus analysis: {str(e)}")
            return self._empty_consensus_result()

    def _analyze_consensus(self, obv_result: Dict, mfi_result: Dict,
                          vfi_result: Dict, ad_result: Dict,
                          obv_signals: List, mfi_signals: List,
                          vfi_signals: List, ad_signals: List) -> Dict:
        """Analyze consensus among volume indicators"""
        try:
            # Extract latest trends
            obv_trend = obv_result.get('obv_trend', ['neutral'])[-1] if obv_result.get('obv_trend') else 'neutral'
            mfi_trend = mfi_result.get('mfi_trend', ['neutral'])[-1] if mfi_result.get('mfi_trend') else 'neutral'
            vfi_trend = vfi_result.get('vfi_trend', ['neutral'])[-1] if vfi_result.get('vfi_trend') else 'neutral'
            ad_trend = ad_result.get('ad_trend', ['neutral'])[-1] if ad_result.get('ad_trend') else 'neutral'

            trends = [obv_trend, mfi_trend, vfi_trend, ad_trend]

            # Count bullish/bearish/neutral trends
            bullish_count = sum(1 for trend in trends if trend == 'bullish')
            bearish_count = sum(1 for trend in trends if trend == 'bearish')
            neutral_count = sum(1 for trend in trends if trend == 'neutral')

            # Determine consensus
            total_indicators = len(trends)
            bullish_ratio = bullish_count / total_indicators
            bearish_ratio = bearish_count / total_indicators

            if bullish_ratio >= 0.75:
                consensus_direction = 'strong_bullish'
                consensus_strength = bullish_ratio
            elif bullish_ratio >= 0.5:
                consensus_direction = 'bullish'
                consensus_strength = bullish_ratio
            elif bearish_ratio >= 0.75:
                consensus_direction = 'strong_bearish'
                consensus_strength = bearish_ratio
            elif bearish_ratio >= 0.5:
                consensus_direction = 'bearish'
                consensus_strength = bearish_ratio
            else:
                consensus_direction = 'neutral'
                consensus_strength = neutral_count / total_indicators

            # Analyze signal consensus
            all_signals = obv_signals + mfi_signals + vfi_signals + ad_signals
            signal_consensus = self._analyze_signal_consensus(all_signals)

            return {
                'consensus_direction': consensus_direction,
                'consensus_strength': consensus_strength,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'signal_consensus': signal_consensus,
                'individual_trends': {
                    'obv': obv_trend,
                    'mfi': mfi_trend,
                    'vfi': vfi_trend,
                    'advance_decline': ad_trend
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing consensus: {str(e)}")
            return {
                'consensus_direction': 'neutral',
                'consensus_strength': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 4,
                'signal_consensus': {},
                'individual_trends': {}
            }

    def _analyze_signal_consensus(self, signals: List) -> Dict:
        """Analyze consensus among generated signals"""
        try:
            if not signals:
                return {'total_signals': 0, 'avg_confidence': 0.0, 'signal_types': {}}

            # Count signal types
            signal_types = {}
            total_confidence = 0.0

            for signal in signals:
                signal_type = signal.signal_type
                confidence = signal.confidence

                if signal_type not in signal_types:
                    signal_types[signal_type] = {'count': 0, 'total_confidence': 0.0}

                signal_types[signal_type]['count'] += 1
                signal_types[signal_type]['total_confidence'] += confidence
                total_confidence += confidence

            # Calculate averages
            avg_confidence = total_confidence / len(signals)

            for signal_type in signal_types:
                signal_types[signal_type]['avg_confidence'] = (
                    signal_types[signal_type]['total_confidence'] /
                    signal_types[signal_type]['count']
                )

            return {
                'total_signals': len(signals),
                'avg_confidence': avg_confidence,
                'signal_types': signal_types
            }

        except Exception as e:
            logger.error(f"Error analyzing signal consensus: {str(e)}")
            return {'total_signals': 0, 'avg_confidence': 0.0, 'signal_types': {}}

    def _calculate_volume_strength_score(self, obv_result: Dict, mfi_result: Dict,
                                       vfi_result: Dict, ad_result: Dict) -> float:
        """Calculate overall volume strength score"""
        try:
            scores = []

            # OBV volume strength
            if 'volume_strength' in obv_result and len(obv_result['volume_strength']) > 0:
                scores.append(obv_result['volume_strength'][-1])

            # MFI volume strength
            if 'volume_strength' in mfi_result and len(mfi_result['volume_strength']) > 0:
                scores.append(mfi_result['volume_strength'][-1])

            # VFI flow strength
            if 'flow_strength' in vfi_result and len(vfi_result['flow_strength']) > 0:
                scores.append(vfi_result['flow_strength'][-1])

            # A/D breadth momentum (normalized)
            if 'breadth_momentum' in ad_result and len(ad_result['breadth_momentum']) > 0:
                momentum = abs(ad_result['breadth_momentum'][-1])
                normalized_momentum = min(1.0, momentum * 5.0)  # Scale to 0-1
                scores.append(normalized_momentum)

            return np.mean(scores) if scores else 0.0

        except Exception as e:
            logger.error(f"Error calculating volume strength score: {str(e)}")
            return 0.0

    def _determine_volume_trend(self, obv_result: Dict, mfi_result: Dict,
                              vfi_result: Dict, ad_result: Dict) -> str:
        """Determine overall volume trend"""
        try:
            trends = []

            # Collect latest trends from all indicators
            if 'obv_trend' in obv_result and obv_result['obv_trend']:
                trends.append(obv_result['obv_trend'][-1])

            if 'mfi_trend' in mfi_result and mfi_result['mfi_trend']:
                trends.append(mfi_result['mfi_trend'][-1])

            if 'vfi_trend' in vfi_result and vfi_result['vfi_trend']:
                trends.append(vfi_result['vfi_trend'][-1])

            if 'ad_trend' in ad_result and ad_result['ad_trend']:
                trends.append(ad_result['ad_trend'][-1])

            if not trends:
                return 'neutral'

            # Count trend directions
            bullish_count = sum(1 for trend in trends if trend == 'bullish')
            bearish_count = sum(1 for trend in trends if trend == 'bearish')

            # Determine overall trend
            if bullish_count > bearish_count:
                return 'bullish'
            elif bearish_count > bullish_count:
                return 'bearish'
            else:
                return 'neutral'

        except Exception as e:
            logger.error(f"Error determining volume trend: {str(e)}")
            return 'neutral'

    def _empty_consensus_result(self) -> Dict:
        """Return empty consensus result"""
        return {
            'consensus_analysis': {
                'consensus_direction': 'neutral',
                'consensus_strength': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 4,
                'signal_consensus': {},
                'individual_trends': {}
            },
            'volume_strength_score': 0.0,
            'volume_trend': 'neutral',
            'individual_results': {},
            'individual_signals': {},
            'timeframe': 'unknown',
            'timestamp': datetime.now()
        }

    def get_suite_performance(self) -> Dict:
        """Get performance statistics for the entire suite"""
        try:
            # Calculate consensus accuracy
            if self.suite_performance['total_analyses'] > 0:
                consensus_rate = (self.suite_performance['consensus_signals'] /
                                self.suite_performance['total_analyses'])
                strong_consensus_rate = (self.suite_performance['strong_consensus_signals'] /
                                       self.suite_performance['total_analyses'])
            else:
                consensus_rate = 0.0
                strong_consensus_rate = 0.0

            # Get individual indicator performance
            individual_performance = {
                'obv': self.obv.get_performance_stats(),
                'mfi': self.mfi.get_performance_stats(),
                'vfi': self.vfi.get_performance_stats(),
                'advance_decline': self.advance_decline.get_performance_stats()
            }

            return {
                'suite_performance': {
                    **self.suite_performance,
                    'consensus_rate': consensus_rate,
                    'strong_consensus_rate': strong_consensus_rate
                },
                'individual_performance': individual_performance,
                'last_updated': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error getting suite performance: {str(e)}")
            return {'error': str(e)}


# Export main classes and functions
__all__ = [
    'OBV', 'OBVSignal', 'OBVSignalType', 'OBVTrend',
    'MFI', 'MFISignal', 'MFISignalType', 'MFITrend',
    'VFI', 'VFISignal', 'VFISignalType', 'VFITrend',
    'AdvanceDecline', 'ADSignal', 'ADSignalType', 'ADTrend',
    'VolumeIndicatorSuite'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Platform3 Development Team"
__description__ = "Advanced volume indicators for forex trading analysis"
