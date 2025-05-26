"""
Volatility Indicators Module
Advanced volatility analysis indicators for forex trading
Optimized for M1-H4 timeframes and risk management
"""

from .BollingerBands import BollingerBands, BollingerBandSignal, BandType
from .ATR import ATR, ATRSignal, ATRSmoothingMethod, VolatilityRegime
from .KeltnerChannels import KeltnerChannels, KeltnerSignal, MAType, ChannelPosition
from .SuperTrend import SuperTrend, SuperTrendSignal, TrendDirection, SignalStrength
from .Vortex import VortexIndicator, VortexIndicatorSignal, VortexSignal
from .ParabolicSAR import ParabolicSAR, ParabolicSARSignal, SARTrend, SARSignalType
from .CCI import CCI, CCISignal, CCIZone, CCISignalType

import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

class VolatilityIndicatorSuite:
    """
    Comprehensive volatility indicator suite for forex trading
    Combines multiple volatility indicators for enhanced analysis
    """
    
    def __init__(self, timeframes: List[str] = None):
        """
        Initialize volatility indicator suite
        
        Args:
            timeframes: List of timeframes to analyze
        """
        self.timeframes = timeframes or ['M1', 'M5', 'M15', 'H1', 'H4']
        
        # Initialize all volatility indicators
        self.bollinger_bands = BollingerBands(period=20, std_dev=2.0, adaptive=True)
        self.atr = ATR(period=14, smoothing_method=ATRSmoothingMethod.WILDER, adaptive=True)
        self.keltner_channels = KeltnerChannels(period=20, atr_period=14, atr_multiplier=2.0, ma_type=MAType.EMA)
        self.supertrend = SuperTrend(atr_period=14, atr_multiplier=3.0, adaptive=True)
        self.vortex = VortexIndicator(period=14, adaptive=True)
        self.parabolic_sar = ParabolicSAR(initial_af=0.02, max_af=0.2, adaptive=True)
        self.cci = CCI(period=20, constant=0.015, adaptive=True)
        
        logger.info("VolatilityIndicatorSuite initialized with all indicators")
    
    def calculate_all_indicators(self, 
                                high: Union[pd.Series, np.ndarray],
                                low: Union[pd.Series, np.ndarray],
                                close: Union[pd.Series, np.ndarray],
                                timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate all volatility indicators
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            
        Returns:
            Dictionary containing all indicator results
        """
        try:
            results = {}
            
            # Bollinger Bands
            results['bollinger_bands'] = self.bollinger_bands.calculate_bands(close, timestamps)
            
            # ATR
            results['atr'] = self.atr.calculate_atr(high, low, close, timestamps)
            
            # Keltner Channels
            results['keltner_channels'] = self.keltner_channels.calculate_channels(high, low, close, timestamps)
            
            # SuperTrend
            results['supertrend'] = self.supertrend.calculate_supertrend(high, low, close, timestamps)
            
            # Vortex Indicator
            results['vortex'] = self.vortex.calculate_vortex(high, low, close, timestamps)
            
            # Parabolic SAR
            results['parabolic_sar'] = self.parabolic_sar.calculate_sar(high, low, close, timestamps)
            
            # CCI
            results['cci'] = self.cci.calculate_cci(high, low, close, timestamps)
            
            logger.debug("All volatility indicators calculated successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            return {}
    
    def generate_all_signals(self, 
                           high: Union[pd.Series, np.ndarray],
                           low: Union[pd.Series, np.ndarray],
                           close: Union[pd.Series, np.ndarray],
                           timestamps: Optional[pd.Series] = None,
                           timeframe: str = 'M15') -> Dict[str, List]:
        """
        Generate signals from all volatility indicators
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            timeframe: Current timeframe
            
        Returns:
            Dictionary containing signals from all indicators
        """
        try:
            signals = {}
            
            # Generate signals from each indicator
            signals['bollinger_bands'] = self.bollinger_bands.generate_signals(close, timestamps, timeframe)
            signals['atr'] = self.atr.generate_signals(high, low, close, timestamps, timeframe)
            signals['keltner_channels'] = self.keltner_channels.generate_signals(high, low, close, timestamps, timeframe)
            signals['supertrend'] = self.supertrend.generate_signals(high, low, close, timestamps, timeframe)
            signals['vortex'] = self.vortex.generate_signals(high, low, close, timestamps, timeframe)
            signals['parabolic_sar'] = self.parabolic_sar.generate_signals(high, low, close, timestamps, timeframe)
            signals['cci'] = self.cci.generate_signals(high, low, close, timestamps, timeframe)
            
            logger.info(f"Generated signals from all volatility indicators for timeframe {timeframe}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating volatility signals: {str(e)}")
            return {}
    
    def get_volatility_consensus(self, 
                               high: Union[pd.Series, np.ndarray],
                               low: Union[pd.Series, np.ndarray],
                               close: Union[pd.Series, np.ndarray],
                               timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Get volatility consensus from multiple indicators
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timestamps: Optional timestamps
            
        Returns:
            Dictionary containing volatility consensus analysis
        """
        try:
            # Calculate all indicators
            results = self.calculate_all_indicators(high, low, close, timestamps)
            
            if not results:
                return {}
            
            # Extract latest values
            latest_price = close.iloc[-1] if isinstance(close, pd.Series) else close[-1]
            
            # Volatility regime analysis
            volatility_regime = self._analyze_volatility_regime(results)
            
            # Trend consensus
            trend_consensus = self._analyze_trend_consensus(results, latest_price)
            
            # Risk level assessment
            risk_level = self._assess_risk_level(results)
            
            # Signal strength consensus
            signal_strength = self._calculate_signal_strength_consensus(results)
            
            consensus = {
                'volatility_regime': volatility_regime,
                'trend_consensus': trend_consensus,
                'risk_level': risk_level,
                'signal_strength': signal_strength,
                'latest_price': latest_price,
                'indicator_count': len([r for r in results.values() if r]),
                'timestamp': timestamps.iloc[-1] if timestamps is not None else None
            }
            
            logger.debug(f"Volatility consensus: regime={volatility_regime}, trend={trend_consensus}, risk={risk_level}")
            return consensus
            
        except Exception as e:
            logger.error(f"Error calculating volatility consensus: {str(e)}")
            return {}
    
    def _analyze_volatility_regime(self, results: Dict[str, Any]) -> str:
        """Analyze overall volatility regime"""
        try:
            regime_scores = []
            
            # ATR regime
            if 'atr' in results and 'volatility_regimes' in results['atr']:
                atr_regime = results['atr']['volatility_regimes'][-1]
                if atr_regime == 'low':
                    regime_scores.append(1)
                elif atr_regime == 'normal':
                    regime_scores.append(2)
                elif atr_regime == 'high':
                    regime_scores.append(3)
                else:  # extreme
                    regime_scores.append(4)
            
            # Bollinger Bands squeeze/expansion
            if 'bollinger_bands' in results and 'squeeze_level' in results['bollinger_bands']:
                squeeze = results['bollinger_bands']['squeeze_level'][-1]
                if squeeze > 0.7:
                    regime_scores.append(1)  # Low volatility
                elif squeeze < 0.3:
                    regime_scores.append(3)  # High volatility
                else:
                    regime_scores.append(2)  # Normal volatility
            
            # Calculate average regime
            if regime_scores:
                avg_score = np.mean(regime_scores)
                if avg_score < 1.5:
                    return 'LOW'
                elif avg_score < 2.5:
                    return 'NORMAL'
                elif avg_score < 3.5:
                    return 'HIGH'
                else:
                    return 'EXTREME'
            
            return 'NORMAL'
            
        except Exception as e:
            logger.error(f"Error analyzing volatility regime: {str(e)}")
            return 'NORMAL'
    
    def _analyze_trend_consensus(self, results: Dict[str, Any], price: float) -> str:
        """Analyze trend consensus from multiple indicators"""
        try:
            trend_votes = []
            
            # SuperTrend
            if 'supertrend' in results and 'trend_directions' in results['supertrend']:
                st_trend = results['supertrend']['trend_directions'][-1]
                trend_votes.append(1 if st_trend == 'uptrend' else -1)
            
            # Parabolic SAR
            if 'parabolic_sar' in results and 'trend_directions' in results['parabolic_sar']:
                sar_trend = results['parabolic_sar']['trend_directions'][-1]
                trend_votes.append(1 if sar_trend == 'uptrend' else -1)
            
            # Bollinger Bands position
            if 'bollinger_bands' in results and 'percent_b' in results['bollinger_bands']:
                percent_b = results['bollinger_bands']['percent_b'][-1]
                if percent_b > 0.6:
                    trend_votes.append(1)
                elif percent_b < 0.4:
                    trend_votes.append(-1)
            
            # Keltner Channels position
            if 'keltner_channels' in results and 'price_positions' in results['keltner_channels']:
                kc_position = results['keltner_channels']['price_positions'][-1]
                if 'upper' in kc_position or 'above' in kc_position:
                    trend_votes.append(1)
                elif 'lower' in kc_position or 'below' in kc_position:
                    trend_votes.append(-1)
            
            # Calculate consensus
            if trend_votes:
                consensus_score = np.mean(trend_votes)
                if consensus_score > 0.3:
                    return 'BULLISH'
                elif consensus_score < -0.3:
                    return 'BEARISH'
                else:
                    return 'NEUTRAL'
            
            return 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"Error analyzing trend consensus: {str(e)}")
            return 'NEUTRAL'
    
    def _assess_risk_level(self, results: Dict[str, Any]) -> str:
        """Assess overall risk level"""
        try:
            risk_scores = []
            
            # ATR risk level
            if 'atr' in results and 'risk_levels' in results['atr']:
                atr_risk = results['atr']['risk_levels'][-1]
                if atr_risk == 'LOW':
                    risk_scores.append(1)
                elif atr_risk == 'MEDIUM':
                    risk_scores.append(2)
                elif atr_risk == 'HIGH':
                    risk_scores.append(3)
                else:  # EXTREME
                    risk_scores.append(4)
            
            # CCI extreme readings
            if 'cci' in results and 'cci_values' in results['cci']:
                cci_value = abs(results['cci']['cci_values'][-1])
                if cci_value > 200:
                    risk_scores.append(4)
                elif cci_value > 100:
                    risk_scores.append(3)
                else:
                    risk_scores.append(2)
            
            # Calculate average risk
            if risk_scores:
                avg_risk = np.mean(risk_scores)
                if avg_risk < 1.5:
                    return 'LOW'
                elif avg_risk < 2.5:
                    return 'MEDIUM'
                elif avg_risk < 3.5:
                    return 'HIGH'
                else:
                    return 'EXTREME'
            
            return 'MEDIUM'
            
        except Exception as e:
            logger.error(f"Error assessing risk level: {str(e)}")
            return 'MEDIUM'
    
    def _calculate_signal_strength_consensus(self, results: Dict[str, Any]) -> float:
        """Calculate overall signal strength consensus"""
        try:
            strength_scores = []
            
            # Collect strength indicators from various sources
            if 'supertrend' in results and 'trend_strength' in results['supertrend']:
                strength_scores.append(min(1.0, results['supertrend']['trend_strength'][-1] / 2.0))
            
            if 'vortex' in results and 'trend_strength' in results['vortex']:
                strength_scores.append(min(1.0, results['vortex']['trend_strength'][-1] / 2.0))
            
            if 'atr' in results and 'trend_strength' in results['atr']:
                strength_scores.append(min(1.0, results['atr']['trend_strength'][-1] / 2.0))
            
            # Calculate average strength
            if strength_scores:
                return np.mean(strength_scores)
            
            return 0.5  # Default moderate strength
            
        except Exception as e:
            logger.error(f"Error calculating signal strength consensus: {str(e)}")
            return 0.5
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Get performance summary from all indicators"""
        try:
            performance = {}
            
            performance['bollinger_bands'] = self.bollinger_bands.get_performance_stats()
            performance['atr'] = self.atr.get_performance_stats()
            performance['keltner_channels'] = self.keltner_channels.get_performance_stats()
            performance['supertrend'] = self.supertrend.get_performance_stats()
            performance['vortex'] = self.vortex.get_performance_stats()
            performance['parabolic_sar'] = self.parabolic_sar.get_performance_stats()
            performance['cci'] = self.cci.get_performance_stats()
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}
    
    def reset_all_performance_stats(self):
        """Reset performance statistics for all indicators"""
        try:
            self.bollinger_bands.reset_performance_stats()
            self.atr.reset_performance_stats()
            self.keltner_channels.reset_performance_stats()
            self.supertrend.reset_performance_stats()
            self.vortex.reset_performance_stats()
            self.parabolic_sar.reset_performance_stats()
            self.cci.reset_performance_stats()
            
            logger.info("All volatility indicator performance stats reset")
            
        except Exception as e:
            logger.error(f"Error resetting performance stats: {str(e)}")

# Export all classes and enums
__all__ = [
    # Main indicator classes
    'BollingerBands', 'ATR', 'KeltnerChannels', 'SuperTrend', 
    'VortexIndicator', 'ParabolicSAR', 'CCI',
    
    # Signal classes
    'BollingerBandSignal', 'ATRSignal', 'KeltnerSignal', 'SuperTrendSignal',
    'VortexIndicatorSignal', 'ParabolicSARSignal', 'CCISignal',
    
    # Enums
    'BandType', 'ATRSmoothingMethod', 'VolatilityRegime', 'MAType', 'ChannelPosition',
    'TrendDirection', 'SignalStrength', 'VortexSignal', 'SARTrend', 'SARSignalType',
    'CCIZone', 'CCISignalType',
    
    # Suite class
    'VolatilityIndicatorSuite'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Platform3 Development Team"
__description__ = "Advanced volatility indicators for forex trading"

logger.info(f"Volatility indicators module loaded - version {__version__}")
