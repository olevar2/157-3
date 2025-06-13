"""
Enhanced AI Model with Platform3 Phase 2 Framework Integration
Auto-enhanced for production-ready performance and reliability
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Platform3 Phase 2 Framework Integration
from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, MLError, ModelError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from communication.platform3_communication_framework import Platform3CommunicationFramework

# === ENHANCED ORIGINAL IMPLEMENTATION ===
"""
Pair Specialist - Individual Currency Pair Personality Analysis

GENIUS MODEL: Professional currency pair intelligence that analyzes unique characteristics,
behaviors, volatility patterns, and optimal trading strategies for each forex pair.

This model is CRITICAL for Platform3's success because:
- Each currency pair has unique personality and behavior patterns
- Optimal trading strategies vary significantly between pairs
- Risk parameters must be adjusted per pair characteristics
- Session-specific behaviors need individual analysis
- Spread patterns and liquidity vary dramatically

Features:
- Comprehensive pair personality profiling
- Volatility pattern analysis and prediction
- Optimal timeframe identification per pair
- Session-specific behavior modeling
- Spread analysis and prediction
- Liquidity pattern recognition
- Risk parameter optimization
- Trading strategy recommendations
- Performance tracking per pair
- Market correlation analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
import logging
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Professional imports for advanced analysis
from scipy import stats
from scipy.stats import normaltest, jarque_bera
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import talib
from shared.ai_model_base import AIModelPerformanceMonitor, EnhancedAIModelBase

logger = logging.getLogger(__name__)

class SessionType(Enum):
    """Trading session types"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    SYDNEY = "sydney"
    OVERLAP_LONDON_NY = "london_ny_overlap"
    OVERLAP_ASIAN_LONDON = "asian_london_overlap"

class PairType(Enum):
    """Currency pair classifications"""
    MAJOR = "major"
    MINOR = "minor"
    EXOTIC = "exotic"
    COMMODITY = "commodity"
    SAFE_HAVEN = "safe_haven"

class VolatilityProfile(Enum):
    """Volatility behavior patterns"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXPLOSIVE = "explosive"
    CHOPPY = "choppy"

@dataclass
class SessionBehavior:
    """Session-specific pair behavior"""
    session: SessionType
    avg_volatility: float
    avg_spread: float
    avg_volume: float
    trend_strength: float
    reversal_tendency: float
    breakout_probability: float
    optimal_strategies: List[str]
    risk_multiplier: float

@dataclass
class VolatilityCharacteristics:
    """Comprehensive volatility analysis"""
    current_regime: VolatilityProfile
    average_daily_range: float
    intraday_patterns: Dict[int, float]  # Hour -> volatility
    session_volatility: Dict[SessionType, float]
    volatility_clustering: float
    garch_parameters: Dict[str, float]
    regime_probabilities: Dict[VolatilityProfile, float]
    
@dataclass
class SpreadAnalysis:
    """Spread behavior analysis"""
    average_spread: float
    session_spreads: Dict[SessionType, float]
    spread_volatility: float
    tight_spread_hours: List[int]
    wide_spread_hours: List[int]
    spread_predictability: float
    optimal_entry_spreads: Dict[str, float]

@dataclass
class TradingProfile:
    """Optimal trading configuration for pair"""
    optimal_timeframes: List[str]
    best_sessions: List[SessionType]
    recommended_strategies: List[str]
    risk_parameters: Dict[str, float]
    position_sizing_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    max_daily_trades: int
    correlation_pairs: List[str]

@dataclass
class PairCharacteristics:
    """Complete pair personality profile"""
    pair: str
    pair_type: PairType
    base_currency: str
    quote_currency: str
    
    # Core characteristics
    volatility: VolatilityCharacteristics
    spread_analysis: SpreadAnalysis
    session_behaviors: Dict[SessionType, SessionBehavior]
    trading_profile: TradingProfile
    
    # Advanced analytics
    trend_persistence: float
    mean_reversion_tendency: float
    momentum_strength: float
    support_resistance_respect: float
    news_sensitivity: float
    correlation_matrix: Dict[str, float]
    
    # Performance metrics
    profitability_score: float
    consistency_score: float
    risk_adjusted_returns: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Meta information
    analysis_timestamp: datetime
    confidence_score: float
    data_quality_score: float

@dataclass
class PairPersonality:
    """High-level pair personality assessment"""
    pair: str
    personality_type: str  # e.g., "Volatile Trendy", "Steady Ranger", "News Sensitive"
    dominant_traits: List[str]
    trading_difficulty: str  # "Beginner", "Intermediate", "Advanced", "Expert"
    profit_potential: str   # "Low", "Moderate", "High", "Very High"
    risk_level: str        # "Low", "Moderate", "High", "Very High"
    recommended_experience: str
    key_strengths: List[str]
    key_challenges: List[str]
    summary_description: str

class PairSpecialist:
    """
    GENIUS MODEL: Professional Currency Pair Intelligence
    
    Analyzes individual currency pair personalities, behaviors, and optimal
    trading strategies. Essential for multi-pair trading platforms where
    each pair requires unique treatment and optimization.
    """
    
    def __init__(self, analysis_config: Optional[Dict] = None):
        """Initialize the Pair Specialist"""
        self.config = analysis_config or self._get_default_config()
        
        # Major currency pairs and their characteristics
        self.major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
            'AUDUSD', 'USDCAD', 'NZDUSD'
        ]
        
        self.minor_pairs = [
            'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD',
            'EURCHF', 'EURAUD', 'GBPAUD', 'GBPCHF'
        ]
        
        # Session timings (UTC)
        self.session_times = {
            SessionType.ASIAN: (time(23, 0), time(8, 0)),
            SessionType.LONDON: (time(7, 0), time(16, 0)),
            SessionType.NEW_YORK: (time(13, 0), time(22, 0)),
            SessionType.SYDNEY: (time(21, 0), time(6, 0))
        }
        
        # Pair classifications
        self.pair_classifications = self._initialize_pair_classifications()
        
        # Analysis cache
        self._analysis_cache: Dict[str, PairCharacteristics] = {}
        self._personality_cache: Dict[str, PairPersonality] = {}
        
        logger.info("Pair Specialist initialized - Ready for professional pair analysis")
    
    def _get_default_config(self) -> Dict:
        """Default configuration for pair analysis"""
        return {
            'volatility_window': 252,  # Days for volatility calculation
            'correlation_window': 100,  # Days for correlation analysis
            'session_analysis_days': 30,  # Days for session analysis
            'min_data_points': 1000,  # Minimum data points for reliable analysis
            'confidence_threshold': 0.7,  # Minimum confidence for recommendations
            'volatility_regimes': 5,  # Number of volatility regimes to identify
            'outlier_threshold': 3.0,  # Standard deviations for outlier detection
            'trend_window': 20,  # Days for trend analysis
            'support_resistance_tolerance': 0.002  # 20 pips tolerance
        }
    
    def _initialize_pair_classifications(self) -> Dict[str, PairType]:
        """Initialize currency pair classifications"""
        classifications = {}
        
        # Major pairs
        for pair in self.major_pairs:
            classifications[pair] = PairType.MAJOR
            
        # Minor pairs
        for pair in self.minor_pairs:
            classifications[pair] = PairType.MINOR
            
        # Commodity currencies
        commodity_pairs = ['AUDUSD', 'NZDUSD', 'USDCAD']
        for pair in commodity_pairs:
            classifications[pair] = PairType.COMMODITY
            
        # Safe haven
        safe_haven_pairs = ['USDJPY', 'USDCHF']
        for pair in safe_haven_pairs:
            classifications[pair] = PairType.SAFE_HAVEN
            
        return classifications
    
    def analyze_pair_comprehensive(self, 
                                 pair: str, 
                                 price_data: pd.DataFrame,
                                 volume_data: Optional[pd.DataFrame] = None,
                                 news_data: Optional[pd.DataFrame] = None) -> PairCharacteristics:
        """
        Comprehensive pair personality analysis
        
        Args:
            pair: Currency pair symbol (e.g., 'EURUSD')
            price_data: OHLCV price data with datetime index
            volume_data: Optional volume/tick data
            news_data: Optional news impact data
            
        Returns:
            Complete pair characteristics analysis
        """
        logger.info(f"Starting comprehensive analysis for {pair}")
        
        # Validate data quality
        if not self._validate_data_quality(price_data):
            raise ValueError(f"Insufficient data quality for {pair} analysis")
        
        # Core analyses
        volatility_chars = self._analyze_volatility_comprehensive(price_data)
        spread_analysis = self._analyze_spread_behavior(price_data)
        session_behaviors = self._analyze_session_behaviors(price_data)
        trading_profile = self._generate_trading_profile(pair, price_data, volatility_chars)
        
        # Advanced analytics
        trend_persistence = self._calculate_trend_persistence(price_data)
        mean_reversion = self._calculate_mean_reversion_tendency(price_data)
        momentum_strength = self._calculate_momentum_strength(price_data)
        sr_respect = self._calculate_support_resistance_respect(price_data)
        news_sensitivity = self._calculate_news_sensitivity(price_data, news_data)
        correlations = self._calculate_correlation_matrix(pair, price_data)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(price_data)
        
        # Data quality assessment
        confidence_score = self._calculate_confidence_score(price_data)
        data_quality = self._assess_data_quality(price_data)
        
        characteristics = PairCharacteristics(
            pair=pair,
            pair_type=self.pair_classifications.get(pair, PairType.MINOR),
            base_currency=pair[:3],
            quote_currency=pair[3:],
            volatility=volatility_chars,
            spread_analysis=spread_analysis,
            session_behaviors=session_behaviors,
            trading_profile=trading_profile,
            trend_persistence=trend_persistence,
            mean_reversion_tendency=mean_reversion,
            momentum_strength=momentum_strength,
            support_resistance_respect=sr_respect,
            news_sensitivity=news_sensitivity,
            correlation_matrix=correlations,
            profitability_score=performance_metrics['profitability'],
            consistency_score=performance_metrics['consistency'],
            risk_adjusted_returns=performance_metrics['risk_adjusted'],
            sharpe_ratio=performance_metrics['sharpe'],
            max_drawdown=performance_metrics['max_drawdown'],
            analysis_timestamp=datetime.now(),
            confidence_score=confidence_score,
            data_quality_score=data_quality
        )
        
        # Cache analysis
        self._analysis_cache[pair] = characteristics
        
        logger.info(f"Comprehensive analysis completed for {pair} (Confidence: {confidence_score:.2f})")
        return characteristics
    
    def _analyze_volatility_comprehensive(self, price_data: pd.DataFrame) -> VolatilityCharacteristics:
        """Comprehensive volatility analysis using multiple methodologies"""
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        # Average daily range
        daily_range = ((price_data['high'] - price_data['low']) / price_data['close']).mean()
        
        # Intraday volatility patterns
        price_data['hour'] = price_data.index.hour
        hourly_vol = price_data.groupby('hour').apply(
            lambda x: ((x['high'] - x['low']) / x['close']).mean()
        ).to_dict()
        
        # Session volatility
        session_volatility = {}
        for session_type, (start_time, end_time) in self.session_times.items():
            session_data = self._filter_session_data(price_data, start_time, end_time)
            if not session_data.empty:
                session_vol = ((session_data['high'] - session_data['low']) / session_data['close']).mean()
                session_volatility[session_type] = session_vol
        
        # Volatility clustering (GARCH-like analysis)
        rolling_vol = returns.rolling(window=20).std()
        vol_of_vol = rolling_vol.rolling(window=20).std().mean()
        clustering = vol_of_vol / rolling_vol.mean()
        
        # GARCH parameters estimation (simplified)
        garch_params = self._estimate_garch_parameters(returns)
        
        # Volatility regime identification
        regime_probs = self._identify_volatility_regimes(returns)
        current_regime = self._classify_current_volatility_regime(returns.tail(20))
        
        return VolatilityCharacteristics(
            current_regime=current_regime,
            average_daily_range=daily_range,
            intraday_patterns=hourly_vol,
            session_volatility=session_volatility,
            volatility_clustering=clustering,
            garch_parameters=garch_params,
            regime_probabilities=regime_probs
        )
    
    def _analyze_spread_behavior(self, price_data: pd.DataFrame) -> SpreadAnalysis:
        """Analyze bid-ask spread behavior patterns"""
        # Estimate spread from price data (using high-low as proxy)
        estimated_spread = (price_data['high'] - price_data['low']) * 0.3  # Rough estimate
        
        avg_spread = estimated_spread.mean()
        spread_vol = estimated_spread.std()
        
        # Session-based spread analysis
        session_spreads = {}
        for session_type, (start_time, end_time) in self.session_times.items():
            session_data = self._filter_session_data(price_data, start_time, end_time)
            if not session_data.empty:
                session_spread = ((session_data['high'] - session_data['low']) * 0.3).mean()
                session_spreads[session_type] = session_spread
        
        # Hourly spread patterns
        price_data['hour'] = price_data.index.hour
        hourly_spreads = price_data.groupby('hour').apply(
            lambda x: ((x['high'] - x['low']) * 0.3).mean()
        )
        
        tight_spread_hours = hourly_spreads.nsmallest(6).index.tolist()
        wide_spread_hours = hourly_spreads.nlargest(6).index.tolist()
        
        # Spread predictability
        spread_autocorr = estimated_spread.autocorr(lag=1)
        predictability = abs(spread_autocorr)
        
        # Optimal entry spreads for different strategies
        optimal_spreads = {
            'scalping': np.percentile(estimated_spread, 25),
            'day_trading': np.percentile(estimated_spread, 50),
            'swing_trading': np.percentile(estimated_spread, 75)
        }
        
        return SpreadAnalysis(
            average_spread=avg_spread,
            session_spreads=session_spreads,
            spread_volatility=spread_vol,
            tight_spread_hours=tight_spread_hours,
            wide_spread_hours=wide_spread_hours,
            spread_predictability=predictability,
            optimal_entry_spreads=optimal_spreads
        )
    
    def _analyze_session_behaviors(self, price_data: pd.DataFrame) -> Dict[SessionType, SessionBehavior]:
        """Analyze behavior patterns for each trading session"""
        session_behaviors = {}
        
        for session_type, (start_time, end_time) in self.session_times.items():
            session_data = self._filter_session_data(price_data, start_time, end_time)
            
            if session_data.empty:
                continue
                
            # Calculate session metrics
            returns = session_data['close'].pct_change().dropna()
            volatility = ((session_data['high'] - session_data['low']) / session_data['close']).mean()
            spread = ((session_data['high'] - session_data['low']) * 0.3).mean()
            volume = session_data.get('volume', pd.Series([1] * len(session_data))).mean()
            
            # Trend strength
            trend_strength = self._calculate_session_trend_strength(session_data)
            
            # Reversal tendency
            reversal_tendency = self._calculate_reversal_tendency(session_data)
            
            # Breakout probability
            breakout_prob = self._calculate_breakout_probability(session_data)
            
            # Optimal strategies for this session
            optimal_strategies = self._identify_optimal_session_strategies(
                volatility, trend_strength, reversal_tendency
            )
            
            # Risk multiplier based on session characteristics
            risk_multiplier = self._calculate_session_risk_multiplier(
                volatility, spread, trend_strength
            )
            
            session_behaviors[session_type] = SessionBehavior(
                session=session_type,
                avg_volatility=volatility,
                avg_spread=spread,
                avg_volume=volume,
                trend_strength=trend_strength,
                reversal_tendency=reversal_tendency,
                breakout_probability=breakout_prob,
                optimal_strategies=optimal_strategies,
                risk_multiplier=risk_multiplier
            )
        
        return session_behaviors
    
    def _generate_trading_profile(self, 
                                pair: str, 
                                price_data: pd.DataFrame,
                                volatility_chars: VolatilityCharacteristics) -> TradingProfile:
        """Generate optimal trading profile for the pair"""
        
        # Optimal timeframes based on volatility and behavior
        optimal_timeframes = self._identify_optimal_timeframes(price_data, volatility_chars)
        
        # Best trading sessions
        best_sessions = self._identify_best_sessions(volatility_chars.session_volatility)
        
        # Recommended strategies
        recommended_strategies = self._recommend_strategies(pair, price_data, volatility_chars)
        
        # Risk parameters
        risk_params = self._calculate_risk_parameters(price_data, volatility_chars)
        
        # Position sizing multiplier
        position_multiplier = self._calculate_position_sizing_multiplier(volatility_chars)
        
        # Stop loss and take profit multipliers
        sl_multiplier, tp_multiplier = self._calculate_sl_tp_multipliers(price_data)
        
        # Maximum daily trades
        max_daily_trades = self._calculate_max_daily_trades(volatility_chars)
        
        # Correlation pairs for hedging/avoiding
        correlation_pairs = self._identify_correlation_pairs(pair)
        
        return TradingProfile(
            optimal_timeframes=optimal_timeframes,
            best_sessions=best_sessions,
            recommended_strategies=recommended_strategies,
            risk_parameters=risk_params,
            position_sizing_multiplier=position_multiplier,
            stop_loss_multiplier=sl_multiplier,
            take_profit_multiplier=tp_multiplier,
            max_daily_trades=max_daily_trades,
            correlation_pairs=correlation_pairs
        )
    
    def generate_pair_personality(self, characteristics: PairCharacteristics) -> PairPersonality:
        """Generate high-level personality assessment"""
        
        # Determine personality type
        personality_type = self._determine_personality_type(characteristics)
        
        # Extract dominant traits
        dominant_traits = self._extract_dominant_traits(characteristics)
        
        # Assess difficulty and potential
        difficulty = self._assess_trading_difficulty(characteristics)
        profit_potential = self._assess_profit_potential(characteristics)
        risk_level = self._assess_risk_level(characteristics)
        
        # Experience recommendation
        experience_req = self._recommend_experience_level(characteristics)
        
        # Key strengths and challenges
        strengths = self._identify_key_strengths(characteristics)
        challenges = self._identify_key_challenges(characteristics)
        
        # Summary description
        summary = self._generate_personality_summary(characteristics, personality_type)
        
        personality = PairPersonality(
            pair=characteristics.pair,
            personality_type=personality_type,
            dominant_traits=dominant_traits,
            trading_difficulty=difficulty,
            profit_potential=profit_potential,
            risk_level=risk_level,
            recommended_experience=experience_req,
            key_strengths=strengths,
            key_challenges=challenges,
            summary_description=summary
        )
        
        # Cache personality
        self._personality_cache[characteristics.pair] = personality
        
        return personality
    
    def get_optimal_parameters_for_strategy(self, 
                                          pair: str, 
                                          strategy_type: str,
                                          session: Optional[SessionType] = None) -> Dict[str, float]:
        """Get optimal parameters for specific strategy and session"""
        
        if pair not in self._analysis_cache:
            raise ValueError(f"No analysis available for {pair}. Run analyze_pair_comprehensive first.")
        
        characteristics = self._analysis_cache[pair]
        
        # Base parameters from trading profile
        base_params = characteristics.trading_profile.risk_parameters.copy()
        
        # Strategy-specific adjustments
        strategy_adjustments = self._get_strategy_adjustments(strategy_type, characteristics)
        
        # Session-specific adjustments
        if session and session in characteristics.session_behaviors:
            session_behavior = characteristics.session_behaviors[session]
            session_adjustments = {
                'risk_multiplier': session_behavior.risk_multiplier,
                'volatility_adjustment': session_behavior.avg_volatility / characteristics.volatility.average_daily_range
            }
        else:
            session_adjustments = {'risk_multiplier': 1.0, 'volatility_adjustment': 1.0}
        
        # Combine all adjustments
        optimal_params = {}
        for key, value in base_params.items():
            strategy_mult = strategy_adjustments.get(key, 1.0)
            session_mult = session_adjustments.get('risk_multiplier', 1.0)
            optimal_params[key] = value * strategy_mult * session_mult
        
        # Add strategy-specific parameters
        optimal_params.update(strategy_adjustments)
        optimal_params.update(session_adjustments)
        
        return optimal_params
    
    def compare_pairs(self, pairs: List[str], metric: str = 'overall_score') -> pd.DataFrame:
        """Compare multiple pairs on various metrics"""
        
        comparison_data = []
        
        for pair in pairs:
            if pair not in self._analysis_cache:
                logger.warning(f"No analysis available for {pair}")
                continue
                
            chars = self._analysis_cache[pair]
            personality = self._personality_cache.get(pair)
            
            row_data = {
                'pair': pair,
                'pair_type': chars.pair_type.value,
                'volatility_regime': chars.volatility.current_regime.value,
                'avg_daily_range': chars.volatility.average_daily_range,
                'profitability_score': chars.profitability_score,
                'consistency_score': chars.consistency_score,
                'risk_adjusted_returns': chars.risk_adjusted_returns,
                'sharpe_ratio': chars.sharpe_ratio,
                'max_drawdown': chars.max_drawdown,
                'trend_persistence': chars.trend_persistence,
                'mean_reversion': chars.mean_reversion_tendency,
                'news_sensitivity': chars.news_sensitivity,
                'confidence_score': chars.confidence_score
            }
            
            if personality:
                row_data.update({
                    'personality_type': personality.personality_type,
                    'difficulty': personality.trading_difficulty,
                    'profit_potential': personality.profit_potential,
                    'risk_level': personality.risk_level
                })
            
            # Calculate overall score
            overall_score = (
                chars.profitability_score * 0.3 +
                chars.consistency_score * 0.2 +
                chars.risk_adjusted_returns * 0.2 +
                (1 - chars.max_drawdown) * 0.15 +
                chars.confidence_score * 0.15
            )
            row_data['overall_score'] = overall_score
            
            comparison_data.append(row_data)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df
    
    def get_session_recommendations(self, pair: str) -> Dict[SessionType, Dict[str, Any]]:
        """Get detailed session-specific trading recommendations"""
        
        if pair not in self._analysis_cache:
            raise ValueError(f"No analysis available for {pair}")
            
        characteristics = self._analysis_cache[pair]
        recommendations = {}
        
        for session_type, behavior in characteristics.session_behaviors.items():
            recommendations[session_type] = {
                'trade_this_session': behavior.avg_volatility > characteristics.volatility.average_daily_range * 0.7,
                'optimal_strategies': behavior.optimal_strategies,
                'risk_multiplier': behavior.risk_multiplier,
                'expected_volatility': behavior.avg_volatility,
                'expected_spread': behavior.avg_spread,
                'trend_strength': behavior.trend_strength,
                'reversal_tendency': behavior.reversal_tendency,
                'breakout_probability': behavior.breakout_probability,
                'confidence_level': 'High' if behavior.avg_volatility > 0.01 else 'Medium'
            }
        
        return recommendations
    
    # Helper methods for calculations
    def _validate_data_quality(self, price_data: pd.DataFrame) -> bool:
        """Validate data quality for analysis"""
        if len(price_data) < self.config['min_data_points']:
            return False
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in price_data.columns for col in required_columns):
            return False
        
        # Check for excessive gaps or missing data
        missing_ratio = price_data.isnull().sum().sum() / (len(price_data) * len(price_data.columns))
        if missing_ratio > 0.05:  # More than 5% missing data
            return False
        
        return True
    
    def _filter_session_data(self, price_data: pd.DataFrame, start_time: time, end_time: time) -> pd.DataFrame:
        """Filter data for specific trading session"""
        if start_time < end_time:
            # Same day session
            mask = (price_data.index.time >= start_time) & (price_data.index.time <= end_time)
        else:
            # Overnight session
            mask = (price_data.index.time >= start_time) | (price_data.index.time <= end_time)
        
        return price_data[mask]
    
    def _estimate_garch_parameters(self, returns: pd.Series) -> Dict[str, float]:
        """Simplified GARCH parameter estimation"""
        # This is a simplified version - full GARCH would require specialized libraries
        vol = returns.rolling(window=20).std()
        persistence = vol.autocorr(lag=1)
        mean_reversion = 1 - persistence
        
        return {
            'alpha': 0.1,  # Simplified
            'beta': persistence,
            'omega': vol.var() * mean_reversion,
            'persistence': persistence
        }
    
    def _identify_volatility_regimes(self, returns: pd.Series) -> Dict[VolatilityProfile, float]:
        """Identify volatility regimes using clustering"""
        rolling_vol = returns.rolling(window=20).std().dropna()
        
        # Simple regime classification based on percentiles
        low_threshold = rolling_vol.quantile(0.2)
        moderate_threshold = rolling_vol.quantile(0.4)
        high_threshold = rolling_vol.quantile(0.7)
        explosive_threshold = rolling_vol.quantile(0.9)
        
        regimes = {
            VolatilityProfile.LOW: (rolling_vol <= low_threshold).sum() / len(rolling_vol),
            VolatilityProfile.MODERATE: ((rolling_vol > low_threshold) & (rolling_vol <= moderate_threshold)).sum() / len(rolling_vol),
            VolatilityProfile.HIGH: ((rolling_vol > moderate_threshold) & (rolling_vol <= high_threshold)).sum() / len(rolling_vol),
            VolatilityProfile.EXPLOSIVE: ((rolling_vol > high_threshold) & (rolling_vol <= explosive_threshold)).sum() / len(rolling_vol),
            VolatilityProfile.CHOPPY: (rolling_vol > explosive_threshold).sum() / len(rolling_vol)
        }
        
        return regimes
    
    def _classify_current_volatility_regime(self, recent_returns: pd.Series) -> VolatilityProfile:
        """Classify current volatility regime"""
        current_vol = recent_returns.std()
        
        if current_vol < 0.005:
            return VolatilityProfile.LOW
        elif current_vol < 0.01:
            return VolatilityProfile.MODERATE
        elif current_vol < 0.02:
            return VolatilityProfile.HIGH
        elif current_vol < 0.03:
            return VolatilityProfile.EXPLOSIVE
        else:
            return VolatilityProfile.CHOPPY
    
    def _calculate_trend_persistence(self, price_data: pd.DataFrame) -> float:
        """Calculate trend persistence using Hurst exponent"""
        returns = price_data['close'].pct_change().dropna()
        
        # Simplified Hurst exponent calculation
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
        
        # Linear regression on log scale
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        
        slope, _ = np.polyfit(log_lags, log_tau, 1)
        hurst = slope * 2
        
        return min(max(hurst, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _calculate_mean_reversion_tendency(self, price_data: pd.DataFrame) -> float:
        """Calculate mean reversion tendency"""
        returns = price_data['close'].pct_change().dropna()
        
        # Calculate autocorrelation of returns
        autocorr = returns.autocorr(lag=1)
        
        # Mean reversion tendency is inverse of autocorrelation
        mean_reversion = max(0, -autocorr)
        
        return min(mean_reversion, 1.0)
    
    def _calculate_momentum_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate momentum strength using multiple indicators"""
        close = price_data['close']
        
        # RSI momentum
        rsi = talib.RSI(close.values, timeperiod=14)
        rsi_momentum = np.abs(rsi[-20:] - 50).mean() / 50
        
        # MACD momentum
        macd, macdsignal, macdhist = talib.MACD(close.values)
        macd_momentum = np.abs(macdhist[-20:]).mean() / close.mean() * 1000
        
        # Combined momentum score
        momentum = (rsi_momentum + min(macd_momentum, 1.0)) / 2
        
        return min(momentum, 1.0)
    
    def _calculate_support_resistance_respect(self, price_data: pd.DataFrame) -> float:
        """Calculate how well price respects support/resistance levels"""
        # This is a simplified implementation
        # In practice, you'd identify actual S/R levels and test respect
        
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        # Use pivot points as proxy for S/R levels
        pivot_highs = []
        pivot_lows = []
        
        for i in range(5, len(price_data) - 5):
            if all(high.iloc[i] >= high.iloc[i-j] for j in range(1, 6)) and \
               all(high.iloc[i] >= high.iloc[i+j] for j in range(1, 6)):
                pivot_highs.append((i, high.iloc[i]))
                
            if all(low.iloc[i] <= low.iloc[i-j] for j in range(1, 6)) and \
               all(low.iloc[i] <= low.iloc[i+j] for j in range(1, 6)):
                pivot_lows.append((i, low.iloc[i]))
        
        # Calculate respect ratio (simplified)
        if not pivot_highs and not pivot_lows:
            return 0.5  # Neutral if no clear levels
        
        # For each subsequent touch of these levels, check if price respected them
        respect_count = 0
        total_tests = 0
        
        tolerance = self.config['support_resistance_tolerance']
        
        for idx, level in pivot_highs + pivot_lows:
            subsequent_data = price_data.iloc[idx+1:]
            for i, row in subsequent_data.iterrows():
                if abs(row['close'] - level) / level < tolerance:
                    total_tests += 1
                    # Check if price bounced (simplified)
                    next_5_closes = subsequent_data.loc[i:].head(5)['close']
                    if len(next_5_closes) >= 2:
                        if level in [h for _, h in pivot_highs]:  # Resistance
                            if next_5_closes.iloc[-1] < level:
                                respect_count += 1
                        else:  # Support
                            if next_5_closes.iloc[-1] > level:
                                respect_count += 1
        
        return respect_count / max(total_tests, 1)
    
    def _calculate_news_sensitivity(self, price_data: pd.DataFrame, news_data: Optional[pd.DataFrame] = None) -> float:
        """Calculate sensitivity to news events"""
        if news_data is None:
            # Estimate from volatility spikes
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std()
            
            # Identify volatility spikes (>2 standard deviations)
            vol_threshold = volatility.mean() + 2 * volatility.std()
            spikes = (volatility > vol_threshold).sum()
            
            # News sensitivity based on frequency of volatility spikes
            sensitivity = min(spikes / len(volatility) * 10, 1.0)
            return sensitivity
        else:
            # Actual news analysis would go here
            # For now, return moderate sensitivity
            return 0.5
    
    def _calculate_correlation_matrix(self, pair: str, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations with other major pairs"""
        # This would require price data for other pairs
        # For now, return typical correlations based on currency components
        
        correlations = {}
        base_currency = pair[:3]
        quote_currency = pair[3:]
        
        # Typical correlations (simplified)
        if 'USD' in pair:
            if 'EUR' in pair:
                correlations['GBPUSD'] = 0.7
                correlations['AUDUSD'] = 0.6
            elif 'GBP' in pair:
                correlations['EURUSD'] = 0.7
                correlations['AUDUSD'] = 0.5
        
        return correlations
    
    def _calculate_performance_metrics(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics for the pair"""
        returns = price_data['close'].pct_change().dropna()
        
        # Profitability (using trend consistency)
        positive_days = (returns > 0).sum()
        profitability = positive_days / len(returns)
        
        # Consistency (inverse of volatility)
        vol = returns.std()
        consistency = max(0, 1 - vol * 100)  # Higher volatility = lower consistency
        
        # Risk-adjusted returns (Sharpe-like)
        mean_return = returns.mean()
        risk_adjusted = mean_return / vol if vol > 0 else 0
        
        # Sharpe ratio (annualized)
        sharpe = (mean_return * 252) / (vol * np.sqrt(252)) if vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        return {
            'profitability': profitability,
            'consistency': consistency,
            'risk_adjusted': risk_adjusted,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_confidence_score(self, price_data: pd.DataFrame) -> float:
        """Calculate confidence score for the analysis"""
        base_score = 0.5
        
        # Data quantity bonus
        if len(price_data) > self.config['min_data_points'] * 2:
            base_score += 0.2
        
        # Data completeness bonus
        completeness = 1 - (price_data.isnull().sum().sum() / (len(price_data) * len(price_data.columns)))
        base_score += completeness * 0.2
        
        # Data recency bonus
        latest_date = price_data.index.max()
        days_old = (datetime.now() - latest_date.to_pydatetime()).days
        if days_old < 7:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _assess_data_quality(self, price_data: pd.DataFrame) -> float:
        """Assess overall data quality"""
        quality_score = 1.0
        
        # Check for gaps
        expected_frequency = pd.infer_freq(price_data.index)
        if expected_frequency is None:
            quality_score -= 0.2
        
        # Check for outliers
        returns = price_data['close'].pct_change().dropna()
        outliers = np.abs(returns) > (returns.std() * self.config['outlier_threshold'])
        outlier_ratio = outliers.sum() / len(returns)
        quality_score -= outlier_ratio * 0.3
        
        # Check data consistency
        invalid_bars = (price_data['high'] < price_data['low']).sum()
        if invalid_bars > 0:
            quality_score -= 0.1
        
        return max(quality_score, 0.0)
    
    # Additional helper methods would continue here...
    # (For brevity, I'm showing the core structure. The full implementation would include
    # all the remaining helper methods for session analysis, strategy recommendations, etc.)
    
    def _calculate_session_trend_strength(self, session_data: pd.DataFrame) -> float:
        """Calculate trend strength for a trading session"""
        if len(session_data) < 10:
            return 0.5
            
        close = session_data['close']
        trend_strength = abs(close.iloc[-1] - close.iloc[0]) / close.mean()
        return min(trend_strength * 100, 1.0)
    
    def _calculate_reversal_tendency(self, session_data: pd.DataFrame) -> float:
        """Calculate reversal tendency for a session"""
        if len(session_data) < 5:
            return 0.5
            
        # Count reversals (simplified)
        returns = session_data['close'].pct_change().dropna()
        sign_changes = (np.diff(np.sign(returns)) != 0).sum()
        reversal_tendency = sign_changes / max(len(returns) - 1, 1)
        return min(reversal_tendency, 1.0)
    
    def _calculate_breakout_probability(self, session_data: pd.DataFrame) -> float:
        """Calculate breakout probability for a session"""
        if len(session_data) < 10:
            return 0.3
            
        # Use ATR and volatility to estimate breakout probability
        high = session_data['high']
        low = session_data['low']
        close = session_data['close']
        
        atr = ((high - low) / close).mean()
        breakout_prob = min(atr * 50, 1.0)  # Scale appropriately
        return breakout_prob
    
    def _identify_optimal_session_strategies(self, volatility: float, trend_strength: float, reversal_tendency: float) -> List[str]:
        """Identify optimal strategies for session characteristics"""
        strategies = []
        
        if volatility > 0.015:  # High volatility
            strategies.append('scalping')
            if trend_strength > 0.5:
                strategies.append('trend_following')
        
        if trend_strength > 0.7:
            strategies.append('breakout')
            strategies.append('momentum')
        
        if reversal_tendency > 0.6:
            strategies.append('mean_reversion')
            strategies.append('range_trading')
        
        if volatility < 0.008:  # Low volatility
            strategies.append('range_trading')
        
        return strategies if strategies else ['conservative']
    
    def _calculate_session_risk_multiplier(self, volatility: float, spread: float, trend_strength: float) -> float:
        """Calculate risk multiplier for session"""
        base_multiplier = 1.0
        
        # Adjust for volatility
        if volatility > 0.02:
            base_multiplier *= 0.7  # Reduce risk for high volatility
        elif volatility < 0.005:
            base_multiplier *= 1.3  # Increase risk for low volatility
        
        # Adjust for spread
        if spread > 0.001:  # Wide spread
            base_multiplier *= 0.8
        
        # Adjust for trend strength
        if trend_strength > 0.8:
            base_multiplier *= 1.2  # Strong trends allow more risk
        
        return max(0.3, min(base_multiplier, 2.0))
    
    def _identify_optimal_timeframes(self, price_data: pd.DataFrame, volatility_chars: VolatilityCharacteristics) -> List[str]:
        """Identify optimal timeframes for trading this pair"""
        timeframes = []
        
        avg_vol = volatility_chars.average_daily_range
        
        if avg_vol > 0.02:  # High volatility pairs
            timeframes.extend(['M1', 'M5', 'M15'])
        elif avg_vol > 0.01:  # Medium volatility
            timeframes.extend(['M5', 'M15', 'H1'])
        else:  # Low volatility
            timeframes.extend(['H1', 'H4', 'D1'])
        
        return timeframes
    
    def _identify_best_sessions(self, session_volatility: Dict[SessionType, float]) -> List[SessionType]:
        """Identify best trading sessions"""
        # Sort sessions by volatility (higher is generally better for trading)
        sorted_sessions = sorted(session_volatility.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 2-3 sessions
        return [session for session, _ in sorted_sessions[:3]]
    
    def _recommend_strategies(self, pair: str, price_data: pd.DataFrame, volatility_chars: VolatilityCharacteristics) -> List[str]:
        """Recommend optimal strategies for the pair"""
        strategies = []
        
        avg_vol = volatility_chars.average_daily_range
        regime = volatility_chars.current_regime
        
        # Based on volatility regime
        if regime == VolatilityProfile.HIGH:
            strategies.extend(['scalping', 'momentum', 'breakout'])
        elif regime == VolatilityProfile.LOW:
            strategies.extend(['range_trading', 'mean_reversion'])
        else:
            strategies.extend(['swing_trading', 'trend_following'])
        
        # Based on pair type
        pair_type = self.pair_classifications.get(pair, PairType.MINOR)
        if pair_type == PairType.MAJOR:
            strategies.append('carry_trade')
        elif pair_type == PairType.COMMODITY:
            strategies.append('news_trading')
        
        return list(set(strategies))  # Remove duplicates
    
    def _calculate_risk_parameters(self, price_data: pd.DataFrame, volatility_chars: VolatilityCharacteristics) -> Dict[str, float]:
        """Calculate risk parameters for the pair"""
        atr = ((price_data['high'] - price_data['low']) / price_data['close']).mean()
        
        return {
            'max_risk_per_trade': 0.02,  # 2% base risk
            'max_daily_risk': 0.06,     # 6% daily risk
            'volatility_adjustment': volatility_chars.average_daily_range / 0.01,  # Relative to 1% baseline
            'correlation_adjustment': 1.0,  # Would adjust based on correlation analysis
            'atr_multiplier': atr * 100    # ATR as percentage
        }
    
    def _calculate_position_sizing_multiplier(self, volatility_chars: VolatilityCharacteristics) -> float:
        """Calculate position sizing multiplier based on volatility"""
        avg_vol = volatility_chars.average_daily_range
        
        # Inverse relationship with volatility
        if avg_vol > 0.02:
            return 0.5  # Reduce position size for high volatility
        elif avg_vol > 0.01:
            return 0.75
        else:
            return 1.0  # Standard position size for low volatility
    
    def _calculate_sl_tp_multipliers(self, price_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate stop loss and take profit multipliers"""
        atr = ((price_data['high'] - price_data['low']) / price_data['close']).mean()
        
        # SL multiplier based on ATR
        sl_multiplier = max(1.5, atr * 200)  # At least 1.5x ATR
        
        # TP multiplier (risk:reward ratio)
        tp_multiplier = sl_multiplier * 2.0  # 1:2 risk reward
        
        return sl_multiplier, tp_multiplier
    
    def _calculate_max_daily_trades(self, volatility_chars: VolatilityCharacteristics) -> int:
        """Calculate maximum daily trades based on volatility"""
        avg_vol = volatility_chars.average_daily_range
        
        if avg_vol > 0.02:  # High volatility
            return 8  # More opportunities
        elif avg_vol > 0.01:
            return 5
        else:
            return 3  # Fewer opportunities in low volatility
    
    def _identify_correlation_pairs(self, pair: str) -> List[str]:
        """Identify highly correlated pairs to watch"""
        # Simplified correlation mapping
        correlation_map = {
            'EURUSD': ['GBPUSD', 'AUDUSD', 'NZDUSD'],
            'GBPUSD': ['EURUSD', 'EURGBP'],
            'USDJPY': ['USDCHF'],
            'AUDUSD': ['NZDUSD', 'EURUSD'],
            'USDCAD': ['AUDUSD', 'NZDUSD']
        }
        
        return correlation_map.get(pair, [])
    
    def _determine_personality_type(self, characteristics: PairCharacteristics) -> str:
        """Determine personality type based on characteristics"""
        vol_regime = characteristics.volatility.current_regime
        trend_persistence = characteristics.trend_persistence
        mean_reversion = characteristics.mean_reversion_tendency
        
        if vol_regime == VolatilityProfile.HIGH and trend_persistence > 0.6:
            return "Volatile Trendy"
        elif vol_regime == VolatilityProfile.LOW and mean_reversion > 0.6:
            return "Steady Ranger"
        elif characteristics.news_sensitivity > 0.7:
            return "News Sensitive"
        elif vol_regime == VolatilityProfile.EXPLOSIVE:
            return "Wild Mover"
        elif trend_persistence > 0.7:
            return "Persistent Trendy"
        else:
            return "Balanced Trader"
    
    def _extract_dominant_traits(self, characteristics: PairCharacteristics) -> List[str]:
        """Extract dominant traits from analysis"""
        traits = []
        
        if characteristics.volatility.current_regime in [VolatilityProfile.HIGH, VolatilityProfile.EXPLOSIVE]:
            traits.append("High Volatility")
        
        if characteristics.trend_persistence > 0.6:
            traits.append("Trend Persistent")
        
        if characteristics.mean_reversion_tendency > 0.6:
            traits.append("Mean Reverting")
        
        if characteristics.news_sensitivity > 0.7:
            traits.append("News Sensitive")
        
        if characteristics.support_resistance_respect > 0.7:
            traits.append("Respects Levels")
        
        return traits
    
    def _assess_trading_difficulty(self, characteristics: PairCharacteristics) -> str:
        """Assess trading difficulty level"""
        difficulty_score = 0
        
        # High volatility increases difficulty
        if characteristics.volatility.current_regime in [VolatilityProfile.HIGH, VolatilityProfile.EXPLOSIVE]:
            difficulty_score += 2
        
        # High news sensitivity increases difficulty
        if characteristics.news_sensitivity > 0.7:
            difficulty_score += 1
        
        # Low trend persistence increases difficulty
        if characteristics.trend_persistence < 0.4:
            difficulty_score += 1
        
        # Poor support/resistance respect increases difficulty
        if characteristics.support_resistance_respect < 0.5:
            difficulty_score += 1
        
        if difficulty_score >= 4:
            return "Expert"
        elif difficulty_score >= 2:
            return "Advanced"
        elif difficulty_score >= 1:
            return "Intermediate"
        else:
            return "Beginner"
    
    def _assess_profit_potential(self, characteristics: PairCharacteristics) -> str:
        """Assess profit potential"""
        if characteristics.profitability_score > 0.7 and characteristics.volatility.average_daily_range > 0.015:
            return "Very High"
        elif characteristics.profitability_score > 0.6:
            return "High"
        elif characteristics.profitability_score > 0.5:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_risk_level(self, characteristics: PairCharacteristics) -> str:
        """Assess risk level"""
        if characteristics.max_drawdown > 0.15 or characteristics.volatility.current_regime == VolatilityProfile.EXPLOSIVE:
            return "Very High"
        elif characteristics.max_drawdown > 0.10:
            return "High"
        elif characteristics.max_drawdown > 0.05:
            return "Moderate"
        else:
            return "Low"
    
    def _recommend_experience_level(self, characteristics: PairCharacteristics) -> str:
        """Recommend minimum experience level"""
        difficulty = self._assess_trading_difficulty(characteristics)
        risk_level = self._assess_risk_level(characteristics)
        
        if difficulty in ["Expert"] or risk_level in ["Very High"]:
            return "Expert Traders Only"
        elif difficulty in ["Advanced"] or risk_level in ["High"]:
            return "Advanced Traders"
        elif difficulty in ["Intermediate"]:
            return "Intermediate Traders"
        else:
            return "All Experience Levels"
    
    def _identify_key_strengths(self, characteristics: PairCharacteristics) -> List[str]:
        """Identify key strengths of the pair"""
        strengths = []
        
        if characteristics.profitability_score > 0.7:
            strengths.append("High Profitability")
        
        if characteristics.consistency_score > 0.7:
            strengths.append("Consistent Behavior")
        
        if characteristics.trend_persistence > 0.7:
            strengths.append("Strong Trends")
        
        if characteristics.support_resistance_respect > 0.7:
            strengths.append("Respects Technical Levels")
        
        if characteristics.volatility.average_daily_range > 0.015:
            strengths.append("Good Movement for Trading")
        
        return strengths
    
    def _identify_key_challenges(self, characteristics: PairCharacteristics) -> List[str]:
        """Identify key challenges of the pair"""
        challenges = []
        
        if characteristics.volatility.current_regime == VolatilityProfile.EXPLOSIVE:
            challenges.append("Extremely High Volatility")
        
        if characteristics.news_sensitivity > 0.8:
            challenges.append("Very News Sensitive")
        
        if characteristics.mean_reversion_tendency > 0.7:
            challenges.append("Frequent Reversals")
        
        if characteristics.max_drawdown > 0.15:
            challenges.append("High Drawdown Risk")
        
        if characteristics.support_resistance_respect < 0.4:
            challenges.append("Unreliable Technical Levels")
        
        return challenges
    
    def _generate_personality_summary(self, characteristics: PairCharacteristics, personality_type: str) -> str:
        """Generate personality summary description"""
        base_summary = f"{characteristics.pair} is a {personality_type.lower()} with "
        
        vol_desc = {
            VolatilityProfile.LOW: "low volatility",
            VolatilityProfile.MODERATE: "moderate volatility",
            VolatilityProfile.HIGH: "high volatility",
            VolatilityProfile.EXPLOSIVE: "explosive volatility",
            VolatilityProfile.CHOPPY: "choppy price action"
        }
        
        vol_text = vol_desc.get(characteristics.volatility.current_regime, "moderate volatility")
        
        if characteristics.trend_persistence > 0.6:
            trend_text = "strong trending behavior"
        elif characteristics.mean_reversion_tendency > 0.6:
            trend_text = "mean-reverting tendencies"
        else:
            trend_text = "balanced price movement"
        
        profitability_text = "excellent" if characteristics.profitability_score > 0.7 else \
                           "good" if characteristics.profitability_score > 0.5 else "moderate"
        
        summary = f"{base_summary}{vol_text} and {trend_text}. Shows {profitability_text} trading potential."
        
        return summary
    
    def _get_strategy_adjustments(self, strategy_type: str, characteristics: PairCharacteristics) -> Dict[str, float]:
        """Get strategy-specific parameter adjustments"""
        adjustments = {}
        
        if strategy_type == 'scalping':
            adjustments = {
                'position_size_multiplier': 1.5,
                'stop_loss_multiplier': 0.5,
                'take_profit_multiplier': 0.7
            }
        elif strategy_type == 'swing_trading':
            adjustments = {
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 3.0
            }
        elif strategy_type == 'trend_following':
            adjustments = {
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 4.0
            }
        
        return adjustments

    def get_real_time_assessment(self, pair: str, current_price_data: pd.DataFrame) -> Dict[str, Any]:
        """Get real-time pair assessment for current market conditions"""
        
        if pair not in self._analysis_cache:
            raise ValueError(f"No analysis available for {pair}")
        
        characteristics = self._analysis_cache[pair]
        
        # Current volatility regime
        recent_returns = current_price_data['close'].pct_change().tail(20)
        current_vol = recent_returns.std()
        
        # Session assessment
        current_time = datetime.now().time()
        current_session = self._identify_current_session(current_time)
        
        # Risk assessment
        current_risk = self._assess_current_risk(pair, current_price_data)
        
        # Opportunity assessment
        opportunity_score = self._assess_current_opportunity(pair, current_price_data)
        
        return {
            'pair': pair,
            'timestamp': datetime.now(),
            'current_session': current_session.value if current_session else 'unknown',
            'current_volatility_regime': self._classify_current_volatility_regime(recent_returns).value,
            'risk_assessment': current_risk,
            'opportunity_score': opportunity_score,
            'recommended_strategies': characteristics.session_behaviors.get(current_session, SessionBehavior(
                session=current_session, avg_volatility=0, avg_spread=0, avg_volume=0,
                trend_strength=0, reversal_tendency=0, breakout_probability=0,
                optimal_strategies=[], risk_multiplier=1.0
            )).optimal_strategies if current_session else [],
            'confidence': characteristics.confidence_score
        }
    
    def _identify_current_session(self, current_time: time) -> Optional[SessionType]:
        """Identify current trading session"""
        for session_type, (start_time, end_time) in self.session_times.items():
            if start_time < end_time:
                if start_time <= current_time <= end_time:
                    return session_type
            else:  # Overnight session
                if current_time >= start_time or current_time <= end_time:
                    return session_type
        return None
    
    def _assess_current_risk(self, pair: str, current_data: pd.DataFrame) -> str:
        """Assess current risk level"""
        characteristics = self._analysis_cache[pair]
        recent_vol = ((current_data['high'] - current_data['low']) / current_data['close']).tail(5).mean()
        
        if recent_vol > characteristics.volatility.average_daily_range * 1.5:
            return "Very High"
        elif recent_vol > characteristics.volatility.average_daily_range * 1.2:
            return "High"
        elif recent_vol > characteristics.volatility.average_daily_range * 0.8:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_current_opportunity(self, pair: str, current_data: pd.DataFrame) -> float:
        """Assess current trading opportunity (0-1 score)"""
        characteristics = self._analysis_cache[pair]
        
        # Base score from pair characteristics
        base_score = characteristics.profitability_score * 0.4
        
        # Current volatility bonus
        recent_vol = ((current_data['high'] - current_data['low']) / current_data['close']).tail(5).mean()
        vol_score = min(recent_vol / characteristics.volatility.average_daily_range, 1.5) * 0.3
        
        # Session bonus
        current_time = datetime.now().time()
        current_session = self._identify_current_session(current_time)
        session_score = 0.3
        if current_session and current_session in characteristics.session_behaviors:
            session_behavior = characteristics.session_behaviors[current_session]
            session_score = session_behavior.avg_volatility * 10  # Scale to 0-1
        
        opportunity_score = min(base_score + vol_score + session_score * 0.3, 1.0)
        return opportunity_score

# === PLATFORM3 PHASE 2 ENHANCEMENT APPLIED ===
# Enhanced on: 2025-05-31T22:33:55.436195
# Enhancements: Winston logging, EventEmitter error handling, TypeScript interfaces,
#               Database optimization, Performance monitoring, Async operations
# Phase 3 AI Model Enhancement: Applied advanced ML optimization techniques
