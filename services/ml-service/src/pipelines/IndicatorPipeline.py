"""
Indicator Pipeline for Technical Analysis
Comprehensive pipeline for computing and normalizing all technical indicators.

This module provides a unified pipeline for calculating, normalizing, and managing
technical indicators for machine learning models. It integrates with the existing
analytics service indicators and provides ML-ready feature outputs.

Key Features:
- Comprehensive indicator calculation
- Multiple normalization methods
- Feature engineering and selection
- Real-time indicator updates
- Integration with Feature Store
- Performance optimization

Author: Platform3 Analytics Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndicatorCategory(Enum):
    """Technical indicator categories."""
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CYCLE = "cycle"
    ADVANCED = "advanced"

class NormalizationMethod(Enum):
    """Normalization methods for indicators."""
    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ROBUST = "robust"
    QUANTILE = "quantile"
    TANH = "tanh"

@dataclass
class IndicatorConfig:
    """Configuration for indicator pipeline."""
    categories: List[IndicatorCategory] = field(default_factory=lambda: list(IndicatorCategory))
    normalization_method: NormalizationMethod = NormalizationMethod.ZSCORE
    lookback_period: int = 252
    update_frequency: int = 1  # bars
    feature_selection: bool = True
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    max_workers: int = 4

@dataclass
class IndicatorResult:
    """Result from indicator pipeline."""
    indicators: Dict[str, pd.Series]
    normalized_indicators: Dict[str, pd.Series]
    feature_matrix: pd.DataFrame
    selected_features: List[str]
    correlation_matrix: pd.DataFrame
    importance_scores: Dict[str, float]
    computation_time: float

class IndicatorPipeline:
    """
    Comprehensive Indicator Pipeline

    Computes and normalizes all technical indicators for ML model consumption.
    Provides feature engineering, selection, and real-time updates.
    """

    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        self.indicators_cache = {}
        self.normalization_params = {}
        self.feature_selector = None

        # Initialize indicator calculators
        self._initialize_calculators()

    def _initialize_calculators(self):
        """Initialize indicator calculation functions."""
        self.calculators = {
            IndicatorCategory.MOMENTUM: {
                'rsi': self._calculate_rsi,
                'macd': self._calculate_macd,
                'stochastic': self._calculate_stochastic,
                'williams_r': self._calculate_williams_r,
                'roc': self._calculate_roc,
                'momentum': self._calculate_momentum
            },
            IndicatorCategory.TREND: {
                'sma': self._calculate_sma,
                'ema': self._calculate_ema,
                'adx': self._calculate_adx,
                'aroon': self._calculate_aroon,
                'dmi': self._calculate_dmi,
                'trix': self._calculate_trix
            },
            IndicatorCategory.VOLATILITY: {
                'bollinger_bands': self._calculate_bollinger_bands,
                'atr': self._calculate_atr,
                'keltner_channels': self._calculate_keltner_channels,
                'donchian_channels': self._calculate_donchian_channels,
                'volatility': self._calculate_volatility
            },
            IndicatorCategory.VOLUME: {
                'obv': self._calculate_obv,
                'mfi': self._calculate_mfi,
                'vfi': self._calculate_vfi,
                'ad_line': self._calculate_ad_line,
                'cmf': self._calculate_cmf
            },
            IndicatorCategory.CYCLE: {
                'alligator': self._calculate_alligator,
                'hurst_exponent': self._calculate_hurst_exponent,
                'fisher_transform': self._calculate_fisher_transform
            },
            IndicatorCategory.ADVANCED: {
                'time_weighted_volatility': self._calculate_time_weighted_volatility,
                'pca_features': self._calculate_pca_features,
                'autoencoder_features': self._calculate_autoencoder_features
            }
        }

    async def compute_indicators(
        self,
        data: pd.DataFrame,
        symbols: List[str] = None
    ) -> IndicatorResult:
        """
        Compute all indicators for given data.

        Args:
            data: OHLCV market data
            symbols: List of symbols to process

        Returns:
            Complete indicator computation result
        """
        start_time = datetime.now()
        logger.info("Starting indicator computation...")

        if symbols is None:
            symbols = ['default']

        # Validate data
        if not self._validate_data(data):
            raise ValueError("Invalid data format for indicator computation")

        # Compute indicators by category
        all_indicators = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []

            for category in self.config.categories:
                if category in self.calculators:
                    future = executor.submit(self._compute_category_indicators, data, category)
                    futures.append((category, future))

            # Collect results
            for category, future in futures:
                try:
                    category_indicators = future.result()
                    all_indicators.update(category_indicators)
                except Exception as e:
                    logger.error(f"Error computing {category.value} indicators: {e}")

        # Normalize indicators
        normalized_indicators = self._normalize_indicators(all_indicators)

        # Create feature matrix
        feature_matrix = self._create_feature_matrix(normalized_indicators)

        # Feature selection
        selected_features = []
        if self.config.feature_selection:
            selected_features = self._select_features(feature_matrix)
            feature_matrix = feature_matrix[selected_features]

        # Calculate correlation matrix
        correlation_matrix = feature_matrix.corr()

        # Calculate feature importance
        importance_scores = self._calculate_feature_importance(feature_matrix)

        computation_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Indicator computation completed in {computation_time:.2f}s. "
                   f"Generated {len(all_indicators)} indicators, {len(selected_features)} selected features.")

        return IndicatorResult(
            indicators=all_indicators,
            normalized_indicators=normalized_indicators,
            feature_matrix=feature_matrix,
            selected_features=selected_features,
            correlation_matrix=correlation_matrix,
            importance_scores=importance_scores,
            computation_time=computation_time
        )

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Expected: {required_columns}")
            return False

        if len(data) < 50:
            logger.error("Insufficient data for indicator computation (minimum 50 bars)")
            return False

        return True

    def _compute_category_indicators(
        self,
        data: pd.DataFrame,
        category: IndicatorCategory
    ) -> Dict[str, pd.Series]:
        """Compute indicators for a specific category."""
        indicators = {}

        if category not in self.calculators:
            return indicators

        for name, calculator in self.calculators[category].items():
            try:
                result = calculator(data)
                if isinstance(result, dict):
                    # Multiple indicators returned
                    for key, value in result.items():
                        indicators[f"{name}_{key}"] = value
                else:
                    # Single indicator
                    indicators[name] = result

            except Exception as e:
                logger.warning(f"Error calculating {name}: {e}")

        return indicators

    def _normalize_indicators(
        self,
        indicators: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """Normalize indicators using specified method."""
        normalized = {}

        for name, series in indicators.items():
            try:
                if self.config.normalization_method == NormalizationMethod.NONE:
                    normalized[name] = series
                elif self.config.normalization_method == NormalizationMethod.MINMAX:
                    normalized[name] = self._minmax_normalize(series)
                elif self.config.normalization_method == NormalizationMethod.ZSCORE:
                    normalized[name] = self._zscore_normalize(series)
                elif self.config.normalization_method == NormalizationMethod.ROBUST:
                    normalized[name] = self._robust_normalize(series)
                elif self.config.normalization_method == NormalizationMethod.QUANTILE:
                    normalized[name] = self._quantile_normalize(series)
                elif self.config.normalization_method == NormalizationMethod.TANH:
                    normalized[name] = self._tanh_normalize(series)
                else:
                    normalized[name] = series

            except Exception as e:
                logger.warning(f"Error normalizing {name}: {e}")
                normalized[name] = series

        return normalized

    def _create_feature_matrix(self, indicators: Dict[str, pd.Series]) -> pd.DataFrame:
        """Create feature matrix from indicators."""
        if not indicators:
            return pd.DataFrame()

        # Align all series to common index
        common_index = None
        for series in indicators.values():
            if common_index is None:
                common_index = series.index
            else:
                common_index = common_index.intersection(series.index)

        # Create matrix
        matrix_data = {}
        for name, series in indicators.items():
            aligned_series = series.reindex(common_index)
            matrix_data[name] = aligned_series

        return pd.DataFrame(matrix_data)

    def _select_features(self, feature_matrix: pd.DataFrame) -> List[str]:
        """Select features based on correlation and variance thresholds."""
        if feature_matrix.empty:
            return []

        selected = list(feature_matrix.columns)

        # Remove low variance features
        variances = feature_matrix.var()
        low_variance = variances[variances < self.config.variance_threshold].index
        selected = [f for f in selected if f not in low_variance]

        # Remove highly correlated features
        if len(selected) > 1:
            corr_matrix = feature_matrix[selected].corr().abs()

            # Find pairs of highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.config.correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

            # Remove one feature from each highly correlated pair
            to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                if feat1 not in to_remove and feat2 not in to_remove:
                    # Remove the one with lower variance
                    if variances[feat1] < variances[feat2]:
                        to_remove.add(feat1)
                    else:
                        to_remove.add(feat2)

            selected = [f for f in selected if f not in to_remove]

        return selected

    def _calculate_feature_importance(self, feature_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores."""
        if feature_matrix.empty:
            return {}

        # Simple variance-based importance
        variances = feature_matrix.var()
        max_var = variances.max()

        if max_var == 0:
            return {col: 0.0 for col in feature_matrix.columns}

        importance = (variances / max_var).to_dict()
        return importance

    # Normalization methods
    def _minmax_normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalization."""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    def _zscore_normalize(self, series: pd.Series) -> pd.Series:
        """Z-score normalization."""
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return pd.Series(0.0, index=series.index)
        return (series - mean_val) / std_val

    def _robust_normalize(self, series: pd.Series) -> pd.Series:
        """Robust normalization using median and IQR."""
        median_val = series.median()
        q75 = series.quantile(0.75)
        q25 = series.quantile(0.25)
        iqr = q75 - q25
        if iqr == 0:
            return pd.Series(0.0, index=series.index)
        return (series - median_val) / iqr

    def _quantile_normalize(self, series: pd.Series) -> pd.Series:
        """Quantile normalization."""
        return series.rank(pct=True)

    def _tanh_normalize(self, series: pd.Series) -> pd.Series:
        """Tanh normalization."""
        mean_val = series.mean()
        std_val = series.std()
        if std_val == 0:
            return pd.Series(0.0, index=series.index)
        return np.tanh((series - mean_val) / (2 * std_val))

    # Basic indicator calculations (simplified implementations)
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD."""
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        return {
            'line': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Stochastic oscillator."""
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()

        return {
            'k': k_percent,
            'd': d_percent
        }

    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()
        return -100 * ((high_max - data['close']) / (high_max - low_min))

    def _calculate_roc(self, data: pd.DataFrame, period: int = 12) -> pd.Series:
        """Calculate Rate of Change."""
        return ((data['close'] / data['close'].shift(period)) - 1) * 100

    def _calculate_momentum(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate Momentum."""
        return data['close'] - data['close'].shift(period)

    def _calculate_sma(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data['close'].rolling(window=period).mean()

    def _calculate_ema(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data['close'].ewm(span=period).mean()

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified)."""
        high_diff = data['high'].diff()
        low_diff = data['low'].diff().abs()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low'] - data['close'].shift()).abs()
        ], axis=1).max(axis=1)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.rolling(window=period).mean()

    def _calculate_aroon(self, data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Aroon."""
        aroon_up = 100 * (period - data['high'].rolling(window=period).apply(lambda x: period - 1 - x.argmax())) / period
        aroon_down = 100 * (period - data['low'].rolling(window=period).apply(lambda x: period - 1 - x.argmin())) / period

        return {
            'up': aroon_up,
            'down': aroon_down,
            'oscillator': aroon_up - aroon_down
        }

    # Placeholder implementations for remaining indicators
    def _calculate_dmi(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate DMI (placeholder)."""
        return {'plus_di': data['close'] * 0, 'minus_di': data['close'] * 0}

    def _calculate_trix(self, data: pd.DataFrame) -> pd.Series:
        """Calculate TRIX (placeholder)."""
        return data['close'].pct_change() * 100

    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()

        return {
            'upper': sma + (2 * std),
            'middle': sma,
            'lower': sma - (2 * std),
            'width': (4 * std) / sma,
            'position': (data['close'] - sma) / (2 * std)
        }

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low'] - data['close'].shift()).abs()
        ], axis=1).max(axis=1)

        return tr.rolling(window=period).mean()

    def _calculate_keltner_channels(self, data: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Keltner Channels."""
        ema = data['close'].ewm(span=period).mean()
        atr = self._calculate_atr(data, period)

        return {
            'upper': ema + (2 * atr),
            'middle': ema,
            'lower': ema - (2 * atr)
        }

    def _calculate_donchian_channels(self, data: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Donchian Channels."""
        return {
            'upper': data['high'].rolling(window=period).max(),
            'lower': data['low'].rolling(window=period).min(),
            'middle': (data['high'].rolling(window=period).max() + data['low'].rolling(window=period).min()) / 2
        }

    def _calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate price volatility."""
        returns = data['close'].pct_change()
        return returns.rolling(window=period).std() * np.sqrt(252)

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['volume'].iloc[0]

        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

    def _calculate_vfi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Flow Indicator (placeholder)."""
        return data['volume'].rolling(window=14).mean()

    def _calculate_ad_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.fillna(0)
        ad_line = (clv * data['volume']).cumsum()
        return ad_line

    def _calculate_cmf(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.fillna(0)
        cmf = (clv * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        return cmf

    # Placeholder implementations for advanced indicators
    def _calculate_alligator(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Alligator (placeholder)."""
        return {
            'jaw': data['close'].rolling(window=13).mean(),
            'teeth': data['close'].rolling(window=8).mean(),
            'lips': data['close'].rolling(window=5).mean()
        }

    def _calculate_hurst_exponent(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Hurst Exponent (placeholder)."""
        return pd.Series(0.5, index=data.index)

    def _calculate_fisher_transform(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Fisher Transform (placeholder)."""
        return data['close'].pct_change()

    def _calculate_time_weighted_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Time-Weighted Volatility (placeholder)."""
        return self._calculate_volatility(data)

    def _calculate_pca_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate PCA Features (placeholder)."""
        return {'pca_1': data['close'], 'pca_2': data['volume']}

    def _calculate_autoencoder_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Autoencoder Features (placeholder)."""
        return {'ae_1': data['close'], 'ae_2': data['volume']}

    async def update_indicators_realtime(
        self,
        new_data: pd.DataFrame,
        existing_result: IndicatorResult
    ) -> IndicatorResult:
        """Update indicators with new real-time data."""
        # For now, recompute all indicators
        # In production, this would be optimized for incremental updates
        return await self.compute_indicators(new_data)

    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        feature_names = []

        for category, calculators in self.calculators.items():
            for name in calculators.keys():
                if name in ['macd', 'stochastic', 'aroon', 'bollinger_bands', 'keltner_channels', 'donchian_channels']:
                    # Multi-output indicators
                    if name == 'macd':
                        feature_names.extend([f"{name}_line", f"{name}_signal", f"{name}_histogram"])
                    elif name == 'stochastic':
                        feature_names.extend([f"{name}_k", f"{name}_d"])
                    elif name == 'aroon':
                        feature_names.extend([f"{name}_up", f"{name}_down", f"{name}_oscillator"])
                    elif name == 'bollinger_bands':
                        feature_names.extend([f"{name}_upper", f"{name}_middle", f"{name}_lower", f"{name}_width", f"{name}_position"])
                    elif name in ['keltner_channels', 'donchian_channels']:
                        feature_names.extend([f"{name}_upper", f"{name}_middle", f"{name}_lower"])
                else:
                    feature_names.append(name)

        return feature_names
