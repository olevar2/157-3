#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Z-Score Indicator - Advanced Statistical Trading Engine
Platform3 Phase 3 - Enhanced Statistical Analysis

The Z-Score measures how many standard deviations an observation is from the mean.
It's used for:
- Identifying overbought/oversold conditions
- Mean reversion trading strategies
- Statistical arbitrage
- Outlier detection
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
import time
from scipy import stats
from numba import jit, njit
import warnings
warnings.filterwarnings('ignore')

class ZScoreIndicator:
    """
    Advanced Z-Score Indicator with Multiple Statistical Methods
    
    Features:
    - Rolling Z-Score calculation
    - Modified Z-Score (using median)
    - Multi-timeframe analysis
    - Dynamic threshold adjustment
    - Outlier detection capabilities
    """
    
    def __init__(self, period: int = 20, threshold: float = 2.0):
        """Initialize Z-Score indicator with Platform3 framework"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        self.period = max(2, period)
        self.threshold = abs(threshold)
        
        self.logger.info(f"Z-Score Indicator initialized - Period: {self.period}, Threshold: {self.threshold}")        
    async def calculate(self, data: Union[np.ndarray, pd.Series, List[float]], 
                       use_modified: bool = False) -> Optional[Dict[str, Any]]:
        """
        Calculate Z-Score with multiple statistical methods
        
        Args:
            data: Price or indicator data
            use_modified: Use modified Z-Score (median-based) for outlier detection
            
        Returns:
            Dictionary containing Z-Score analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.debug("Starting Z-Score calculation")
            
            # Input validation and conversion
            data_array = self._validate_and_convert_data(data)
            if data_array is None:
                raise ServiceError("Invalid input data for Z-Score calculation", "INVALID_INPUT")
            
            if len(data_array) < self.period:
                raise ServiceError(f"Insufficient data: need {self.period}, got {len(data_array)}", "INSUFFICIENT_DATA")
            
            # Calculate Z-Score using appropriate method
            if use_modified:
                result = await self._calculate_modified_zscore(data_array)
            else:
                result = await self._calculate_standard_zscore(data_array)
            
            # Add additional analysis
            result.update(await self._perform_statistical_analysis(data_array, result['zscore']))
            
            execution_time = time.time() - start_time
            result['execution_time'] = execution_time
            result['timestamp'] = datetime.now().isoformat()
            
            self.logger.info(f"Z-Score calculation completed in {execution_time:.4f}s")
            return result
            
        except ServiceError as e:
            self.logger.error(f"Service error in Z-Score calculation: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in Z-Score calculation: {e}")
            self.error_system.handle_error(e, self.__class__.__name__)
            return None    
    async def _calculate_standard_zscore(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate standard Z-Score using rolling mean and standard deviation"""
        
        zscore_values = np.full(len(data), np.nan)
        rolling_mean = np.full(len(data), np.nan)
        rolling_std = np.full(len(data), np.nan)
        
        # Use optimized rolling calculation
        for i in range(self.period - 1, len(data)):
            window_data = data[i - self.period + 1:i + 1]
            
            mean_val = np.mean(window_data)
            std_val = np.std(window_data, ddof=1)
            
            rolling_mean[i] = mean_val
            rolling_std[i] = std_val
            
            if std_val > 1e-8:  # Avoid division by zero
                zscore_values[i] = (data[i] - mean_val) / std_val
            else:
                zscore_values[i] = 0.0
        
        return {
            'zscore': zscore_values,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'method': 'standard',
            'period': self.period
        }
    
    async def _calculate_modified_zscore(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate Modified Z-Score using median and MAD (Median Absolute Deviation)"""
        
        zscore_values = np.full(len(data), np.nan)
        rolling_median = np.full(len(data), np.nan)
        rolling_mad = np.full(len(data), np.nan)
        
        for i in range(self.period - 1, len(data)):
            window_data = data[i - self.period + 1:i + 1]
            
            median_val = np.median(window_data)
            mad_val = np.median(np.abs(window_data - median_val))
            
            rolling_median[i] = median_val
            rolling_mad[i] = mad_val
            
            if mad_val > 1e-8:
                # Modified Z-Score formula: 0.6745 * (x - median) / MAD
                zscore_values[i] = 0.6745 * (data[i] - median_val) / mad_val
            else:
                zscore_values[i] = 0.0
        
        return {
            'zscore': zscore_values,
            'rolling_median': rolling_median,
            'rolling_mad': rolling_mad,
            'method': 'modified',
            'period': self.period
        }    
    async def _perform_statistical_analysis(self, data: np.ndarray, zscore: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on Z-Score results"""
        
        # Remove NaN values for statistics
        valid_zscore = zscore[~np.isnan(zscore)]
        valid_data = data[~np.isnan(zscore)]
        
        if len(valid_zscore) == 0:
            return {'analysis': 'insufficient_data'}
        
        # Statistical metrics
        analysis = {
            'mean_zscore': float(np.mean(valid_zscore)),
            'std_zscore': float(np.std(valid_zscore)),
            'min_zscore': float(np.min(valid_zscore)),
            'max_zscore': float(np.max(valid_zscore)),
            'current_zscore': float(zscore[-1]) if not np.isnan(zscore[-1]) else None,
        }
        
        # Threshold analysis
        extreme_high = np.sum(valid_zscore > self.threshold)
        extreme_low = np.sum(valid_zscore < -self.threshold)
        total_observations = len(valid_zscore)
        
        analysis.update({
            'extreme_high_count': int(extreme_high),
            'extreme_low_count': int(extreme_low),
            'extreme_high_pct': float(extreme_high / total_observations * 100),
            'extreme_low_pct': float(extreme_low / total_observations * 100),
            'threshold': self.threshold
        })
        
        # Current signal analysis
        current_zscore = analysis['current_zscore']
        if current_zscore is not None:
            if current_zscore > self.threshold:
                signal = 'overbought'
                signal_strength = min(abs(current_zscore) / self.threshold, 3.0)
            elif current_zscore < -self.threshold:
                signal = 'oversold'
                signal_strength = min(abs(current_zscore) / self.threshold, 3.0)
            else:
                signal = 'neutral'
                signal_strength = abs(current_zscore) / self.threshold
            
            analysis.update({
                'signal': signal,
                'signal_strength': float(signal_strength),
                'is_extreme': abs(current_zscore) > self.threshold
            })
        
        return analysis    
    def _validate_and_convert_data(self, data: Union[np.ndarray, pd.Series, List[float]]) -> Optional[np.ndarray]:
        """Validate and convert input data to numpy array"""
        try:
            if isinstance(data, pd.Series):
                return data.values
            elif isinstance(data, list):
                return np.array(data, dtype=np.float64)
            elif isinstance(data, np.ndarray):
                return data.astype(np.float64)
            else:
                self.logger.error(f"Unsupported data type: {type(data)}")
                return None
        except Exception as e:
            self.logger.error(f"Data conversion error: {e}")
            return None
    
    async def calculate_multi_timeframe(self, data: Union[np.ndarray, pd.Series], 
                                      periods: List[int] = [10, 20, 50]) -> Optional[Dict[str, Any]]:
        """Calculate Z-Score across multiple timeframes"""
        try:
            results = {}
            data_array = self._validate_and_convert_data(data)
            
            if data_array is None:
                return None
            
            for period in periods:
                original_period = self.period
                self.period = period
                
                result = await self.calculate(data_array)
                if result:
                    results[f'period_{period}'] = {
                        'zscore': result['zscore'],
                        'current_zscore': result['analysis']['current_zscore'],
                        'signal': result['analysis'].get('signal', 'neutral'),
                        'signal_strength': result['analysis'].get('signal_strength', 0.0)
                    }
                
                self.period = original_period
            
            # Consensus analysis
            current_signals = [results[f'period_{p}']['signal'] for p in periods 
                             if f'period_{p}' in results and results[f'period_{p}']['current_zscore'] is not None]
            
            if current_signals:
                signal_counts = {'overbought': 0, 'oversold': 0, 'neutral': 0}
                for signal in current_signals:
                    signal_counts[signal] += 1
                
                consensus = max(signal_counts.items(), key=lambda x: x[1])
                results['consensus'] = {
                    'signal': consensus[0],
                    'agreement_pct': consensus[1] / len(current_signals) * 100,
                    'total_timeframes': len(current_signals)
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe Z-Score calculation error: {e}")
            return None    
    async def detect_outliers(self, data: Union[np.ndarray, pd.Series], 
                            method: str = 'modified', threshold: float = 3.5) -> Optional[Dict[str, Any]]:
        """
        Advanced outlier detection using Z-Score methods
        
        Args:
            data: Input data for outlier detection
            method: 'standard' or 'modified' Z-Score
            threshold: Z-Score threshold for outlier detection
            
        Returns:
            Dictionary containing outlier analysis
        """
        try:
            data_array = self._validate_and_convert_data(data)
            if data_array is None:
                return None
            
            original_threshold = self.threshold
            self.threshold = threshold
            
            # Calculate Z-Score
            result = await self.calculate(data_array, use_modified=(method == 'modified'))
            if not result:
                return None
            
            zscore = result['zscore']
            valid_mask = ~np.isnan(zscore)
            
            # Identify outliers
            outlier_mask = np.abs(zscore) > threshold
            outlier_indices = np.where(outlier_mask)[0].tolist()
            outlier_values = data_array[outlier_mask].tolist()
            outlier_zscores = zscore[outlier_mask].tolist()
            
            # Statistical summary
            outlier_analysis = {
                'method': method,
                'threshold': threshold,
                'total_points': len(data_array),
                'valid_points': int(np.sum(valid_mask)),
                'outlier_count': len(outlier_indices),
                'outlier_percentage': len(outlier_indices) / np.sum(valid_mask) * 100 if np.sum(valid_mask) > 0 else 0,
                'outlier_indices': outlier_indices,
                'outlier_values': outlier_values,
                'outlier_zscores': outlier_zscores,
                'max_abs_zscore': float(np.max(np.abs(zscore[valid_mask]))) if np.sum(valid_mask) > 0 else 0
            }
            
            # Restore original threshold
            self.threshold = original_threshold
            
            return outlier_analysis
            
        except Exception as e:
            self.logger.error(f"Outlier detection error: {e}")
            self.threshold = original_threshold
            return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get Z-Score indicator parameters"""
        return {
            'indicator_name': 'Z-Score',
            'version': '1.0.0',
            'period': self.period,
            'threshold': self.threshold,
            'methods': ['standard', 'modified'],
            'features': [
                'Rolling Z-Score calculation',
                'Modified Z-Score (median-based)',
                'Multi-timeframe analysis',
                'Outlier detection',
                'Signal generation'
            ]
        }

# Export for Platform3 integration
__all__ = ['ZScoreIndicator']