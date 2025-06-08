#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autocorrelation Indicator - Advanced Statistical Trading Engine
Platform3 Phase 3 - Enhanced Statistical Analysis

Autocorrelation measures the correlation of a time series with itself at different time lags.
It's used for:
- Identifying cyclical patterns and periodicities
- Market momentum analysis
- Trend persistence detection
- Mean reversion identification
- Time series forecasting
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
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class AutocorrelationIndicator:
    """
    Advanced Autocorrelation Indicator with Multiple Analysis Methods
    
    Features:
    - Lag-based autocorrelation analysis
    - Partial autocorrelation function (PACF)
    - Dominant cycle detection
    - Trend persistence measurement
    - Market rhythm analysis
    """    
    def __init__(self, max_lags: int = 50, min_periods: int = 100):
        """Initialize Autocorrelation indicator with Platform3 framework"""
        self.logger = Platform3Logger(self.__class__.__name__)
        self.error_system = Platform3ErrorSystem()
        self.db_manager = Platform3DatabaseManager()
        self.comm_framework = Platform3CommunicationFramework()
        
        self.max_lags = max(1, max_lags)
        self.min_periods = max(self.max_lags * 2, min_periods)
        
        self.logger.info(f"Autocorrelation Indicator initialized - Max Lags: {self.max_lags}, Min Periods: {self.min_periods}")
        
    async def calculate(self, data: Union[np.ndarray, pd.Series, List[float]], 
                       method: str = 'full') -> Optional[Dict[str, Any]]:
        """
        Calculate autocorrelation analysis with multiple methods
        
        Args:
            data: Price or indicator data
            method: 'full' (ACF+PACF+cycles), 'acf' (autocorrelation only), 'pacf' (partial autocorrelation)
            
        Returns:
            Dictionary containing autocorrelation analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.debug("Starting autocorrelation calculation")
            
            # Input validation and conversion
            data_array = self._validate_and_convert_data(data)
            if data_array is None:
                raise ServiceError("Invalid input data for autocorrelation calculation", "INVALID_INPUT")
            
            if len(data_array) < self.min_periods:
                raise ServiceError(f"Insufficient data: need {self.min_periods}, got {len(data_array)}", "INSUFFICIENT_DATA")
            
            # Calculate based on method
            if method == 'full':
                result = await self._calculate_full_analysis(data_array)
            elif method == 'acf':
                result = await self._calculate_acf_only(data_array)
            elif method == 'pacf':
                result = await self._calculate_pacf_only(data_array)
            else:
                raise ServiceError(f"Unknown method: {method}", "INVALID_METHOD")
            
            # Add metadata
            execution_time = time.time() - start_time
            result.update({
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'data_length': len(data_array)
            })
            
            self.logger.info(f"Autocorrelation calculation completed in {execution_time:.4f}s")
            return result
            
        except ServiceError as e:
            self.logger.error(f"Service error in autocorrelation calculation: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in autocorrelation calculation: {e}")
            self.error_system.handle_error(e, self.__class__.__name__)
            return None    
    async def _calculate_full_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate complete autocorrelation analysis"""
        
        # Prepare data (remove trend if necessary)
        detrended_data = self._detrend_data(data)
        
        # Calculate ACF
        acf_values, acf_confidence = self._calculate_autocorrelation_function(detrended_data)
        
        # Calculate PACF
        pacf_values, pacf_confidence = self._calculate_partial_autocorrelation(detrended_data)
        
        # Detect dominant cycles
        cycles = await self._detect_dominant_cycles(detrended_data, acf_values)
        
        # Analyze trend persistence
        persistence = self._analyze_trend_persistence(acf_values)
        
        # Market rhythm analysis
        rhythm = self._analyze_market_rhythm(acf_values, pacf_values)
        
        return {
            'acf': {
                'values': acf_values.tolist(),
                'confidence_intervals': acf_confidence.tolist(),
                'lags': list(range(len(acf_values)))
            },
            'pacf': {
                'values': pacf_values.tolist(),
                'confidence_intervals': pacf_confidence.tolist(),
                'lags': list(range(len(pacf_values)))
            },
            'cycles': cycles,
            'persistence': persistence,
            'rhythm': rhythm,
            'original_data_length': len(data),
            'analysis_type': 'full'
        }
    
    def _calculate_autocorrelation_function(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Autocorrelation Function (ACF)"""
        n = len(data)
        max_lags = min(self.max_lags, n // 4)  # Ensure sufficient data for each lag
        
        # Center the data
        data_centered = data - np.mean(data)
        
        # Calculate ACF using numpy correlate
        autocorr = np.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize by variance (lag 0)
        autocorr = autocorr / autocorr[0]
        
        # Keep only requested lags
        acf_values = autocorr[:max_lags + 1]
        
        # Calculate confidence intervals (95%)
        # For large samples, ACF ~ N(0, 1/n) under null hypothesis
        confidence_interval = 1.96 / np.sqrt(n)
        confidence_intervals = np.full(len(acf_values), confidence_interval)
        
        return acf_values, confidence_intervals    
    def _calculate_partial_autocorrelation(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Partial Autocorrelation Function (PACF) using Yule-Walker equations"""
        n = len(data)
        max_lags = min(self.max_lags, n // 4)
        
        # Center the data
        data_centered = data - np.mean(data)
        
        # Calculate ACF first
        acf_full, _ = self._calculate_autocorrelation_function(data_centered)
        
        pacf_values = np.zeros(max_lags + 1)
        pacf_values[0] = 1.0  # PACF at lag 0 is always 1
        
        if max_lags > 0:
            pacf_values[1] = acf_full[1]  # PACF at lag 1 equals ACF at lag 1
        
        # Calculate PACF using Durbin-Levinson algorithm
        for k in range(2, max_lags + 1):
            if k < len(acf_full):
                # Build the Toeplitz matrix
                gamma_matrix = np.array([[acf_full[abs(i-j)] for j in range(k)] for i in range(k)])
                gamma_vector = acf_full[1:k+1]
                
                try:
                    # Solve the Yule-Walker equations
                    phi = np.linalg.solve(gamma_matrix, gamma_vector)
                    pacf_values[k] = phi[-1]
                except np.linalg.LinAlgError:
                    pacf_values[k] = 0.0
            else:
                pacf_values[k] = 0.0
        
        # Calculate confidence intervals for PACF
        confidence_interval = 1.96 / np.sqrt(n)
        confidence_intervals = np.full(len(pacf_values), confidence_interval)
        
        return pacf_values, confidence_intervals
    
    async def _detect_dominant_cycles(self, data: np.ndarray, acf_values: np.ndarray) -> Dict[str, Any]:
        """Detect dominant cycles using spectral analysis and ACF peaks"""
        try:
            # Find peaks in ACF (excluding lag 0)
            acf_peaks_indices = []
            acf_peaks_values = []
            
            for i in range(2, len(acf_values)):
                if (acf_values[i] > acf_values[i-1] and 
                    acf_values[i] > acf_values[i+1] if i+1 < len(acf_values) else True):
                    if acf_values[i] > 0.1:  # Significant threshold
                        acf_peaks_indices.append(i)
                        acf_peaks_values.append(float(acf_values[i]))
            
            # FFT-based spectral analysis for cycle detection
            fft_result = fft(data - np.mean(data))
            frequencies = fftfreq(len(data))
            power_spectrum = np.abs(fft_result) ** 2
            
            # Find dominant frequencies (positive frequencies only)
            half_len = len(frequencies) // 2
            pos_frequencies = frequencies[1:half_len]
            pos_power = power_spectrum[1:half_len]
            
            # Find peaks in power spectrum
            if len(pos_power) > 10:
                spectral_peaks = signal.find_peaks(pos_power, height=np.mean(pos_power))[0]
                dominant_periods = []
                
                for peak_idx in spectral_peaks[:5]:  # Top 5 peaks
                    if pos_frequencies[peak_idx] > 0:
                        period = 1.0 / pos_frequencies[peak_idx]
                        if 2 <= period <= len(data) // 4:  # Reasonable period range
                            dominant_periods.append({
                                'period': float(period),
                                'frequency': float(pos_frequencies[peak_idx]),
                                'power': float(pos_power[peak_idx])
                            })
                
                # Sort by power
                dominant_periods.sort(key=lambda x: x['power'], reverse=True)
            else:
                dominant_periods = []
            
            return {
                'acf_peaks': {
                    'lags': acf_peaks_indices,
                    'values': acf_peaks_values
                },
                'spectral_peaks': dominant_periods[:3],  # Top 3 cycles
                'dominant_cycle': dominant_periods[0] if dominant_periods else None
            }
            
        except Exception as e:
            self.logger.error(f"Cycle detection error: {e}")
            return {'acf_peaks': {'lags': [], 'values': []}, 'spectral_peaks': [], 'dominant_cycle': None}    
    def _analyze_trend_persistence(self, acf_values: np.ndarray) -> Dict[str, Any]:
        """Analyze trend persistence and mean reversion characteristics"""
        
        # Calculate persistence metrics
        lag1_autocorr = float(acf_values[1]) if len(acf_values) > 1 else 0.0
        
        # Persistence classification
        if lag1_autocorr > 0.7:
            persistence_type = "strong_persistence"
        elif lag1_autocorr > 0.3:
            persistence_type = "moderate_persistence"
        elif lag1_autocorr > -0.3:
            persistence_type = "neutral"
        elif lag1_autocorr > -0.7:
            persistence_type = "moderate_mean_reversion"
        else:
            persistence_type = "strong_mean_reversion"
        
        # Calculate decay rate (how fast ACF decays)
        decay_rate = 0.0
        significant_lags = 0
        
        for i in range(1, min(len(acf_values), 20)):
            if abs(acf_values[i]) > 0.1:  # Significant threshold
                significant_lags = i
            if i > 1 and acf_values[i] != 0:
                decay_rate += abs((acf_values[i] - acf_values[i-1]) / acf_values[i-1])
        
        if significant_lags > 0:
            decay_rate /= significant_lags
        
        return {
            'lag1_autocorrelation': lag1_autocorr,
            'persistence_type': persistence_type,
            'persistence_strength': abs(lag1_autocorr),
            'decay_rate': float(decay_rate),
            'significant_lags': int(significant_lags),
            'interpretation': self._interpret_persistence(lag1_autocorr, decay_rate)
        }
    
    def _analyze_market_rhythm(self, acf_values: np.ndarray, pacf_values: np.ndarray) -> Dict[str, Any]:
        """Analyze market rhythm and timing patterns"""
        
        # Find rhythmic patterns in ACF
        positive_peaks = []
        negative_peaks = []
        
        for i in range(1, len(acf_values)):
            if i == 1 or (acf_values[i] > acf_values[i-1] and 
                         (i+1 >= len(acf_values) or acf_values[i] > acf_values[i+1])):
                if acf_values[i] > 0.1:
                    positive_peaks.append(i)
                elif acf_values[i] < -0.1:
                    negative_peaks.append(i)
        
        # Calculate rhythm metrics
        rhythm_strength = 0.0
        if len(positive_peaks) > 1:
            intervals = np.diff(positive_peaks)
            rhythm_strength = 1.0 / (1.0 + np.std(intervals)) if np.std(intervals) > 0 else 1.0
        
        # Identify market timing signals
        timing_signals = []
        if len(pacf_values) > 5:
            for i in range(1, min(6, len(pacf_values))):
                if abs(pacf_values[i]) > 0.2:  # Significant PACF
                    signal_type = "buy_timing" if pacf_values[i] > 0 else "sell_timing"
                    timing_signals.append({
                        'lag': i,
                        'value': float(pacf_values[i]),
                        'type': signal_type,
                        'strength': abs(pacf_values[i])
                    })
        
        return {
            'rhythm_strength': float(rhythm_strength),
            'positive_peaks': positive_peaks,
            'negative_peaks': negative_peaks,
            'timing_signals': timing_signals,
            'market_state': self._determine_market_state(acf_values, pacf_values)
        }    
    def _interpret_persistence(self, lag1_autocorr: float, decay_rate: float) -> str:
        """Interpret persistence analysis for trading insights"""
        if lag1_autocorr > 0.5:
            if decay_rate < 0.1:
                return "Strong trending market - momentum strategies favored"
            else:
                return "Trending market with periodic corrections"
        elif lag1_autocorr < -0.3:
            return "Mean reverting market - contrarian strategies favored"
        else:
            if decay_rate > 0.2:
                return "Choppy market - range-bound trading strategies"
            else:
                return "Balanced market - mixed strategy approach"
    
    def _determine_market_state(self, acf_values: np.ndarray, pacf_values: np.ndarray) -> str:
        """Determine overall market state from autocorrelation patterns"""
        lag1_acf = acf_values[1] if len(acf_values) > 1 else 0
        lag1_pacf = pacf_values[1] if len(pacf_values) > 1 else 0
        
        # Analyze pattern
        if lag1_acf > 0.3 and lag1_pacf > 0.3:
            return "trending_up"
        elif lag1_acf > 0.3 and lag1_pacf < -0.3:
            return "trending_down"
        elif abs(lag1_acf) < 0.2 and abs(lag1_pacf) < 0.2:
            return "random_walk"
        elif lag1_acf < -0.2:
            return "mean_reverting"
        else:
            return "mixed_signals"
    
    def _detrend_data(self, data: np.ndarray) -> np.ndarray:
        """Remove linear trend from data for better autocorrelation analysis"""
        x = np.arange(len(data))
        
        # Fit linear trend
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        
        # Return detrended data
        return data - trend
    
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
    
    async def _calculate_acf_only(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate only ACF analysis"""
        acf_values, acf_confidence = self._calculate_autocorrelation_function(data)
        
        return {
            'acf': {
                'values': acf_values.tolist(),
                'confidence_intervals': acf_confidence.tolist(),
                'lags': list(range(len(acf_values)))
            },
            'analysis_type': 'acf_only'
        }
    
    async def _calculate_pacf_only(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate only PACF analysis"""
        pacf_values, pacf_confidence = self._calculate_partial_autocorrelation(data)
        
        return {
            'pacf': {
                'values': pacf_values.tolist(),
                'confidence_intervals': pacf_confidence.tolist(),
                'lags': list(range(len(pacf_values)))
            },
            'analysis_type': 'pacf_only'
        }    
    def get_parameters(self) -> Dict[str, Any]:
        """Get Autocorrelation indicator parameters"""
        return {
            'indicator_name': 'Autocorrelation',
            'version': '1.0.0',
            'max_lags': self.max_lags,
            'min_periods': self.min_periods,
            'methods': ['full', 'acf', 'pacf'],
            'features': [
                'Autocorrelation Function (ACF)',
                'Partial Autocorrelation Function (PACF)',
                'Dominant cycle detection',
                'Trend persistence analysis',
                'Market rhythm analysis',
                'Spectral analysis integration'
            ]
        }

# Export for Platform3 integration
__all__ = ['AutocorrelationIndicator']