"""
Phase Analysis - Advanced Cycle Phase Position and Timing Analysis
=================================================================

This module provides comprehensive phase analysis for market cycles including:
- Current phase position identification
- Phase velocity and acceleration
- Phase coherence and stability measurement
- Cycle timing and turning point prediction
- Multi-cycle phase relationships
- Phase-based signal generation

Key Features:
- Hilbert Transform phase analysis
- Instantaneous phase tracking
- Phase coherence measurement
- Turning point prediction
- Multi-timeframe phase alignment
- Phase momentum analysis

Author: Platform3 Development Team
Date: May 29, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, circmean, circstd
import warnings

class PhaseAnalysis:
    """
    Advanced phase analysis system for market cycle timing.
    
    This class provides comprehensive phase analysis including:
    - Current phase position and trajectory
    - Phase velocity and acceleration
    - Coherence and stability measurement
    - Turning point prediction
    - Signal generation based on phase
    """
    
    def __init__(self, 
                 min_cycle_length: int = 8,
                 max_cycle_length: int = 120,
                 phase_smoothing: int = 5,
                 coherence_window: int = 20,
                 prediction_horizon: int = 5):
        """
        Initialize the Phase Analysis system.
        
        Parameters:
        -----------
        min_cycle_length : int
            Minimum cycle length for analysis (default: 8 periods)
        max_cycle_length : int
            Maximum cycle length for analysis (default: 120 periods)
        phase_smoothing : int
            Smoothing window for phase calculations (default: 5)
        coherence_window : int
            Window size for coherence analysis (default: 20)
        prediction_horizon : int
            Number of periods ahead to predict (default: 5)
        """
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.phase_smoothing = phase_smoothing
        self.coherence_window = coherence_window
        self.prediction_horizon = prediction_horizon
        
        # Phase tracking history
        self.phase_history = []
        self.velocity_history = []
        self.coherence_history = []
        
    def calculate(self, 
                  data: Union[pd.Series, np.ndarray],
                  cycle_period: Optional[float] = None,
                  price_column: str = 'close') -> Dict[str, Union[float, List, Dict]]:
        """
        Perform comprehensive phase analysis.
        
        Parameters:
        -----------
        data : pd.Series or np.ndarray
            Price data for phase analysis
        cycle_period : float, optional
            Known cycle period (if None, will be estimated)
        price_column : str
            Column name if data is DataFrame (default: 'close')
            
        Returns:
        --------
        Dict containing:
            - current_phase: Current phase position (0-2π)
            - phase_degrees: Phase in degrees (0-360)
            - phase_velocity: Rate of phase change
            - phase_acceleration: Phase acceleration
            - coherence: Phase coherence measure (0-1)
            - phase_stability: Phase stability score (0-1)
            - turning_points: Predicted turning points
            - cycle_position: Position within cycle (0-1)
            - phase_momentum: Phase momentum indicator
            - signal_strength: Trading signal strength
        """
        try:
            # Data validation and preprocessing
            if isinstance(data, pd.DataFrame):
                prices = data[price_column].values
            else:
                prices = np.array(data)
                
            if len(prices) < self.max_cycle_length:
                raise ValueError(f"Insufficient data: need at least {self.max_cycle_length} periods")
                
            # Remove NaN values and detrend
            prices = prices[~np.isnan(prices)]
            detrended_prices = self._detrend_data(prices)
            
            # Estimate cycle period if not provided
            if cycle_period is None:
                cycle_period = self._estimate_cycle_period(detrended_prices)
                if cycle_period is None:
                    return self._return_empty_results("Unable to estimate cycle period")
            
            # Core phase analysis
            phase_info = self._calculate_instantaneous_phase(detrended_prices, cycle_period)
            velocity_info = self._calculate_phase_velocity(phase_info['phase_series'])
            coherence_info = self._calculate_phase_coherence(phase_info['phase_series'], cycle_period)
            
            # Advanced analysis
            turning_points = self._predict_turning_points(phase_info, velocity_info, cycle_period)
            cycle_position = self._calculate_cycle_position(phase_info['current_phase'])
            phase_momentum = self._calculate_phase_momentum(phase_info, velocity_info)
            signal_analysis = self._generate_phase_signals(phase_info, coherence_info, turning_points)
            
            # Multi-timeframe analysis
            multi_tf_analysis = self._multi_timeframe_phase_analysis(detrended_prices, cycle_period)
            
            # Update history
            self._update_phase_history(phase_info['current_phase'], 
                                     velocity_info['current_velocity'],
                                     coherence_info['coherence'])
            
            return {
                'current_phase': phase_info['current_phase'],
                'phase_degrees': np.degrees(phase_info['current_phase']),
                'phase_velocity': velocity_info['current_velocity'],
                'phase_acceleration': velocity_info['acceleration'],
                'coherence': coherence_info['coherence'],
                'phase_stability': coherence_info['stability'],
                'turning_points': turning_points,
                'cycle_position': cycle_position,
                'phase_momentum': phase_momentum,
                'signal_strength': signal_analysis['signal_strength'],
                'phase_signals': signal_analysis['signals'],
                'multi_timeframe': multi_tf_analysis,
                'phase_quality': coherence_info['quality'],
                'cycle_period_used': cycle_period
            }
            
        except Exception as e:
            return self._return_empty_results(f"Error: {str(e)}")
    
    def _detrend_data(self, prices: np.ndarray) -> np.ndarray:
        """Detrend data for phase analysis."""
        try:
            # Linear detrending
            x = np.arange(len(prices))
            coeffs = np.polyfit(x, prices, 1)
            trend = np.polyval(coeffs, x)
            detrended = prices - trend
            
            # Optional high-pass filtering
            if len(prices) > 2 * self.max_cycle_length:
                try:
                    # Butterworth high-pass filter
                    cutoff_freq = 1.0 / self.max_cycle_length
                    b, a = signal.butter(2, cutoff_freq, btype='high', fs=1.0)
                    filtered = signal.filtfilt(b, a, detrended)
                    return filtered
                except Exception:
                    pass
            
            return detrended
            
        except Exception:
            return np.diff(prices, prepend=prices[0])
    
    def _estimate_cycle_period(self, data: np.ndarray) -> Optional[float]:
        """Estimate dominant cycle period using spectral analysis."""
        try:
            # FFT-based estimation
            n = len(data)
            fft_vals = np.fft.fft(data * np.hanning(n))
            freqs = np.fft.fftfreq(n)
            power = np.abs(fft_vals) ** 2
            
            # Focus on valid frequency range
            min_freq = 1.0 / self.max_cycle_length
            max_freq = 1.0 / self.min_cycle_length
            
            valid_mask = (freqs > min_freq) & (freqs < max_freq) & (freqs > 0)
            
            if not np.any(valid_mask):
                return None
            
            valid_freqs = freqs[valid_mask]
            valid_power = power[valid_mask]
            
            # Find dominant frequency
            max_power_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[max_power_idx]
            
            return 1.0 / dominant_freq
            
        except Exception:
            return None
    
    def _calculate_instantaneous_phase(self, data: np.ndarray, cycle_period: float) -> Dict:
        """Calculate instantaneous phase using Hilbert Transform."""
        try:
            # Hilbert transform for analytic signal
            analytic_signal = signal.hilbert(data)
            instantaneous_phase = np.angle(analytic_signal)
            
            # Unwrap phase for continuous tracking
            unwrapped_phase = np.unwrap(instantaneous_phase)
            
            # Smooth phase if requested
            if self.phase_smoothing > 1:
                smoothed_phase = np.convolve(
                    unwrapped_phase, 
                    np.ones(self.phase_smoothing) / self.phase_smoothing, 
                    mode='same'
                )
            else:
                smoothed_phase = unwrapped_phase
            
            # Current phase (wrapped to 0-2π)
            current_phase = smoothed_phase[-1] % (2 * np.pi)
            
            # Phase series for further analysis
            phase_series = smoothed_phase % (2 * np.pi)
            
            # Amplitude envelope
            amplitude_envelope = np.abs(analytic_signal)
            current_amplitude = amplitude_envelope[-1]
            
            return {
                'current_phase': current_phase,
                'phase_series': phase_series,
                'unwrapped_phase': unwrapped_phase,
                'amplitude_envelope': amplitude_envelope,
                'current_amplitude': current_amplitude
            }
            
        except Exception as e:
            return {
                'current_phase': 0.0,
                'phase_series': np.array([]),
                'unwrapped_phase': np.array([]),
                'amplitude_envelope': np.array([]),
                'current_amplitude': 0.0,
                'error': str(e)
            }
    
    def _calculate_phase_velocity(self, phase_series: np.ndarray) -> Dict:
        """Calculate phase velocity and acceleration."""
        try:
            if len(phase_series) < 3:
                return {
                    'current_velocity': 0.0,
                    'velocity_series': np.array([]),
                    'acceleration': 0.0
                }
            
            # Calculate instantaneous velocity (phase change rate)
            unwrapped_phases = np.unwrap(phase_series)
            velocity_series = np.diff(unwrapped_phases)
            
            # Smooth velocity
            if len(velocity_series) > self.phase_smoothing:
                smoothed_velocity = np.convolve(
                    velocity_series,
                    np.ones(self.phase_smoothing) / self.phase_smoothing,
                    mode='same'
                )
            else:
                smoothed_velocity = velocity_series
            
            current_velocity = smoothed_velocity[-1] if len(smoothed_velocity) > 0 else 0.0
            
            # Calculate acceleration (velocity change rate)
            if len(smoothed_velocity) > 1:
                acceleration_series = np.diff(smoothed_velocity)
                acceleration = acceleration_series[-1] if len(acceleration_series) > 0 else 0.0
            else:
                acceleration = 0.0
            
            return {
                'current_velocity': current_velocity,
                'velocity_series': smoothed_velocity,
                'acceleration': acceleration,
                'velocity_trend': acceleration  # Positive = accelerating, negative = decelerating
            }
            
        except Exception as e:
            return {
                'current_velocity': 0.0,
                'velocity_series': np.array([]),
                'acceleration': 0.0,
                'error': str(e)
            }
    
    def _calculate_phase_coherence(self, phase_series: np.ndarray, cycle_period: float) -> Dict:
        """Calculate phase coherence and stability measures."""
        try:
            if len(phase_series) < self.coherence_window:
                return {
                    'coherence': 0.0,
                    'stability': 0.0,
                    'quality': 0.0
                }
            
            # Recent phase data for coherence analysis
            recent_phases = phase_series[-self.coherence_window:]
            
            # Expected phase increment per period
            expected_increment = 2 * np.pi / cycle_period
            
            # Calculate phase differences
            phase_diffs = np.diff(np.unwrap(recent_phases))
            
            # Coherence based on consistency of phase increments
            phase_errors = np.abs(phase_diffs - expected_increment)
            mean_error = np.mean(phase_errors)
            coherence = max(0.0, 1.0 - (mean_error / np.pi))
            
            # Stability based on phase difference variance
            phase_diff_std = np.std(phase_diffs)
            stability = max(0.0, 1.0 - (phase_diff_std / (expected_increment + 1e-10)))
            
            # Overall quality combining coherence and stability
            quality = (coherence + stability) / 2.0
            
            # Circular statistics for phase analysis
            circular_mean = circmean(recent_phases)
            circular_std = circstd(recent_phases)
            circular_consistency = max(0.0, 1.0 - (circular_std / np.pi))
            
            return {
                'coherence': coherence,
                'stability': stability,
                'quality': quality,
                'circular_mean': circular_mean,
                'circular_consistency': circular_consistency,
                'phase_error': mean_error
            }
            
        except Exception as e:
            return {
                'coherence': 0.0,
                'stability': 0.0,
                'quality': 0.0,
                'error': str(e)
            }
    
    def _predict_turning_points(self, phase_info: Dict, velocity_info: Dict, cycle_period: float) -> Dict:
        """Predict upcoming cycle turning points."""
        try:
            current_phase = phase_info['current_phase']
            current_velocity = velocity_info['current_velocity']
            
            # Key turning point phases
            turning_phases = {
                'peak': np.pi / 2,      # 90 degrees
                'trough': 3 * np.pi / 2  # 270 degrees
            }
            
            predictions = {}
            
            for point_name, target_phase in turning_phases.items():
                # Calculate phase distance to turning point
                phase_distance = target_phase - current_phase
                
                # Handle phase wrapping
                if phase_distance < 0:
                    phase_distance += 2 * np.pi
                if phase_distance > np.pi:
                    phase_distance = phase_distance - 2 * np.pi
                
                # Estimate time to turning point based on current velocity
                if abs(current_velocity) > 1e-6:
                    time_to_turning = phase_distance / current_velocity
                    
                    # Validate prediction
                    if 0 < time_to_turning <= self.prediction_horizon * cycle_period:
                        confidence = min(1.0, velocity_info.get('velocity_series', [0])[-5:].std() + 0.1)
                        confidence = max(0.0, 1.0 - confidence)
                        
                        predictions[point_name] = {
                            'periods_ahead': time_to_turning,
                            'target_phase': target_phase,
                            'phase_distance': phase_distance,
                            'confidence': confidence
                        }
                    else:
                        predictions[point_name] = {
                            'periods_ahead': None,
                            'target_phase': target_phase,
                            'phase_distance': phase_distance,
                            'confidence': 0.0
                        }
                else:
                    predictions[point_name] = {
                        'periods_ahead': None,
                        'target_phase': target_phase,
                        'phase_distance': phase_distance,
                        'confidence': 0.0
                    }
            
            # Next significant turning point
            next_turning = None
            min_distance = float('inf')
            
            for point_name, pred in predictions.items():
                if pred['periods_ahead'] is not None and pred['periods_ahead'] < min_distance:
                    min_distance = pred['periods_ahead']
                    next_turning = {
                        'type': point_name,
                        'periods_ahead': pred['periods_ahead'],
                        'confidence': pred['confidence']
                    }
            
            return {
                'predictions': predictions,
                'next_turning_point': next_turning,
                'prediction_horizon': self.prediction_horizon
            }
            
        except Exception as e:
            return {
                'predictions': {},
                'next_turning_point': None,
                'error': str(e)
            }
    
    def _calculate_cycle_position(self, current_phase: float) -> Dict:
        """Calculate current position within the cycle."""
        try:
            # Normalize phase to 0-1 cycle position
            cycle_position = current_phase / (2 * np.pi)
            
            # Cycle phase labels
            if 0 <= cycle_position < 0.25:
                phase_name = "Early Uptrend"
                phase_description = "Rising from trough"
            elif 0.25 <= cycle_position < 0.5:
                phase_name = "Late Uptrend"
                phase_description = "Approaching peak"
            elif 0.5 <= cycle_position < 0.75:
                phase_name = "Early Downtrend"
                phase_description = "Declining from peak"
            else:
                phase_name = "Late Downtrend"
                phase_description = "Approaching trough"
            
            # Distance to key points
            distance_to_peak = abs(cycle_position - 0.25)
            distance_to_trough = abs(cycle_position - 0.75)
            
            return {
                'cycle_position': cycle_position,
                'phase_name': phase_name,
                'phase_description': phase_description,
                'distance_to_peak': distance_to_peak,
                'distance_to_trough': distance_to_trough,
                'cycle_progress': cycle_position
            }
            
        except Exception as e:
            return {
                'cycle_position': 0.0,
                'phase_name': "Unknown",
                'phase_description': "Cannot determine",
                'error': str(e)
            }
    
    def _calculate_phase_momentum(self, phase_info: Dict, velocity_info: Dict) -> Dict:
        """Calculate phase momentum indicators."""
        try:
            current_velocity = velocity_info['current_velocity']
            acceleration = velocity_info['acceleration']
            current_amplitude = phase_info['current_amplitude']
            
            # Phase momentum combining velocity and amplitude
            momentum = current_velocity * current_amplitude
            
            # Momentum classification
            if momentum > 0.1:
                momentum_state = "Strong Positive"
            elif momentum > 0.05:
                momentum_state = "Moderate Positive"
            elif momentum > -0.05:
                momentum_state = "Neutral"
            elif momentum > -0.1:
                momentum_state = "Moderate Negative"
            else:
                momentum_state = "Strong Negative"
            
            # Acceleration classification
            if acceleration > 0.01:
                acceleration_state = "Accelerating"
            elif acceleration < -0.01:
                acceleration_state = "Decelerating"
            else:
                acceleration_state = "Constant"
            
            return {
                'momentum': momentum,
                'momentum_state': momentum_state,
                'acceleration_state': acceleration_state,
                'velocity_component': current_velocity,
                'amplitude_component': current_amplitude
            }
            
        except Exception as e:
            return {
                'momentum': 0.0,
                'momentum_state': "Unknown",
                'acceleration_state': "Unknown",
                'error': str(e)
            }
    
    def _generate_phase_signals(self, phase_info: Dict, coherence_info: Dict, turning_points: Dict) -> Dict:
        """Generate trading signals based on phase analysis."""
        try:
            current_phase = phase_info['current_phase']
            coherence = coherence_info['coherence']
            quality = coherence_info['quality']
            
            signals = []
            signal_strength = 0.0
            
            # Phase-based signals
            cycle_position = current_phase / (2 * np.pi)
            
            # Buy signal near trough (phase around 270° or 3π/2)
            trough_distance = abs(current_phase - (3 * np.pi / 2))
            if trough_distance < np.pi / 4:  # Within 45° of trough
                buy_strength = (1 - trough_distance / (np.pi / 4)) * coherence
                if buy_strength > 0.5:
                    signals.append({
                        'type': 'buy',
                        'strength': buy_strength,
                        'reason': 'Near cycle trough'
                    })
                    signal_strength += buy_strength
            
            # Sell signal near peak (phase around 90° or π/2)
            peak_distance = abs(current_phase - (np.pi / 2))
            if peak_distance < np.pi / 4:  # Within 45° of peak
                sell_strength = (1 - peak_distance / (np.pi / 4)) * coherence
                if sell_strength > 0.5:
                    signals.append({
                        'type': 'sell',
                        'strength': sell_strength,
                        'reason': 'Near cycle peak'
                    })
                    signal_strength += sell_strength
            
            # Turning point prediction signals
            next_turning = turning_points.get('next_turning_point')
            if next_turning and next_turning['confidence'] > 0.6:
                if next_turning['type'] == 'trough':
                    signals.append({
                        'type': 'buy_soon',
                        'strength': next_turning['confidence'],
                        'reason': f"Trough predicted in {next_turning['periods_ahead']:.1f} periods",
                        'timing': next_turning['periods_ahead']
                    })
                elif next_turning['type'] == 'peak':
                    signals.append({
                        'type': 'sell_soon',
                        'strength': next_turning['confidence'],
                        'reason': f"Peak predicted in {next_turning['periods_ahead']:.1f} periods",
                        'timing': next_turning['periods_ahead']
                    })
            
            # Overall signal strength weighted by quality
            signal_strength = signal_strength * quality
            
            return {
                'signals': signals,
                'signal_strength': min(1.0, signal_strength),
                'signal_count': len(signals),
                'quality_factor': quality
            }
            
        except Exception as e:
            return {
                'signals': [],
                'signal_strength': 0.0,
                'signal_count': 0,
                'error': str(e)
            }
    
    def _multi_timeframe_phase_analysis(self, data: np.ndarray, base_cycle_period: float) -> Dict:
        """Analyze phase across multiple timeframes."""
        try:
            timeframes = {
                'short': base_cycle_period * 0.5,
                'medium': base_cycle_period,
                'long': base_cycle_period * 2.0
            }
            
            tf_results = {}
            
            for tf_name, cycle_period in timeframes.items():
                if cycle_period < self.min_cycle_length or cycle_period > self.max_cycle_length:
                    continue
                
                try:
                    # Calculate phase for this timeframe
                    phase_info = self._calculate_instantaneous_phase(data, cycle_period)
                    velocity_info = self._calculate_phase_velocity(phase_info['phase_series'])
                    
                    tf_results[tf_name] = {
                        'cycle_period': cycle_period,
                        'current_phase': phase_info['current_phase'],
                        'phase_degrees': np.degrees(phase_info['current_phase']),
                        'velocity': velocity_info['current_velocity'],
                        'cycle_position': phase_info['current_phase'] / (2 * np.pi)
                    }
                except Exception:
                    continue
            
            # Calculate phase alignment across timeframes
            phases = [result['current_phase'] for result in tf_results.values()]
            if len(phases) > 1:
                # Phase alignment score (how well phases are synchronized)
                phase_diffs = []
                for i in range(len(phases)):
                    for j in range(i + 1, len(phases)):
                        diff = abs(phases[i] - phases[j])
                        diff = min(diff, 2 * np.pi - diff)  # Circular distance
                        phase_diffs.append(diff)
                
                alignment_score = 1.0 - np.mean(phase_diffs) / np.pi
                alignment_score = max(0.0, min(1.0, alignment_score))
            else:
                alignment_score = 0.0
            
            return {
                'timeframes': tf_results,
                'alignment_score': alignment_score,
                'timeframe_count': len(tf_results)
            }
            
        except Exception as e:
            return {
                'timeframes': {},
                'alignment_score': 0.0,
                'error': str(e)
            }
    
    def _update_phase_history(self, phase: float, velocity: float, coherence: float):
        """Update phase tracking history."""
        self.phase_history.append(phase)
        self.velocity_history.append(velocity)
        self.coherence_history.append(coherence)
        
        # Keep limited history
        max_history = 100
        if len(self.phase_history) > max_history:
            self.phase_history = self.phase_history[-max_history:]
            self.velocity_history = self.velocity_history[-max_history:]
            self.coherence_history = self.coherence_history[-max_history:]
    
    def _return_empty_results(self, error_msg: str = "") -> Dict:
        """Return empty results structure."""
        return {
            'current_phase': 0.0,
            'phase_degrees': 0.0,
            'phase_velocity': 0.0,
            'phase_acceleration': 0.0,
            'coherence': 0.0,
            'phase_stability': 0.0,
            'turning_points': {'predictions': {}, 'next_turning_point': None},
            'cycle_position': {'cycle_position': 0.0, 'phase_name': 'Unknown'},
            'phase_momentum': {'momentum': 0.0, 'momentum_state': 'Unknown'},
            'signal_strength': 0.0,
            'phase_signals': {'signals': [], 'signal_strength': 0.0},
            'multi_timeframe': {'timeframes': {}, 'alignment_score': 0.0},
            'error': error_msg
        }
    
    def get_phase_statistics(self) -> Dict:
        """Get statistics from phase history."""
        if not self.phase_history:
            return {'error': 'No phase history available'}
        
        return {
            'average_phase': circmean(self.phase_history),
            'phase_consistency': circstd(self.phase_history),
            'average_velocity': np.mean(self.velocity_history),
            'velocity_stability': np.std(self.velocity_history),
            'average_coherence': np.mean(self.coherence_history),
            'coherence_trend': np.polyfit(range(len(self.coherence_history)), self.coherence_history, 1)[0],
            'history_length': len(self.phase_history)
        }

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data with known phase characteristics
    np.random.seed(42)
    t = np.arange(200)
    
    # Create signal with known 20-period cycle
    cycle_period = 20
    signal_data = (
        np.sin(2 * np.pi * t / cycle_period) +     # Main cycle
        0.2 * np.sin(2 * np.pi * t / (cycle_period * 2)) +  # Harmonic
        0.1 * np.random.randn(len(t))              # Noise
    )
    
    # Add trend
    signal_data += 0.01 * t
    
    # Initialize phase analyzer
    phase_analyzer = PhaseAnalysis(
        min_cycle_length=10,
        max_cycle_length=40,
        phase_smoothing=3,
        coherence_window=15
    )
    
    # Perform phase analysis
    results = phase_analyzer.calculate(signal_data, cycle_period=cycle_period)
    
    print("Phase Analysis Results:")
    print(f"Current Phase: {results['phase_degrees']:.1f}°")
    print(f"Phase Velocity: {results['phase_velocity']:.3f}")
    print(f"Phase Coherence: {results['coherence']:.3f}")
    print(f"Cycle Position: {results['cycle_position']['phase_name']}")
    print(f"Phase Momentum: {results['phase_momentum']['momentum_state']}")
    print(f"Signal Strength: {results['signal_strength']:.3f}")
    print(f"Phase Signals: {len(results['phase_signals']['signals'])} signals")
    
    if results['turning_points']['next_turning_point']:
        next_tp = results['turning_points']['next_turning_point']
        print(f"Next Turning Point: {next_tp['type']} in {next_tp['periods_ahead']:.1f} periods")
