"""
Pattern Recognition AI Module
Provides AI-enhanced pattern recognition capabilities for trading patterns
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Pattern:
    """Represents a trading pattern"""
    name: str
    confidence: float
    start_time: datetime
    end_time: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    pattern_type: str = "unknown"
    
@dataclass
class PatternSignal:
    """Pattern recognition signal"""
    pattern: Pattern
    signal_strength: float
    timestamp: datetime
    metadata: Dict[str, Any]

class PatternRecognitionAI:
    """AI-enhanced pattern recognition system"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PatternRecognitionAI")
        self.patterns = {}
        self.confidence_threshold = 0.7
        self.logger.info("PatternRecognitionAI initialized")
    
    def recognize_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        """Recognize patterns in price data"""
        patterns = []
        
        try:
            # Head and shoulders pattern
            head_shoulders = self._detect_head_and_shoulders(data)
            if head_shoulders:
                patterns.append(head_shoulders)
            
            # Double top/bottom
            double_patterns = self._detect_double_patterns(data)
            patterns.extend(double_patterns)
            
            # Triangle patterns
            triangles = self._detect_triangles(data)
            patterns.extend(triangles)
            
            # Flag and pennant patterns
            flags = self._detect_flags_and_pennants(data)
            patterns.extend(flags)
            
            self.logger.debug(f"Recognized {len(patterns)} patterns")
            
        except Exception as e:
            self.logger.error(f"Error recognizing patterns: {e}")
        
        return patterns
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> Optional[Pattern]:
        """Detect head and shoulders pattern"""
        if len(data) < 20:
            return None
        
        try:
            # Simplified head and shoulders detection
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            # Look for three peaks pattern
            peaks = self._find_peaks(high_prices)
            if len(peaks) >= 3:
                # Check if middle peak is highest
                left_peak, head_peak, right_peak = peaks[-3:]
                
                if (high_prices[head_peak] > high_prices[left_peak] and 
                    high_prices[head_peak] > high_prices[right_peak] and
                    abs(high_prices[left_peak] - high_prices[right_peak]) < 
                    high_prices[head_peak] * 0.02):  # Shoulders roughly equal
                    
                    confidence = min(0.9, 0.5 + abs(high_prices[head_peak] - 
                                                  max(high_prices[left_peak], high_prices[right_peak])) / 
                                   high_prices[head_peak])
                    
                    return Pattern(
                        name="Head and Shoulders",
                        confidence=confidence,
                        start_time=data.index[left_peak],
                        end_time=data.index[right_peak],
                        pattern_type="reversal"
                    )
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {e}")
        
        return None
    
    def _detect_double_patterns(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect double top/bottom patterns"""
        patterns = []
        
        try:
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            # Double top
            peaks = self._find_peaks(high_prices)
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    peak1, peak2 = peaks[i], peaks[i + 1]
                    if (abs(high_prices[peak1] - high_prices[peak2]) < 
                        high_prices[peak1] * 0.02 and  # Peaks roughly equal
                        peak2 - peak1 > 5):  # Sufficient distance
                        
                        confidence = 0.6 + (1.0 - abs(high_prices[peak1] - high_prices[peak2]) / 
                                           high_prices[peak1]) * 0.3
                        
                        patterns.append(Pattern(
                            name="Double Top",
                            confidence=confidence,
                            start_time=data.index[peak1],
                            end_time=data.index[peak2],
                            pattern_type="reversal"
                        ))
            
            # Double bottom
            troughs = self._find_troughs(low_prices)
            if len(troughs) >= 2:
                for i in range(len(troughs) - 1):
                    trough1, trough2 = troughs[i], troughs[i + 1]
                    if (abs(low_prices[trough1] - low_prices[trough2]) < 
                        low_prices[trough1] * 0.02 and  # Troughs roughly equal
                        trough2 - trough1 > 5):  # Sufficient distance
                        
                        confidence = 0.6 + (1.0 - abs(low_prices[trough1] - low_prices[trough2]) / 
                                           low_prices[trough1]) * 0.3
                        
                        patterns.append(Pattern(
                            name="Double Bottom",
                            confidence=confidence,
                            start_time=data.index[trough1],
                            end_time=data.index[trough2],
                            pattern_type="reversal"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error detecting double patterns: {e}")
        
        return patterns
    
    def _detect_triangles(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect triangle patterns"""
        patterns = []
        
        try:
            if len(data) < 15:
                return patterns
            
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            # Simplified triangle detection
            # Look for converging trend lines
            window_size = min(20, len(data) // 2)
            
            for start_idx in range(len(data) - window_size):
                end_idx = start_idx + window_size
                
                highs_slice = high_prices[start_idx:end_idx]
                lows_slice = low_prices[start_idx:end_idx]
                
                # Calculate trend lines
                high_trend = np.polyfit(range(len(highs_slice)), highs_slice, 1)[0]
                low_trend = np.polyfit(range(len(lows_slice)), lows_slice, 1)[0]
                
                # Check for convergence
                if abs(high_trend) > 0.001 and abs(low_trend) > 0.001:
                    if high_trend < 0 and low_trend > 0:  # Ascending triangle
                        patterns.append(Pattern(
                            name="Ascending Triangle",
                            confidence=0.7,
                            start_time=data.index[start_idx],
                            end_time=data.index[end_idx - 1],
                            pattern_type="continuation"
                        ))
                    elif high_trend > 0 and low_trend < 0:  # Descending triangle
                        patterns.append(Pattern(
                            name="Descending Triangle",
                            confidence=0.7,
                            start_time=data.index[start_idx],
                            end_time=data.index[end_idx - 1],
                            pattern_type="continuation"
                        ))
                    elif high_trend < 0 and low_trend > 0 and abs(high_trend + low_trend) < 0.001:  # Symmetrical
                        patterns.append(Pattern(
                            name="Symmetrical Triangle",
                            confidence=0.65,
                            start_time=data.index[start_idx],
                            end_time=data.index[end_idx - 1],
                            pattern_type="continuation"
                        ))
        
        except Exception as e:
            self.logger.error(f"Error detecting triangles: {e}")
        
        return patterns
    
    def _detect_flags_and_pennants(self, data: pd.DataFrame) -> List[Pattern]:
        """Detect flag and pennant patterns"""
        patterns = []
        
        try:
            if len(data) < 10:
                return patterns
            
            close_prices = data['close'].values
            
            # Look for sharp moves followed by consolidation
            for i in range(5, len(data) - 5):
                # Check for sharp move (flag pole)
                pole_start = max(0, i - 10)
                pole_move = abs(close_prices[i] - close_prices[pole_start]) / close_prices[pole_start]
                
                if pole_move > 0.05:  # Significant move (5%+)
                    # Check for consolidation after the move
                    consolidation_end = min(len(data), i + 10)
                    consolidation_prices = close_prices[i:consolidation_end]
                    
                    if len(consolidation_prices) > 3:
                        volatility = np.std(consolidation_prices) / np.mean(consolidation_prices)
                        
                        if volatility < 0.02:  # Low volatility consolidation
                            if close_prices[i] > close_prices[pole_start]:
                                pattern_name = "Bull Flag"
                            else:
                                pattern_name = "Bear Flag"
                            
                            patterns.append(Pattern(
                                name=pattern_name,
                                confidence=0.75,
                                start_time=data.index[pole_start],
                                end_time=data.index[consolidation_end - 1],
                                pattern_type="continuation"
                            ))
        
        except Exception as e:
            self.logger.error(f"Error detecting flags and pennants: {e}")
        
        return patterns
    
    def _find_peaks(self, prices: np.ndarray, min_distance: int = 3) -> List[int]:
        """Find peaks in price data"""
        peaks = []
        
        for i in range(min_distance, len(prices) - min_distance):
            is_peak = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and prices[j] >= prices[i]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    def _find_troughs(self, prices: np.ndarray, min_distance: int = 3) -> List[int]:
        """Find troughs in price data"""
        troughs = []
        
        for i in range(min_distance, len(prices) - min_distance):
            is_trough = True
            
            # Check if current point is lower than surrounding points
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and prices[j] <= prices[i]:
                    is_trough = False
                    break
            
            if is_trough:
                troughs.append(i)
        
        return troughs
    
    def generate_signals(self, patterns: List[Pattern]) -> List[PatternSignal]:
        """Generate trading signals from recognized patterns"""
        signals = []
        
        for pattern in patterns:
            if pattern.confidence >= self.confidence_threshold:
                signal_strength = pattern.confidence
                
                # Adjust signal strength based on pattern type
                if pattern.pattern_type == "reversal":
                    signal_strength *= 1.1  # Boost reversal patterns
                elif pattern.pattern_type == "continuation":
                    signal_strength *= 0.9  # Slightly reduce continuation patterns
                
                signal = PatternSignal(
                    pattern=pattern,
                    signal_strength=min(1.0, signal_strength),
                    timestamp=datetime.now(),
                    metadata={
                        "pattern_type": pattern.pattern_type,
                        "confidence": pattern.confidence,
                        "pattern_name": pattern.name
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    def update_parameters(self, **kwargs):
        """Update AI parameters"""
        if 'confidence_threshold' in kwargs:
            self.confidence_threshold = kwargs['confidence_threshold']
            self.logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "patterns_detected": len(self.patterns),
            "confidence_threshold": self.confidence_threshold,
            "available_patterns": [
                "Head and Shoulders", "Double Top", "Double Bottom",
                "Ascending Triangle", "Descending Triangle", 
                "Symmetrical Triangle", "Bull Flag", "Bear Flag"
            ]
        }
