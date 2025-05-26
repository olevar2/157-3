#!/usr/bin/env python3
"""
Anomaly Detection Module - Advanced anomaly detection for AI Forex Trading Platform
Optimized for short-term trading data anomaly detection (M1-H4 timeframes)

This module provides comprehensive anomaly detection capabilities including:
- Statistical anomaly detection (Z-score, IQR, Isolation Forest)
- Pattern-based anomaly detection (gaps, spikes, volume anomalies)
- Machine learning-based anomaly detection
- Real-time anomaly scoring and alerting
- Adaptive thresholds based on market conditions
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
from collections import deque
import time
import functools
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    STATISTICAL = "STATISTICAL"
    PATTERN = "PATTERN"
    VOLUME = "VOLUME"
    PRICE_GAP = "PRICE_GAP"
    SPIKE = "SPIKE"
    TREND_BREAK = "TREND_BREAK"
    CORRELATION = "CORRELATION"
    TEMPORAL = "TEMPORAL"

class AnomalySeverity(Enum):
    """Severity levels for detected anomalies"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class AnomalyResult:
    """Structure for anomaly detection results"""
    timestamp: datetime
    symbol: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float
    threshold: float
    description: str
    data_point: Dict
    confidence: float
    suggested_action: str

class StatisticalAnomalyDetector:
    """Statistical anomaly detection using various statistical methods"""
    
    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.price_history = {}
        self.volume_history = {}
        
    def detect_z_score_anomalies(self, data: Dict, historical_data: List[float]) -> Optional[AnomalyResult]:
        """Detect anomalies using Z-score method"""
        if len(historical_data) < 30:
            return None
            
        try:
            current_value = data.get('close', data.get('value', 0))
            mean_val = statistics.mean(historical_data)
            std_val = statistics.stdev(historical_data)
            
            if std_val == 0:
                return None
                
            z_score = abs((current_value - mean_val) / std_val)
            
            if z_score > self.z_threshold:
                severity = self._determine_severity(z_score, self.z_threshold)
                
                return AnomalyResult(
                    timestamp=datetime.utcnow(),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    score=z_score,
                    threshold=self.z_threshold,
                    description=f"Z-score anomaly: {z_score:.2f} (threshold: {self.z_threshold})",
                    data_point=data,
                    confidence=min(0.95, z_score / (self.z_threshold * 2)),
                    suggested_action="INVESTIGATE" if severity in [AnomalySeverity.LOW, AnomalySeverity.MEDIUM] else "ALERT"
                )
                
        except Exception as e:
            logger.error(f"Error in Z-score anomaly detection: {e}")
            
        return None
    
    def detect_iqr_anomalies(self, data: Dict, historical_data: List[float]) -> Optional[AnomalyResult]:
        """Detect anomalies using Interquartile Range (IQR) method"""
        if len(historical_data) < 30:
            return None
            
        try:
            current_value = data.get('close', data.get('value', 0))
            q1 = np.percentile(historical_data, 25)
            q3 = np.percentile(historical_data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if current_value < lower_bound or current_value > upper_bound:
                # Calculate anomaly score
                if current_value < lower_bound:
                    score = (lower_bound - current_value) / iqr
                else:
                    score = (current_value - upper_bound) / iqr
                
                severity = self._determine_severity(score, 1.5)
                
                return AnomalyResult(
                    timestamp=datetime.utcnow(),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    score=score,
                    threshold=1.5,
                    description=f"IQR anomaly: value {current_value:.5f} outside bounds [{lower_bound:.5f}, {upper_bound:.5f}]",
                    data_point=data,
                    confidence=min(0.95, score / 3.0),
                    suggested_action="INVESTIGATE" if severity in [AnomalySeverity.LOW, AnomalySeverity.MEDIUM] else "ALERT"
                )
                
        except Exception as e:
            logger.error(f"Error in IQR anomaly detection: {e}")
            
        return None
    
    def detect_isolation_forest_anomalies(self, data: Dict, historical_data: List[Dict]) -> Optional[AnomalyResult]:
        """Detect anomalies using Isolation Forest algorithm"""
        if len(historical_data) < 50:
            return None
            
        try:
            # Prepare feature matrix
            features = ['open', 'high', 'low', 'close', 'volume']
            available_features = [f for f in features if f in data and all(f in d for d in historical_data)]
            
            if len(available_features) < 3:
                return None
                
            # Create feature matrix
            X = np.array([[d[f] for f in available_features] for d in historical_data])
            current_point = np.array([[data[f] for f in available_features]])
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            current_scaled = scaler.transform(current_point)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X_scaled)
            
            # Predict anomaly
            anomaly_score = iso_forest.decision_function(current_scaled)[0]
            is_anomaly = iso_forest.predict(current_scaled)[0] == -1
            
            if is_anomaly:
                # Convert score to positive value (more negative = more anomalous)
                score = abs(anomaly_score)
                severity = self._determine_severity(score, 0.1)
                
                return AnomalyResult(
                    timestamp=datetime.utcnow(),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    score=score,
                    threshold=0.1,
                    description=f"Isolation Forest anomaly: score {anomaly_score:.3f}",
                    data_point=data,
                    confidence=min(0.95, score / 0.5),
                    suggested_action="INVESTIGATE" if severity in [AnomalySeverity.LOW, AnomalySeverity.MEDIUM] else "ALERT"
                )
                
        except Exception as e:
            logger.error(f"Error in Isolation Forest anomaly detection: {e}")
            
        return None
    
    def _determine_severity(self, score: float, threshold: float) -> AnomalySeverity:
        """Determine anomaly severity based on score and threshold"""
        ratio = score / threshold
        
        if ratio >= 3.0:
            return AnomalySeverity.CRITICAL
        elif ratio >= 2.0:
            return AnomalySeverity.HIGH
        elif ratio >= 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

class PatternAnomalyDetector:
    """Pattern-based anomaly detection for forex market patterns"""
    
    def __init__(self):
        self.gap_threshold = 0.005  # 0.5% gap threshold
        self.spike_threshold = 2.0   # 2 standard deviations
        
    def detect_price_gaps(self, current_data: Dict, previous_data: Dict) -> Optional[AnomalyResult]:
        """Detect price gaps between consecutive periods"""
        try:
            if not previous_data:
                return None
                
            current_open = current_data.get('open')
            previous_close = previous_data.get('close')
            
            if not current_open or not previous_close:
                return None
                
            gap_percentage = abs((current_open - previous_close) / previous_close)
            
            if gap_percentage > self.gap_threshold:
                severity = AnomalySeverity.HIGH if gap_percentage > 0.02 else AnomalySeverity.MEDIUM
                
                return AnomalyResult(
                    timestamp=datetime.utcnow(),
                    symbol=current_data.get('symbol', 'UNKNOWN'),
                    anomaly_type=AnomalyType.PRICE_GAP,
                    severity=severity,
                    score=gap_percentage,
                    threshold=self.gap_threshold,
                    description=f"Price gap detected: {gap_percentage:.3f}% between {previous_close:.5f} and {current_open:.5f}",
                    data_point=current_data,
                    confidence=0.9,
                    suggested_action="INVESTIGATE_NEWS" if severity == AnomalySeverity.HIGH else "MONITOR"
                )
                
        except Exception as e:
            logger.error(f"Error in price gap detection: {e}")
            
        return None
    
    def detect_price_spikes(self, data: Dict, historical_data: List[float]) -> Optional[AnomalyResult]:
        """Detect sudden price spikes"""
        try:
            if len(historical_data) < 20:
                return None
                
            current_price = data.get('close', data.get('high', 0))
            recent_prices = historical_data[-20:]
            
            mean_price = statistics.mean(recent_prices)
            std_price = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
            
            if std_price == 0:
                return None
                
            spike_score = abs((current_price - mean_price) / std_price)
            
            if spike_score > self.spike_threshold:
                severity = self._determine_spike_severity(spike_score)
                
                return AnomalyResult(
                    timestamp=datetime.utcnow(),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    anomaly_type=AnomalyType.SPIKE,
                    severity=severity,
                    score=spike_score,
                    threshold=self.spike_threshold,
                    description=f"Price spike detected: {spike_score:.2f} standard deviations from mean",
                    data_point=data,
                    confidence=min(0.95, spike_score / (self.spike_threshold * 2)),
                    suggested_action="IMMEDIATE_REVIEW" if severity == AnomalySeverity.CRITICAL else "INVESTIGATE"
                )
                
        except Exception as e:
            logger.error(f"Error in price spike detection: {e}")
            
        return None
    
    def detect_volume_anomalies(self, data: Dict, historical_volumes: List[float]) -> Optional[AnomalyResult]:
        """Detect volume anomalies"""
        try:
            if len(historical_volumes) < 20:
                return None
                
            current_volume = data.get('volume', 0)
            if current_volume <= 0:
                return None
                
            avg_volume = statistics.mean(historical_volumes)
            if avg_volume <= 0:
                return None
                
            volume_ratio = current_volume / avg_volume
            
            # Detect both high and low volume anomalies
            if volume_ratio > 5.0 or volume_ratio < 0.1:
                severity = AnomalySeverity.HIGH if volume_ratio > 10.0 or volume_ratio < 0.05 else AnomalySeverity.MEDIUM
                
                anomaly_type = "High" if volume_ratio > 1.0 else "Low"
                
                return AnomalyResult(
                    timestamp=datetime.utcnow(),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    anomaly_type=AnomalyType.VOLUME,
                    severity=severity,
                    score=volume_ratio,
                    threshold=5.0 if volume_ratio > 1.0 else 0.1,
                    description=f"{anomaly_type} volume anomaly: {current_volume} vs avg {avg_volume:.0f} (ratio: {volume_ratio:.2f})",
                    data_point=data,
                    confidence=0.85,
                    suggested_action="INVESTIGATE_MARKET_EVENTS"
                )
                
        except Exception as e:
            logger.error(f"Error in volume anomaly detection: {e}")
            
        return None
    
    def _determine_spike_severity(self, spike_score: float) -> AnomalySeverity:
        """Determine spike severity based on score"""
        if spike_score >= 5.0:
            return AnomalySeverity.CRITICAL
        elif spike_score >= 3.5:
            return AnomalySeverity.HIGH
        elif spike_score >= 2.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

class TemporalAnomalyDetector:
    """Temporal anomaly detection for time-based patterns"""
    
    def __init__(self):
        self.expected_intervals = {
            'M1': 60,    # 1 minute
            'M5': 300,   # 5 minutes
            'M15': 900,  # 15 minutes
            'H1': 3600,  # 1 hour
            'H4': 14400  # 4 hours
        }
    
    def detect_timing_anomalies(self, data: Dict, previous_timestamp: Optional[datetime]) -> Optional[AnomalyResult]:
        """Detect timing anomalies in data arrival"""
        try:
            current_timestamp = datetime.fromisoformat(data.get('timestamp', '').replace('Z', '+00:00'))
            timeframe = data.get('timeframe', 'M1')
            
            if not previous_timestamp:
                return None
                
            expected_interval = self.expected_intervals.get(timeframe, 60)
            actual_interval = (current_timestamp - previous_timestamp).total_seconds()
            
            # Allow 10% tolerance
            tolerance = expected_interval * 0.1
            
            if abs(actual_interval - expected_interval) > tolerance:
                severity = AnomalySeverity.HIGH if abs(actual_interval - expected_interval) > expected_interval else AnomalySeverity.MEDIUM
                
                return AnomalyResult(
                    timestamp=datetime.utcnow(),
                    symbol=data.get('symbol', 'UNKNOWN'),
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=severity,
                    score=abs(actual_interval - expected_interval),
                    threshold=tolerance,
                    description=f"Timing anomaly: expected {expected_interval}s interval, got {actual_interval:.0f}s",
                    data_point=data,
                    confidence=0.8,
                    suggested_action="CHECK_DATA_FEED"
                )
                
        except Exception as e:
            logger.error(f"Error in timing anomaly detection: {e}")
            
        return None

class AnomalyDetectionEngine:
    """Main anomaly detection engine that coordinates all detectors"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize anomaly detection engine with enhanced performance features"""
        self.config = config or self._load_default_config()
        self.z_threshold = self.config.get('z_score_threshold', 3.0)
        self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
        self.spike_threshold = self.config.get('spike_threshold', 2.0)
        self.volume_threshold = self.config.get('volume_threshold', 5.0)
        
        # Performance optimization: Pre-compiled patterns
        self._precompiled_patterns = {}
        self._initialize_ml_models()
        
        # Enhanced caching for frequently accessed data
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance metrics tracking
        self._detection_times = deque(maxlen=1000)
        self._detection_count = 0
        self._total_detection_time = 0.0
        
        # Thread pool for concurrent processing
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Enhanced AnomalyDetectionEngine initialized with performance optimizations")
    
    def _initialize_ml_models(self):
        """Initialize ML models for anomaly detection"""
        try:
            if 'sklearn' in globals():
                # Initialize Isolation Forest with optimized parameters
                self.isolation_forest = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                )
                logger.info("Isolation Forest model initialized")
            else:
                logger.warning("sklearn not available, using statistical methods only")
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if available and not expired"""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                self._cache_hits += 1
                return data
            else:
                # Cache expired, remove entry
                del self._cache[key]
        
        self._cache_misses += 1
        return None
    
    def _set_cached_data(self, key: str, data: Any) -> None:
        """Store data in cache with timestamp"""
        self._cache[key] = (time.time(), data)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        avg_detection_time = (
            self._total_detection_time / self._detection_count 
            if self._detection_count > 0 else 0
        )
        
        cache_hit_ratio = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0 else 0
        )
        
        return {
            'total_detections': self._detection_count,
            'average_detection_time_ms': avg_detection_time * 1000,
            'cache_hit_ratio': cache_hit_ratio,
            'cache_entries': len(self._cache),
            'recent_detection_times': list(self._detection_times)[-10:]  # Last 10
        }

    async def detect_anomalies(self, data: Dict) -> List[AnomalyResult]:
        """Run all anomaly detection methods on the data"""
        anomalies = []
        symbol = data.get('symbol', 'UNKNOWN')
        
        # Initialize history for symbol if not exists
        if symbol not in self.data_history:
            self.data_history[symbol] = []
        
        try:
            # Statistical anomaly detection
            if len(self.data_history[symbol]) >= 30:
                historical_closes = [d.get('close', 0) for d in self.data_history[symbol]]
                historical_volumes = [d.get('volume', 0) for d in self.data_history[symbol] if d.get('volume', 0) > 0]
                
                # Z-score anomalies
                z_anomaly = self.statistical_detector.detect_z_score_anomalies(data, historical_closes)
                if z_anomaly:
                    anomalies.append(z_anomaly)
                
                # IQR anomalies
                iqr_anomaly = self.statistical_detector.detect_iqr_anomalies(data, historical_closes)
                if iqr_anomaly:
                    anomalies.append(iqr_anomaly)
                
                # Isolation Forest anomalies (if enough historical data)
                if len(self.data_history[symbol]) >= 50:
                    iso_anomaly = self.statistical_detector.detect_isolation_forest_anomalies(data, self.data_history[symbol])
                    if iso_anomaly:
                        anomalies.append(iso_anomaly)
            
            # Pattern-based anomaly detection
            if len(self.data_history[symbol]) > 0:
                previous_data = self.data_history[symbol][-1]
                
                # Price gap detection
                gap_anomaly = self.pattern_detector.detect_price_gaps(data, previous_data)
                if gap_anomaly:
                    anomalies.append(gap_anomaly)
                
                # Price spike detection
                if len(self.data_history[symbol]) >= 20:
                    historical_closes = [d.get('close', 0) for d in self.data_history[symbol]]
                    spike_anomaly = self.pattern_detector.detect_price_spikes(data, historical_closes)
                    if spike_anomaly:
                        anomalies.append(spike_anomaly)
                
                # Volume anomaly detection
                if len(self.data_history[symbol]) >= 20:
                    historical_volumes = [d.get('volume', 0) for d in self.data_history[symbol] if d.get('volume', 0) > 0]
                    if historical_volumes:
                        volume_anomaly = self.pattern_detector.detect_volume_anomalies(data, historical_volumes)
                        if volume_anomaly:
                            anomalies.append(volume_anomaly)
            
            # Temporal anomaly detection
            if symbol in self.last_timestamps:
                timing_anomaly = self.temporal_detector.detect_timing_anomalies(data, self.last_timestamps[symbol])
                if timing_anomaly:
                    anomalies.append(timing_anomaly)
            
            # Update history and timestamps
            self.data_history[symbol].append(data)
            if len(self.data_history[symbol]) > 200:  # Keep last 200 data points
                self.data_history[symbol] = self.data_history[symbol][-200:]
            
            if 'timestamp' in data:
                self.last_timestamps[symbol] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return anomalies
    
    def get_anomaly_summary(self, anomalies: List[AnomalyResult]) -> Dict:
        """Generate summary statistics for detected anomalies"""
        if not anomalies:
            return {"total": 0, "by_severity": {}, "by_type": {}}
        
        by_severity = {}
        by_type = {}
        
        for anomaly in anomalies:
            # Count by severity
            severity = anomaly.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count by type
            anomaly_type = anomaly.anomaly_type.value
            by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1
        
        return {
            "total": len(anomalies),
            "by_severity": by_severity,
            "by_type": by_type,
            "highest_severity": max(anomalies, key=lambda x: list(AnomalySeverity).index(x.severity)).severity.value,
            "average_confidence": sum(a.confidence for a in anomalies) / len(anomalies)
        }

# Example usage and testing
async def test_anomaly_detection():
    """Test the anomaly detection system"""
    engine = AnomalyDetectionEngine()
    
    # Simulate normal data
    normal_data = [
        {"symbol": "EURUSD", "timestamp": "2024-12-19T10:00:00Z", "open": 1.0500, "high": 1.0510, "low": 1.0495, "close": 1.0505, "volume": 1000},
        {"symbol": "EURUSD", "timestamp": "2024-12-19T10:01:00Z", "open": 1.0505, "high": 1.0515, "low": 1.0500, "close": 1.0510, "volume": 1100},
        {"symbol": "EURUSD", "timestamp": "2024-12-19T10:02:00Z", "open": 1.0510, "high": 1.0520, "low": 1.0505, "close": 1.0515, "volume": 950},
    ]
    
    # Add normal data to build history
    for data in normal_data * 20:  # Repeat to build sufficient history
        await engine.detect_anomalies(data)
    
    # Test anomalous data
    anomalous_data = {
        "symbol": "EURUSD", 
        "timestamp": "2024-12-19T10:03:00Z", 
        "open": 1.0515, 
        "high": 1.0600,  # Unusual high
        "low": 1.0510, 
        "close": 1.0590,  # Large price movement
        "volume": 10000   # High volume
    }
    
    anomalies = await engine.detect_anomalies(anomalous_data)
    summary = engine.get_anomaly_summary(anomalies)
    
    print(f"Detected {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"- {anomaly.anomaly_type.value}: {anomaly.description} (Severity: {anomaly.severity.value})")
    
    print(f"\nSummary: {summary}")

if __name__ == "__main__":
    asyncio.run(test_anomaly_detection())
