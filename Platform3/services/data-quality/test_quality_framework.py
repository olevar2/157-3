#!/usr/bin/env python3
"""
Test Suite for Data Quality Framework
Comprehensive testing for validation rules and anomaly detection
"""

import asyncio
import unittest
import json
from datetime import datetime, timedelta
from quality_monitor import DataQualityMonitor, ValidationResult, Severity
from anomaly_detection import AnomalyDetectionEngine, AnomalyType, AnomalySeverity

class TestDataQualityFramework(unittest.TestCase):
    """Test cases for the data quality framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = DataQualityMonitor()
        self.anomaly_engine = AnomalyDetectionEngine()
        
    def test_ohlc_validation_valid_data(self):
        """Test OHLC validation with valid data"""
        valid_data = {
            'symbol': 'EURUSD',
            'open': 1.0500,
            'high': 1.0510,
            'low': 1.0495,
            'close': 1.0505,
            'volume': 1000,
            'timestamp': '2024-12-19T10:00:00Z'
        }
        
        # This should pass validation
        results = asyncio.run(self.monitor.validate_market_data(valid_data))
        failed_results = [r for r in results if not r.passed]
        self.assertEqual(len(failed_results), 0, "Valid OHLC data should pass validation")
    
    def test_ohlc_validation_invalid_high_low(self):
        """Test OHLC validation with invalid high/low relationship"""
        invalid_data = {
            'symbol': 'EURUSD',
            'open': 1.0500,
            'high': 1.0490,  # High < Low (invalid)
            'low': 1.0495,
            'close': 1.0505,
            'volume': 1000,
            'timestamp': '2024-12-19T10:00:00Z'
        }
        
        results = asyncio.run(self.monitor.validate_market_data(invalid_data))
        failed_results = [r for r in results if not r.passed and r.rule_name == 'high_low_range']
        self.assertEqual(len(failed_results), 1, "Invalid high/low should fail validation")
        self.assertEqual(failed_results[0].severity, Severity.CRITICAL)
    
    def test_ohlc_validation_invalid_open_close_range(self):
        """Test OHLC validation with open/close outside high/low range"""
        invalid_data = {
            'symbol': 'EURUSD',
            'open': 1.0520,  # Open > High (invalid)
            'high': 1.0510,
            'low': 1.0495,
            'close': 1.0505,
            'volume': 1000,
            'timestamp': '2024-12-19T10:00:00Z'
        }
        
        results = asyncio.run(self.monitor.validate_market_data(invalid_data))
        failed_results = [r for r in results if not r.passed and r.rule_name == 'open_close_within_range']
        self.assertEqual(len(failed_results), 1, "Open/Close outside range should fail validation")
    
    def test_volume_validation_negative(self):
        """Test volume validation with negative volume"""
        invalid_data = {
            'symbol': 'EURUSD',
            'open': 1.0500,
            'high': 1.0510,
            'low': 1.0495,
            'close': 1.0505,
            'volume': -100,  # Negative volume (invalid)
            'timestamp': '2024-12-19T10:00:00Z'
        }
        
        results = asyncio.run(self.monitor.validate_market_data(invalid_data))
        failed_results = [r for r in results if not r.passed and r.rule_name == 'non_negative_volume']
        self.assertEqual(len(failed_results), 1, "Negative volume should fail validation")
    
    def test_spread_validation(self):
        """Test spread validation for scalping"""
        # Test with excessive spread
        high_spread_data = {
            'symbol': 'EURUSD',
            'bid': 1.0500,
            'ask': 1.0510,  # 10 pip spread (excessive for EURUSD)
            'timestamp': '2024-12-19T10:00:00Z'
        }
        
        results = asyncio.run(self.monitor.validate_market_data(high_spread_data))
        failed_results = [r for r in results if not r.passed and r.rule_name == 'spread_validation']
        self.assertEqual(len(failed_results), 1, "Excessive spread should fail validation")
    
    def test_future_timestamp_validation(self):
        """Test timestamp validation with future timestamp"""
        future_time = datetime.utcnow() + timedelta(minutes=5)
        invalid_data = {
            'symbol': 'EURUSD',
            'open': 1.0500,
            'high': 1.0510,
            'low': 1.0495,
            'close': 1.0505,
            'volume': 1000,
            'timestamp': future_time.isoformat() + 'Z'
        }
        
        results = asyncio.run(self.monitor.validate_market_data(invalid_data))
        failed_results = [r for r in results if not r.passed and r.rule_name == 'future_timestamp']
        self.assertEqual(len(failed_results), 1, "Future timestamp should fail validation")
    
    def test_trading_data_validation_order_size(self):
        """Test trading data validation for order size limits"""
        invalid_order = {
            'order_type': 'market',
            'lot_size': 150,  # Exceeds 100 lot limit
            'price': 1.0500,
            'symbol': 'EURUSD'
        }
        
        results = asyncio.run(self.monitor.validate_trading_data(invalid_order))
        failed_results = [r for r in results if not r.passed and r.rule_name == 'order_size_limits']
        self.assertEqual(len(failed_results), 1, "Excessive order size should fail validation")
    
    def test_trading_data_validation_negative_price(self):
        """Test trading data validation for negative price"""
        invalid_order = {
            'order_type': 'limit',
            'lot_size': 1.0,
            'price': -1.0500,  # Negative price (invalid)
            'symbol': 'EURUSD'
        }
        
        results = asyncio.run(self.monitor.validate_trading_data(invalid_order))
        failed_results = [r for r in results if not r.passed and r.rule_name == 'price_validity']
        self.assertEqual(len(failed_results), 1, "Negative price should fail validation")
    
    def test_account_validation_negative_balance(self):
        """Test account validation for negative balance"""
        invalid_account = {
            'balance': -1000,  # Negative balance (invalid)
            'equity': 500,
            'used_margin': 200
        }
        
        results = asyncio.run(self.monitor.validate_trading_data(invalid_account))
        failed_results = [r for r in results if not r.passed and r.rule_name == 'balance_consistency']
        self.assertEqual(len(failed_results), 1, "Negative balance should fail validation")
    
    def test_anomaly_detection_z_score(self):
        """Test Z-score anomaly detection"""
        # Build normal data history
        normal_data = []
        for i in range(50):
            data = {
                'symbol': 'EURUSD',
                'timestamp': f'2024-12-19T10:{i:02d}:00Z',
                'open': 1.0500 + (i * 0.0001),
                'high': 1.0510 + (i * 0.0001),
                'low': 1.0495 + (i * 0.0001),
                'close': 1.0505 + (i * 0.0001),
                'volume': 1000
            }
            normal_data.append(data)
            asyncio.run(self.anomaly_engine.detect_anomalies(data))
        
        # Test with anomalous data
        anomalous_data = {
            'symbol': 'EURUSD',
            'timestamp': '2024-12-19T10:51:00Z',
            'open': 1.0550,
            'high': 1.0600,  # Significant price jump
            'low': 1.0545,
            'close': 1.0590,
            'volume': 1000
        }
        
        anomalies = asyncio.run(self.anomaly_engine.detect_anomalies(anomalous_data))
        statistical_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.STATISTICAL]
        self.assertGreater(len(statistical_anomalies), 0, "Should detect statistical anomaly")
    
    def test_anomaly_detection_volume_spike(self):
        """Test volume spike anomaly detection"""
        # Build normal volume history
        normal_data = []
        for i in range(30):
            data = {
                'symbol': 'EURUSD',
                'timestamp': f'2024-12-19T10:{i:02d}:00Z',
                'open': 1.0500,
                'high': 1.0510,
                'low': 1.0495,
                'close': 1.0505,
                'volume': 1000  # Normal volume
            }
            normal_data.append(data)
            asyncio.run(self.anomaly_engine.detect_anomalies(data))
        
        # Test with volume spike
        volume_spike_data = {
            'symbol': 'EURUSD',
            'timestamp': '2024-12-19T10:31:00Z',
            'open': 1.0500,
            'high': 1.0510,
            'low': 1.0495,
            'close': 1.0505,
            'volume': 10000  # 10x normal volume
        }
        
        anomalies = asyncio.run(self.anomaly_engine.detect_anomalies(volume_spike_data))
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME]
        self.assertGreater(len(volume_anomalies), 0, "Should detect volume anomaly")
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation"""
        # Add some validation results
        self.monitor.validation_results = [
            ValidationResult(
                rule_id="MD001",
                rule_name="test_rule",
                passed=True,
                severity=Severity.MEDIUM,
                timestamp=datetime.utcnow()
            ),
            ValidationResult(
                rule_id="MD002",
                rule_name="test_rule_2",
                passed=False,
                severity=Severity.HIGH,
                timestamp=datetime.utcnow()
            )
        ]
        
        metrics = asyncio.run(self.monitor.calculate_quality_metrics())
        
        self.assertEqual(metrics.total_records_processed, 2)
        self.assertEqual(metrics.failed_validations, 1)
        self.assertEqual(metrics.high_alerts, 1)
        self.assertEqual(metrics.data_quality_score, 0.5)  # 1 passed out of 2
    
    def test_anomaly_summary_generation(self):
        """Test anomaly summary generation"""
        anomalies = [
            type('AnomalyResult', (), {
                'severity': AnomalySeverity.HIGH,
                'anomaly_type': AnomalyType.STATISTICAL,
                'confidence': 0.9
            })(),
            type('AnomalyResult', (), {
                'severity': AnomalySeverity.MEDIUM,
                'anomaly_type': AnomalyType.VOLUME,
                'confidence': 0.8
            })(),
            type('AnomalyResult', (), {
                'severity': AnomalySeverity.HIGH,
                'anomaly_type': AnomalyType.SPIKE,
                'confidence': 0.95
            })()
        ]
        
        summary = self.anomaly_engine.get_anomaly_summary(anomalies)
        
        self.assertEqual(summary['total'], 3)
        self.assertEqual(summary['by_severity']['HIGH'], 2)
        self.assertEqual(summary['by_severity']['MEDIUM'], 1)
        self.assertEqual(summary['by_type']['STATISTICAL'], 1)
        self.assertEqual(summary['by_type']['VOLUME'], 1)
        self.assertEqual(summary['by_type']['SPIKE'], 1)
        self.assertEqual(summary['highest_severity'], 'HIGH')
        self.assertAlmostEqual(summary['average_confidence'], 0.883, places=2)

def run_integration_test():
    """Run integration test with sample data"""
    print("Running Data Quality Framework Integration Test...")
    
    # Initialize components
    monitor = DataQualityMonitor()
    anomaly_engine = AnomalyDetectionEngine()
    
    # Test data samples
    test_samples = [
        # Normal data
        {
            'symbol': 'EURUSD',
            'timestamp': '2024-12-19T10:00:00Z',
            'open': 1.0500,
            'high': 1.0510,
            'low': 1.0495,
            'close': 1.0505,
            'volume': 1000,
            'bid': 1.0504,
            'ask': 1.0506
        },
        # Invalid OHLC data
        {
            'symbol': 'EURUSD',
            'timestamp': '2024-12-19T10:01:00Z',
            'open': 1.0505,
            'high': 1.0500,  # High < Low (invalid)
            'low': 1.0510,
            'close': 1.0508,
            'volume': 1100,
            'bid': 1.0507,
            'ask': 1.0509
        },
        # Anomalous price movement
        {
            'symbol': 'EURUSD',
            'timestamp': '2024-12-19T10:02:00Z',
            'open': 1.0508,
            'high': 1.0600,  # Large price jump
            'low': 1.0505,
            'close': 1.0590,
            'volume': 5000,  # High volume
            'bid': 1.0589,
            'ask': 1.0591
        }
    ]
    
    total_validations = 0
    total_anomalies = 0
    
    for i, sample in enumerate(test_samples):
        print(f"\nProcessing sample {i+1}: {sample['symbol']} at {sample['timestamp']}")
        
        # Run validation
        validation_results = asyncio.run(monitor.validate_market_data(sample))
        failed_validations = [r for r in validation_results if not r.passed]
        
        print(f"  Validation results: {len(validation_results)} total, {len(failed_validations)} failed")
        for result in failed_validations:
            print(f"    ❌ {result.rule_name}: {result.error_message} (Severity: {result.severity.value})")
        
        # Run anomaly detection
        anomalies = asyncio.run(anomaly_engine.detect_anomalies(sample))
        print(f"  Anomaly detection: {len(anomalies)} anomalies detected")
        for anomaly in anomalies:
            print(f"    🚨 {anomaly.anomaly_type.value}: {anomaly.description} (Severity: {anomaly.severity.value})")
        
        total_validations += len(validation_results)
        total_anomalies += len(anomalies)
    
    # Calculate final metrics
    metrics = asyncio.run(monitor.calculate_quality_metrics())
    anomaly_summary = anomaly_engine.get_anomaly_summary(
        [a for sample in test_samples for a in asyncio.run(anomaly_engine.detect_anomalies(sample))]
    )
    
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total validations performed: {total_validations}")
    print(f"Total anomalies detected: {total_anomalies}")
    print(f"Data quality score: {metrics.data_quality_score:.2%}")
    print(f"Data freshness score: {metrics.data_freshness_score:.2%}")
    print(f"Anomaly rate: {metrics.anomaly_rate:.2%}")
    print(f"\nAnomaly breakdown: {anomaly_summary}")
    print(f"\n✅ Integration test completed successfully!")

if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n" + "="*60)
    run_integration_test()
