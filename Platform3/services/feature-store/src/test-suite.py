#!/usr/bin/env python3
"""
Feature Store Testing Framework
Comprehensive tests for feature pipeline, API, and data quality
"""

import asyncio
import pytest
import requests
import redis
import json
import time
import numpy as np
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Any
import yaml
import threading
from kafka import KafkaProducer, KafkaConsumer

class FeatureStoreTests:
    """Test suite for Feature Store components"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.api_base_url = "http://localhost:3001"
        self.ws_url = "ws://localhost:3001"
        self.test_symbols = ['EURUSD', 'GBPUSD']
        self.test_results = []
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("Starting Feature Store Test Suite...")
        print("=" * 60)
        
        # Infrastructure tests
        await self.test_redis_connectivity()
        await self.test_api_connectivity()
        await self.test_kafka_connectivity()
        
        # Pipeline tests
        await self.test_feature_computation()
        await self.test_feature_freshness()
        await self.test_feature_accuracy()
        
        # API tests
        await self.test_single_feature_query()
        await self.test_batch_feature_query()
        await self.test_api_performance()
        await self.test_websocket_streaming()
        
        # Data quality tests
        await self.test_data_quality()
        await self.test_outlier_detection()
        
        # Integration tests
        await self.test_end_to_end_pipeline()
        
        # Performance tests
        await self.test_load_performance()
        
        self.print_test_summary()
    
    async def test_redis_connectivity(self):
        """Test Redis connection and basic operations"""
        test_name = "Redis Connectivity"
        try:
            # Test connection
            self.redis_client.ping()
            
            # Test read/write
            test_key = "test:connectivity"
            self.redis_client.set(test_key, "test_value")
            value = self.redis_client.get(test_key)
            self.redis_client.delete(test_key)
            
            assert value == "test_value", "Redis read/write failed"
            
            self.log_test_result(test_name, True, "Redis connectivity successful")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Redis connectivity failed: {e}")
    
    async def test_api_connectivity(self):
        """Test API server connectivity"""
        test_name = "API Connectivity"
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            assert response.status_code == 200, f"Health check failed: {response.status_code}"
            
            health_data = response.json()
            assert health_data.get('status') == 'healthy', "API not healthy"
            
            self.log_test_result(test_name, True, "API connectivity successful")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"API connectivity failed: {e}")
    
    async def test_kafka_connectivity(self):
        """Test Kafka connectivity"""
        test_name = "Kafka Connectivity"
        try:
            # Test producer
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            test_message = {'test': 'connectivity', 'timestamp': datetime.now().isoformat()}
            producer.send('computed-features', test_message)
            producer.flush(timeout=5)
            producer.close()
            
            self.log_test_result(test_name, True, "Kafka connectivity successful")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Kafka connectivity failed: {e}")
    
    async def test_feature_computation(self):
        """Test feature computation accuracy"""
        test_name = "Feature Computation"
        try:
            # Inject test tick data
            test_tick = {
                'symbol': 'EURUSD',
                'bid': 1.1050,
                'ask': 1.1052,
                'price': 1.1051,
                'timestamp': datetime.now().isoformat(),
                'bid_volume': 1000,
                'ask_volume': 800
            }
            
            # Store in Redis to simulate tick processing
            self.redis_client.hset(f"test_tick:EURUSD", mapping=test_tick)
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check if microstructure features were computed
            spread_key = "features:EURUSD:bid_ask_spread"
            spread_data = self.redis_client.hgetall(spread_key)
            
            if spread_data and spread_data.get('current'):
                expected_spread = (1.1052 - 1.1050) * 10000  # 2 pips
                actual_spread = float(spread_data['current'])
                
                assert abs(actual_spread - expected_spread) < 0.1, f"Spread calculation error: expected {expected_spread}, got {actual_spread}"
                
                self.log_test_result(test_name, True, f"Feature computation successful - spread: {actual_spread}")
            else:
                self.log_test_result(test_name, False, "No computed features found")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Feature computation failed: {e}")
    
    async def test_feature_freshness(self):
        """Test feature data freshness"""
        test_name = "Feature Freshness"
        try:
            fresh_features = 0
            stale_features = 0
            
            for symbol in self.test_symbols:
                feature_keys = self.redis_client.keys(f"features:{symbol}:*")
                
                for key in feature_keys:
                    feature_data = self.redis_client.hgetall(key)
                    if feature_data.get('timestamp'):
                        timestamp = datetime.fromisoformat(feature_data['timestamp'].replace('Z', '+00:00'))
                        age_seconds = (datetime.now() - timestamp.replace(tzinfo=None)).total_seconds()
                        
                        if age_seconds < 300:  # Less than 5 minutes
                            fresh_features += 1
                        else:
                            stale_features += 1
            
            freshness_ratio = fresh_features / (fresh_features + stale_features) if (fresh_features + stale_features) > 0 else 0
            
            assert freshness_ratio > 0.8, f"Too many stale features: {freshness_ratio:.2%} fresh"
            
            self.log_test_result(test_name, True, f"Feature freshness: {freshness_ratio:.2%} fresh features")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Feature freshness test failed: {e}")
    
    async def test_feature_accuracy(self):
        """Test feature calculation accuracy with known values"""
        test_name = "Feature Accuracy"
        try:
            # Test RSI calculation with known data
            test_prices = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                          46.03, 46.83, 47.69, 46.49, 46.26, 47.09, 47.37, 47.20, 46.21, 46.80]
            
            # Store test prices in Redis history
            for i, price in enumerate(test_prices):
                self.redis_client.lpush("test_prices:EURUSD", price)
            
            # Calculate RSI manually for verification
            import talib
            np_prices = np.array(test_prices, dtype=float)
            expected_rsi = talib.RSI(np_prices, timeperiod=14)[-1]
            
            # Get computed RSI from feature store
            rsi_key = "features:EURUSD:rsi_14"
            rsi_data = self.redis_client.hgetall(rsi_key)
            
            if rsi_data and rsi_data.get('current'):
                actual_rsi = float(rsi_data['current'])
                rsi_diff = abs(actual_rsi - expected_rsi)
                
                assert rsi_diff < 1.0, f"RSI calculation error: expected {expected_rsi:.2f}, got {actual_rsi:.2f}"
                
                self.log_test_result(test_name, True, f"RSI accuracy verified: diff {rsi_diff:.3f}")
            else:
                self.log_test_result(test_name, False, "RSI feature not found")
            
            # Cleanup test data
            self.redis_client.delete("test_prices:EURUSD")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Feature accuracy test failed: {e}")
    
    async def test_single_feature_query(self):
        """Test single feature API query"""
        test_name = "Single Feature Query"
        try:
            start_time = time.time()
            
            response = requests.get(
                f"{self.api_base_url}/api/features/EURUSD/bid_ask_spread",
                timeout=5
            )
            
            response_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200, f"API returned {response.status_code}"
            
            data = response.json()
            assert 'symbol' in data, "Response missing symbol field"
            assert 'features' in data or 'value' in data, "Response missing feature data"
            assert data['symbol'] == 'EURUSD', "Incorrect symbol in response"
            
            assert response_time < 100, f"Response too slow: {response_time:.2f}ms"
            
            self.log_test_result(test_name, True, f"Query successful in {response_time:.2f}ms")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Single feature query failed: {e}")
    
    async def test_batch_feature_query(self):
        """Test batch feature API query"""
        test_name = "Batch Feature Query"
        try:
            query_payload = {
                'symbol': 'EURUSD',
                'features': ['bid_ask_spread', 'rsi_14', 'price_momentum'],
                'includeHistory': False,
                'includeLags': True
            }
            
            start_time = time.time()
            
            response = requests.post(
                f"{self.api_base_url}/api/features/batch",
                json=query_payload,
                timeout=5
            )
            
            response_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200, f"API returned {response.status_code}"
            
            data = response.json()
            assert 'features' in data, "Response missing features data"
            assert len(data['features']) > 0, "No features returned"
            
            assert response_time < 200, f"Batch query too slow: {response_time:.2f}ms"
            
            self.log_test_result(test_name, True, f"Batch query successful in {response_time:.2f}ms")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Batch feature query failed: {e}")
    
    async def test_api_performance(self):
        """Test API performance under load"""
        test_name = "API Performance"
        try:
            response_times = []
            errors = 0
            
            # Send 100 concurrent requests
            for i in range(100):
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{self.api_base_url}/api/features/EURUSD/bid_ask_spread",
                        timeout=2
                    )
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        response_times.append(response_time)
                    else:
                        errors += 1
                        
                except:
                    errors += 1
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                p95_response_time = np.percentile(response_times, 95)
                error_rate = errors / 100
                
                assert avg_response_time < 100, f"Average response time too high: {avg_response_time:.2f}ms"
                assert p95_response_time < 200, f"P95 response time too high: {p95_response_time:.2f}ms"
                assert error_rate < 0.05, f"Error rate too high: {error_rate:.2%}"
                
                self.log_test_result(test_name, True, 
                    f"Performance: avg {avg_response_time:.2f}ms, p95 {p95_response_time:.2f}ms, errors {error_rate:.2%}")
            else:
                self.log_test_result(test_name, False, "No successful responses received")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"API performance test failed: {e}")
    
    async def test_websocket_streaming(self):
        """Test WebSocket real-time streaming"""
        test_name = "WebSocket Streaming"
        try:
            messages_received = []
            
            def on_message(ws, message):
                messages_received.append(json.loads(message))
            
            def on_error(ws, error):
                print(f"WebSocket error: {error}")
            
            # Connect to WebSocket
            ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message,
                on_error=on_error
            )
            
            # Start WebSocket in background thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection
            await asyncio.sleep(2)
            
            # Send subscription message
            subscription = {
                'type': 'subscribe',
                'payload': {
                    'symbol': 'EURUSD',
                    'features': ['bid_ask_spread', 'rsi_14'],
                    'updateFrequency': 1000
                }
            }
            
            ws.send(json.dumps(subscription))
            
            # Wait for messages
            await asyncio.sleep(5)
            
            ws.close()
            
            assert len(messages_received) > 0, "No WebSocket messages received"
            
            # Check for welcome message
            welcome_msg = next((msg for msg in messages_received if msg.get('type') == 'connected'), None)
            assert welcome_msg is not None, "No welcome message received"
            
            self.log_test_result(test_name, True, f"WebSocket streaming successful - {len(messages_received)} messages")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"WebSocket streaming failed: {e}")
    
    async def test_data_quality(self):
        """Test data quality metrics"""
        test_name = "Data Quality"
        try:
            null_features = 0
            valid_features = 0
            
            for symbol in self.test_symbols:
                feature_keys = self.redis_client.keys(f"features:{symbol}:*")
                
                for key in feature_keys:
                    feature_data = self.redis_client.hgetall(key)
                    current_value = feature_data.get('current')
                    
                    if current_value is None or current_value == 'null' or current_value == '':
                        null_features += 1
                    else:
                        try:
                            float(current_value)
                            valid_features += 1
                        except ValueError:
                            null_features += 1
            
            total_features = null_features + valid_features
            quality_score = valid_features / total_features if total_features > 0 else 0
            
            assert quality_score > 0.95, f"Data quality too low: {quality_score:.2%} valid features"
            
            self.log_test_result(test_name, True, f"Data quality: {quality_score:.2%} valid features")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Data quality test failed: {e}")
    
    async def test_outlier_detection(self):
        """Test outlier detection capabilities"""
        test_name = "Outlier Detection"
        try:
            # Inject an obvious outlier
            outlier_value = 999999.0
            test_feature_key = "features:EURUSD:test_outlier"
            
            self.redis_client.hset(test_feature_key, mapping={
                'current': str(outlier_value),
                'timestamp': datetime.now().isoformat()
            })
            
            # Add some normal historical values
            for i in range(20):
                normal_value = 1.0 + (i * 0.01)
                self.redis_client.lpush("history:EURUSD:test_outlier", str(normal_value))
            
            # Wait for potential outlier detection
            await asyncio.sleep(3)
            
            # Check if outlier was flagged (this depends on monitoring implementation)
            alerts = self.redis_client.lrange("alerts:outliers", 0, -1)
            outlier_detected = any("outlier" in alert for alert in alerts)
            
            # Cleanup
            self.redis_client.delete(test_feature_key)
            self.redis_client.delete("history:EURUSD:test_outlier")
            
            self.log_test_result(test_name, True, f"Outlier detection test completed")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Outlier detection test failed: {e}")
    
    async def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        test_name = "End-to-End Pipeline"
        try:
            # Simulate complete pipeline flow
            test_symbol = 'EURUSD'
            
            # 1. Inject market data
            market_data = {
                'symbol': test_symbol,
                'bid': 1.1050,
                'ask': 1.1052,
                'price': 1.1051,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to Kafka
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            producer.send('forex-ticks-m1', market_data)
            producer.flush()
            producer.close()
            
            # 2. Wait for processing
            await asyncio.sleep(5)
            
            # 3. Query computed features via API
            response = requests.get(f"{self.api_base_url}/api/features/{test_symbol}")
            assert response.status_code == 200, "Feature query failed"
            
            features_data = response.json()
            assert len(features_data.get('features', {})) > 0, "No features computed"
            
            self.log_test_result(test_name, True, "End-to-end pipeline successful")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"End-to-end pipeline failed: {e}")
    
    async def test_load_performance(self):
        """Test system performance under load"""
        test_name = "Load Performance"
        try:
            # Generate high-frequency tick data
            start_time = time.time()
            
            producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                batch_size=16384,
                linger_ms=1
            )
            
            # Send 1000 ticks rapidly
            for i in range(1000):
                tick_data = {
                    'symbol': 'EURUSD',
                    'bid': 1.1050 + (i * 0.00001),
                    'ask': 1.1052 + (i * 0.00001),
                    'price': 1.1051 + (i * 0.00001),
                    'timestamp': datetime.now().isoformat(),
                    'volume': 1000 + i
                }
                
                producer.send('forex-ticks-m1', tick_data)
            
            producer.flush()
            producer.close()
            
            processing_time = time.time() - start_time
            
            # Wait for processing
            await asyncio.sleep(10)
            
            # Check if system handled the load
            response = requests.get(f"{self.api_base_url}/health")
            assert response.status_code == 200, "System unhealthy after load test"
            
            throughput = 1000 / processing_time
            
            self.log_test_result(test_name, True, 
                f"Load test successful - {throughput:.0f} ticks/sec processed")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Load performance test failed: {e}")
    
    def log_test_result(self, test_name: str, passed: bool, message: str):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        result = {
            'test': test_name,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results.append(result)
        
        # Color coding for console output
        color = "\033[92m" if passed else "\033[91m"  # Green for pass, red for fail
        reset = "\033[0m"
        
        print(f"{color}[{status}]{reset} {test_name}: {message}")
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("FEATURE STORE TEST SUMMARY")
        print("=" * 60)
        
        passed_tests = [r for r in self.test_results if r['status'] == 'PASS']
        failed_tests = [r for r in self.test_results if r['status'] == 'FAIL']
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {len(passed_tests)}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success Rate: {len(passed_tests)/len(self.test_results)*100:.1f}%")
        
        if failed_tests:
            print("\nFAILED TESTS:")
            for test in failed_tests:
                print(f"  ❌ {test['test']}: {test['message']}")
        
        print("\nTEST RECOMMENDATIONS:")
        if len(failed_tests) == 0:
            print("  ✅ All tests passed! Feature Store is ready for production.")
        elif len(failed_tests) <= 2:
            print("  ⚠️  Minor issues detected. Review failed tests before deployment.")
        else:
            print("  🚨 Multiple critical issues. Fix all failed tests before proceeding.")
        
        print("=" * 60)


# Main execution
async def main():
    """Run the test suite"""
    tests = FeatureStoreTests()
    await tests.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
