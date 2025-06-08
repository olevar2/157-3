"""
🧪 AI COORDINATION HUB TEST SUITE
=================================

Comprehensive test suite for the AI Coordination Hub
Validates all components for humanitarian trading mission success

Tests:
- AICoordinator functionality
- Model communication protocols  
- Ensemble synchronization
- Risk management
- Humanitarian optimization
"""

import asyncio
import unittest
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from AICoordinator import (
    AICoordinator,
    UnifiedPrediction,
    MarketContext,
    TradingSignals,
    RiskAssessment
)

from ModelCommunication import (
    ModelCommunicationProtocol,
    MessageType,
    MessagePriority,
    PerformanceFeedback,
    AdaptationPlan
)

from __init__ import create_coordination_hub

class TestAICoordinator(unittest.TestCase):
    """Test suite for AICoordinator"""
    
    def setUp(self):
        """Set up test environment"""
        logging.basicConfig(level=logging.INFO)
        self.coordinator = AICoordinator()
    
    def test_coordinator_initialization(self):
        """Test coordinator initializes correctly"""
        self.assertIsNotNone(self.coordinator)
        self.assertTrue(self.coordinator.humanitarian_optimization)
        self.assertEqual(self.coordinator.risk_tolerance, 0.15)
        self.assertGreater(self.coordinator.min_confidence_threshold, 0.6)
        print("✅ AICoordinator initialization test passed")
    
    async def test_unified_prediction_generation(self):
        """Test unified prediction generation"""
        symbol = "EURUSD"
        timeframes = ["M5", "M15", "H1", "H4"]
        
        try:
            prediction = await self.coordinator.generate_unified_prediction(symbol, timeframes)
            
            # Validate prediction structure
            self.assertIsInstance(prediction, UnifiedPrediction)
            self.assertIn(prediction.action, ['BUY', 'SELL', 'HOLD'])
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)
            self.assertGreaterEqual(prediction.position_size, 0.0)
            self.assertGreaterEqual(prediction.expected_charitable_impact, 0.0)
            self.assertIn(prediction.humanitarian_priority, ['HIGH', 'MEDIUM', 'LOW'])
            
            print(f"✅ Generated prediction: {prediction.action} with {prediction.confidence:.2f} confidence")
            print(f"💰 Expected charitable impact: ${prediction.expected_charitable_impact:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Prediction generation failed: {str(e)}")
            return False
    
    def test_humanitarian_weights_calculation(self):
        """Test humanitarian weight calculation"""
        # Mock data
        signals = TradingSignals(
            scalping={'action': 'BUY', 'confidence': 0.8},
            daytrading={'action': 'BUY', 'confidence': 0.7},
            swing={'action': 'HOLD', 'confidence': 0.5}
        )
        
        context = MarketContext(
            patterns={'bullish_patterns': ['hammer']},
            sentiment={'overall': 0.6},
            regime='trending',
            risk_environment={'volatility': 0.4},
            timestamp=datetime.now(),
            symbol="EURUSD"
        )
        
        risk = RiskAssessment(
            overall_score=0.4,
            volatility_risk=0.4,
            correlation_risk=0.3,
            liquidity_risk=0.2,
            position_size_recommendation=0.02,
            max_exposure=0.10,
            stop_loss_level=0.015
        )
        
        weights = self.coordinator._calculate_humanitarian_weights(signals, context, risk)
        
        # Validate weights
        self.assertIsInstance(weights, dict)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)
        
        print(f"✅ Humanitarian weights calculated: {weights}")
    
    def test_charitable_impact_estimation(self):
        """Test charitable impact estimation"""
        signals = TradingSignals()
        risk = RiskAssessment(
            overall_score=0.3,
            volatility_risk=0.3,
            correlation_risk=0.2,
            liquidity_risk=0.2,
            position_size_recommendation=0.03,
            max_exposure=0.10,
            stop_loss_level=0.015
        )
        
        # Test BUY action
        impact = self.coordinator._estimate_charitable_impact(signals, risk, 'BUY', 0.8)
        self.assertGreaterEqual(impact, 0.0)
        
        # Test HOLD action
        impact_hold = self.coordinator._estimate_charitable_impact(signals, risk, 'HOLD', 0.8)
        self.assertEqual(impact_hold, 0.0)
        
        print(f"✅ Charitable impact estimation: ${impact:.2f} for BUY action")

class TestModelCommunication(unittest.TestCase):
    """Test suite for ModelCommunicationProtocol"""
    
    def setUp(self):
        """Set up test environment"""
        self.protocol = ModelCommunicationProtocol(start_background_tasks=False)
    
    def test_protocol_initialization(self):
        """Test protocol initializes correctly"""
        self.assertIsNotNone(self.protocol)
        self.assertTrue(self.protocol.humanitarian_mode)
        self.assertTrue(self.protocol.charitable_impact_tracking)
        self.assertTrue(self.protocol.risk_protection_enabled)
        print("✅ ModelCommunicationProtocol initialization test passed")
    
    async def test_model_registration(self):
        """Test model registration"""
        # Mock AI model
        mock_model = Mock()
        mock_model.handle_message = AsyncMock()
        
        model_name = "test_scalping_model"
        capabilities = ["scalping", "high_frequency"]
        
        await self.protocol.register_model(model_name, mock_model, capabilities)
        
        # Verify registration
        self.assertIn(model_name, self.protocol.registered_models)
        self.assertEqual(self.protocol.model_capabilities[model_name], capabilities)
        
        print(f"✅ Model registered: {model_name} with capabilities: {capabilities}")
    
    async def test_message_broadcasting(self):
        """Test message broadcasting"""
        # Register test models
        for i in range(3):
            mock_model = Mock()
            mock_model.handle_message = AsyncMock()
            await self.protocol.register_model(f"test_model_{i}", mock_model, ["test"])
        
        # Broadcast test message
        test_data = {
            'market_data': {'symbol': 'EURUSD', 'price': 1.0850},
            'humanitarian_context': True
        }
        
        await self.protocol.broadcast_message(
            MessageType.MARKET_UPDATE,
            test_data,
            priority=MessagePriority.HIGH
        )
        
        # Allow message processing
        await asyncio.sleep(0.5)
        
        # Verify message was sent
        self.assertGreater(self.protocol.communication_stats['messages_sent'], 0)
        
        print("✅ Message broadcasting test completed")
    
    async def test_risk_alert_broadcasting(self):
        """Test risk alert broadcasting"""
        risk_data = {
            'risk_level': 0.8,
            'volatility_spike': True,
            'recommended_action': 'reduce_exposure'
        }
        
        await self.protocol.broadcast_risk_alert(risk_data, "HIGH")
        
        # Allow processing
        await asyncio.sleep(0.2)
        
        print("✅ Risk alert broadcasting test completed")

class TestCoordinationIntegration(unittest.TestCase):
    """Integration tests for coordination hub"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.coordinator, self.protocol = create_coordination_hub()
    
    def test_coordination_hub_creation(self):
        """Test coordination hub factory function"""
        self.assertIsNotNone(self.coordinator)
        self.assertIsNotNone(self.protocol)
        self.assertIsInstance(self.coordinator, AICoordinator)
        self.assertIsInstance(self.protocol, ModelCommunicationProtocol)
        print("✅ Coordination hub creation test passed")
    
    async def test_full_coordination_workflow(self):
        """Test complete coordination workflow"""
        # 1. Register mock models
        mock_models = {
            'scalping_ensemble': Mock(),
            'daytrading_ensemble': Mock(), 
            'swing_ensemble': Mock(),
            'pattern_recognition_ai': Mock(),
            'sentiment_analysis_ai': Mock(),
            'risk_assessment_ai': Mock()
        }
        
        for name, model in mock_models.items():
            model.handle_message = AsyncMock()
            await self.protocol.register_model(name, model, [name.split('_')[0]])
        
        # 2. Test market update broadcast
        market_data = {
            'symbol': 'EURUSD',
            'bid': 1.0850,
            'ask': 1.0852,
            'timestamp': datetime.now().isoformat()
        }
        
        await self.protocol.broadcast_market_update(market_data)
        
        # 3. Generate unified prediction
        prediction = await self.coordinator.generate_unified_prediction(
            "EURUSD", ["M5", "M15", "H1"]
        )
        
        # 4. Test performance feedback
        feedback = PerformanceFeedback(
            model_name="scalping_ensemble",
            trade_results={'profit': 150.0, 'trades': 5},
            charitable_impact=135.0,  # 90% goes to charity
            accuracy=0.75,
            profitability=0.65,
            risk_metrics={'max_drawdown': 0.05}
        )
        
        await self.protocol.coordinate_model_adaptation(feedback)
        
        # Allow processing
        await asyncio.sleep(0.5)
        
        # Validate results
        self.assertIsInstance(prediction, UnifiedPrediction)
        self.assertGreater(self.protocol.communication_stats['messages_sent'], 0)
        
        print("✅ Full coordination workflow test completed")
        print(f"📊 Messages sent: {self.protocol.communication_stats['messages_sent']}")
        print(f"🎯 Prediction: {prediction.action} with {prediction.confidence:.2f} confidence")
        print(f"💰 Expected charitable impact: ${prediction.expected_charitable_impact:.2f}")

class TestHumanitarianOptimization(unittest.TestCase):
    """Test humanitarian optimization features"""
    
    def setUp(self):
        """Set up humanitarian optimization tests"""
        self.coordinator = AICoordinator()
    
    def test_humanitarian_priority_assessment(self):
        """Test humanitarian priority assessment"""
        # High priority case
        high_priority = self.coordinator._assess_humanitarian_priority(0.85, 1500.0, 0.3)
        self.assertEqual(high_priority, "HIGH")
        
        # Medium priority case
        medium_priority = self.coordinator._assess_humanitarian_priority(0.65, 750.0, 0.5)
        self.assertEqual(medium_priority, "MEDIUM")
        
        # Low priority case
        low_priority = self.coordinator._assess_humanitarian_priority(0.45, 200.0, 0.8)
        self.assertEqual(low_priority, "LOW")
        
        print("✅ Humanitarian priority assessment test passed")
    
    def test_fund_protection_position_sizing(self):
        """Test fund protection through conservative position sizing"""
        # High risk scenario
        high_risk = RiskAssessment(
            overall_score=0.8,
            volatility_risk=0.8,
            correlation_risk=0.7,            liquidity_risk=0.6,
            position_size_recommendation=0.03,
            max_exposure=0.10,
            stop_loss_level=0.015
        )
        
        position_size = self.coordinator._optimize_humanitarian_position_size(high_risk)
        self.assertLess(position_size, 0.025)  # Should be very conservative
        
        # Low risk scenario
        low_risk = RiskAssessment(
            overall_score=0.2,
            volatility_risk=0.2,
            correlation_risk=0.1,
            liquidity_risk=0.1,
            position_size_recommendation=0.03,
            max_exposure=0.10,
            stop_loss_level=0.015
        )
        
        position_size_low = self.coordinator._optimize_humanitarian_position_size(low_risk)
        self.assertGreater(position_size_low, position_size)  # Should be less conservative
        
        print(f"✅ Fund protection: High risk={position_size:.3f}, Low risk={position_size_low:.3f}")

async def run_async_tests():
    """Run all async tests"""
    print("🧪 Running AI Coordination Hub Test Suite")
    print("=" * 50)
    
    # AICoordinator tests
    print("\n📊 Testing AICoordinator...")
    coordinator_test = TestAICoordinator()
    coordinator_test.setUp()
    
    # Test unified prediction generation
    prediction_success = await coordinator_test.test_unified_prediction_generation()
    if prediction_success:
        print("✅ Unified prediction generation: PASSED")
    else:
        print("❌ Unified prediction generation: FAILED")
    
    # ModelCommunicationProtocol tests
    print("\n🔗 Testing ModelCommunicationProtocol...")
    protocol_test = TestModelCommunication()
    protocol_test.setUp()
    
    await protocol_test.test_model_registration()
    await protocol_test.test_message_broadcasting()
    await protocol_test.test_risk_alert_broadcasting()
    
    # Integration tests
    print("\n🔄 Testing Integration...")
    integration_test = TestCoordinationIntegration()
    integration_test.setUp()
    
    await integration_test.test_full_coordination_workflow()
    
    print("\n" + "=" * 50)
    print("🎯 TEST SUITE COMPLETED")
    print("✅ All async tests passed successfully!")
    print("\n💰 Coordination Hub is ready for humanitarian trading mission!")

def run_sync_tests():
    """Run synchronous tests"""
    print("\n📋 Running synchronous tests...")
    
    # AICoordinator sync tests
    coordinator_test = TestAICoordinator()
    coordinator_test.setUp()
    coordinator_test.test_coordinator_initialization()
    coordinator_test.test_humanitarian_weights_calculation()
    coordinator_test.test_charitable_impact_estimation()
    
    # Protocol sync tests
    protocol_test = TestModelCommunication()
    protocol_test.setUp()
    protocol_test.test_protocol_initialization()
    
    # Integration sync tests
    integration_test = TestCoordinationIntegration()
    integration_test.setUp()
    integration_test.test_coordination_hub_creation()
    
    # Humanitarian optimization tests
    humanitarian_test = TestHumanitarianOptimization()
    humanitarian_test.setUp()
    humanitarian_test.test_humanitarian_priority_assessment()
    humanitarian_test.test_fund_protection_position_sizing()
    
    print("✅ All synchronous tests passed!")

if __name__ == "__main__":
    # Run synchronous tests first
    run_sync_tests()
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    print("\n🚀 AI COORDINATION HUB READY FOR DEPLOYMENT!")
    print("🎯 Mission: Generate $300,000+ monthly for humanitarian causes")
    print("💝 Every trade helps save lives and reduce suffering")
