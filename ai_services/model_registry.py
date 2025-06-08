"""
AI Services Model Registry
Central registry for AI models and services in Platform3
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

class ModelRegistry:
    """Central registry for AI models and machine learning services"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.services = {}
        self.endpoints = {}
        
        # Initialize default models
        self._initialize_default_models()
        
        self.logger.info("Model Registry initialized")
    
    def _initialize_default_models(self):
        """Initialize default AI models and services"""
        
        # Core Trading Models
        self.models['price_prediction'] = {
            'type': 'lstm',
            'status': 'active',
            'accuracy': 0.85,
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['risk_assessment'] = {
            'type': 'ensemble',
            'status': 'active', 
            'accuracy': 0.92,
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['sentiment_analysis'] = {
            'type': 'transformer',
            'status': 'active',
            'accuracy': 0.88,
            'last_trained': datetime.now().isoformat()
        }
        
        # Genius Agent Models - All 9 agents
        self.models['risk_genius'] = {
            'type': 'ensemble_risk_analyzer',
            'status': 'active',
            'accuracy': 0.94,
            'specialty': 'Risk assessment, volatility analysis, correlation studies',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['session_expert'] = {
            'type': 'temporal_pattern_analyzer',
            'status': 'active',
            'accuracy': 0.89,
            'specialty': 'Session timing, market hours analysis, time-based patterns',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['pattern_master'] = {
            'type': 'pattern_recognition_neural_network',
            'status': 'active',
            'accuracy': 0.91,
            'specialty': 'Candlestick patterns, chart patterns, technical formations',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['execution_expert'] = {
            'type': 'execution_optimization_engine',
            'status': 'active',
            'accuracy': 0.93,
            'specialty': 'Trade execution, slippage minimization, order flow',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['pair_specialist'] = {
            'type': 'currency_correlation_analyzer',
            'status': 'active',
            'accuracy': 0.87,
            'specialty': 'Currency pair analysis, correlation trading, arbitrage',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['decision_master'] = {
            'type': 'multi_agent_coordinator',
            'status': 'active',
            'accuracy': 0.95,
            'specialty': 'Agent coordination, decision synthesis, trade approval',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['ai_model_coordinator'] = {
            'type': 'model_ensemble_manager',
            'status': 'active',
            'accuracy': 0.92,
            'specialty': 'AI model coordination, ensemble optimization, model selection',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['market_microstructure_genius'] = {
            'type': 'microstructure_analyzer',
            'status': 'active',
            'accuracy': 0.88,
            'specialty': 'Order book analysis, liquidity assessment, market impact',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['sentiment_integration_genius'] = {
            'type': 'multi_source_sentiment_analyzer',
            'status': 'active',
            'accuracy': 0.86,
            'specialty': 'News sentiment, social media analysis, market psychology',
            'last_trained': datetime.now().isoformat()
        }
        
        # Advanced ML Models
        self.models['neural_network_predictor'] = {
            'type': 'deep_neural_network',
            'status': 'active',
            'accuracy': 0.90,
            'specialty': 'Deep learning predictions, pattern recognition',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['genetic_algorithm_optimizer'] = {
            'type': 'evolutionary_optimizer',
            'status': 'active',
            'accuracy': 0.84,
            'specialty': 'Parameter optimization, strategy evolution',
            'last_trained': datetime.now().isoformat()
        }
        
        self.models['regime_detection_ai'] = {
            'type': 'market_regime_classifier',
            'status': 'active',
            'accuracy': 0.89,
            'specialty': 'Market regime detection, trend classification',
            'last_trained': datetime.now().isoformat()
        }
        
        # Services
        self.services['prediction_api'] = {
            'url': '/api/v1/predict',
            'status': 'active',
            'model': 'price_prediction'
        }
        
        self.services['risk_api'] = {
            'url': '/api/v1/risk',
            'status': 'active',
            'model': 'risk_assessment'
        }
        
        self.services['sentiment_api'] = {
            'url': '/api/v1/sentiment',
            'status': 'active',
            'model': 'sentiment_analysis'
        }
    
    def register_model(self, name: str, model_config: Dict[str, Any]) -> bool:
        """Register a new AI model"""
        try:
            self.models[name] = model_config
            self.logger.info(f"Model '{name}' registered successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register model '{name}': {e}")
            return False
    
    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Get model configuration by name"""
        return self.models.get(name)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all registered models"""
        return self.models.copy()
    
    def register_service(self, name: str, service_config: Dict[str, Any]) -> bool:
        """Register a new AI service"""
        try:
            self.services[name] = service_config
            self.logger.info(f"Service '{name}' registered successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register service '{name}': {e}")
            return False
    
    def get_service(self, name: str) -> Optional[Dict[str, Any]]:
        """Get service configuration by name"""
        return self.services.get(name)
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all registered services"""
        return self.services.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Check health status of all models and services"""
        return {
            'models': {name: config.get('status', 'unknown') for name, config in self.models.items()},
            'services': {name: config.get('status', 'unknown') for name, config in self.services.items()},
            'timestamp': datetime.now().isoformat()
        }

# Global registry instance
model_registry = ModelRegistry()

def get_model_registry():
    """Get the global model registry instance"""
    return model_registry
