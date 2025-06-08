#!/usr/bin/env python3
"""
Platform3 AI Services
Core AI services for model invocation and agent integration
"""

from typing import Dict, List, Any, Optional, Union
import logging
import importlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIServiceProvider:
    """Provider for AI services and model invocations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_models = {
            "genius_agent": "engines.ai_enhancement.genius_agent_integration.GeniusAgentIntegration",
            "sentiment_analyzer": "engines.sentiment.SentimentAnalyzer",
            "ml_advanced": "engines.ml_advanced.PlatformMLEngine",
            "pattern_recognition": "engines.pattern.PatternRecognitionEngine"
        }
        
        self.logger.info(f"AI Service Provider initialized with {len(self.available_models)} models")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name not in self.available_models:
            return {"error": f"Model {model_name} not found"}
            
        return {
            "name": model_name,
            "path": self.available_models[model_name],
            "status": "available"
        }
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models"""
        return {
            "models": list(self.available_models.keys()),
            "count": len(self.available_models)
        }

class ModelRegistry:
    """
    Registry for all AI models in Platform3.
    Used by the REST API server and other services.
    """
    
    def __init__(self):
        self.models = {}
        self.model_versions = {}
        self.model_performance = {}
        self.active_models = set()
        self.logger = logging.getLogger(__name__)
        self._initialize_registry()
        
    def _initialize_registry(self):
        """Initialize the model registry with default models"""
        self.logger.info("Initializing model registry")
        
        # Register default models
        self.register_model(
            "genius_agent_integration",
            "engines.ai_enhancement.genius_agent_integration.GeniusAgentIntegration",
            "1.0.0",
            {"description": "Main integration interface for genius agents and indicators"}
        )
        
        self.logger.info(f"Model registry initialized with {len(self.models)} models")
        
    def register_model(self, model_id: str, model_path: str, version: str, metadata: Dict[str, Any] = None):
        """Register a model in the registry"""
        self.models[model_id] = model_path
        self.model_versions[model_id] = version
        self.model_performance[model_id] = {}
        self.active_models.add(model_id)
        
        if metadata:
            self.update_model_metadata(model_id, metadata)
            
        self.logger.info(f"Registered model: {model_id} version {version}")
        
    def get_model(self, model_id: str) -> Optional[str]:
        """Get model path by ID"""
        return self.models.get(model_id)
        
    def get_model_version(self, model_id: str) -> Optional[str]:
        """Get model version by ID"""
        return self.model_versions.get(model_id)
        
    def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]):
        """Update model metadata"""
        if model_id not in self.models:
            return False
            
        if model_id not in self.model_performance:
            self.model_performance[model_id] = {}
            
        self.model_performance[model_id].update(metadata)
        return True
        
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return {
            "count": len(self.models),
            "active_count": len(self.active_models),
            "models": [
                {
                    "id": model_id,
                    "path": path,
                    "version": self.model_versions.get(model_id, "unknown"),
                    "active": model_id in self.active_models
                }
                for model_id, path in self.models.items()
            ]
        }
        
    def activate_model(self, model_id: str) -> bool:
        """Activate a model"""
        if model_id not in self.models:
            return False
        self.active_models.add(model_id)
        return True
        
    def deactivate_model(self, model_id: str) -> bool:
        """Deactivate a model"""
        if model_id not in self.models:
            return False
        if model_id in self.active_models:
            self.active_models.remove(model_id)
        return True

# Create a global instance of the model registry for importing
model_registry = ModelRegistry()

# Create a global instance of the AI service provider
service_provider = AIServiceProvider()

# Initialize default models
def initialize_default_models():
    """Initialize default models in the registry"""
    # Risk models
    model_registry.register_model(
        'risk_genius', 
        'ai_platform.ai_models.risk_genius.RiskGenius',
        'risk_assessment',
        {'description': 'Professional Risk Management Genius with ultra-fast calculations'}
    )
    
    # Execution models
    model_registry.register_model(
        'execution_expert', 
        'ai_platform.ai_models.execution_expert.ExecutionExpert',
        'execution_optimization',
        {'description': 'Advanced Trade Execution and Timing Expert'}
    )
    
    # Decision models
    model_registry.register_model(
        'decision_master', 
        'ai_platform.ai_models.decision_master.DecisionMaster',
        'trading_decisions',
        {'description': 'Professional Trading Decision Genius'}
    )
    
    # Initialize based on AIServiceProvider models as well
    for model_name, model_path in service_provider.available_models.items():
        model_registry.register_model(
            model_name,
            model_path,
            'service',
            {'description': f'Service model: {model_name}'}
        )
    
    logger.info("Default models initialized in registry")

# Initialize models when module is imported
initialize_default_models()

def get_model_registry():
    """Get the global model registry instance"""
    return model_registry

# Export the main functions
__all__ = ['get_model_registry', 'AIServiceProvider', 'ModelRegistry']
