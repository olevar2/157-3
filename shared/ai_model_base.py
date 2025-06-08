"""
Enhanced AI Model Base Class for Platform3
Provides common functionality for all AI models in the platform
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import hashlib
from abc import ABC, abstractmethod

# Import Platform3 shared components
from shared.platform3_logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem


class EnhancedAIModelBase(ABC):
    """
    Enhanced base class for all AI models in Platform3
    
    Provides:
    - Standardized initialization and lifecycle management
    - Common logging and error handling
    - Performance tracking and metrics
    - Model metadata and versioning
    - Configuration management
    """
    
    def __init__(self, model_name: str, version: str, description: str):
        self.model_name = model_name
        self.version = version
        self.description = description
        self.model_id = self._generate_model_id()
        
        # Initialize logging and error handling
        self.logger = Platform3Logger(f"AI.{model_name}")
        self.error_system = Platform3ErrorSystem(f"AI.{model_name}")
        
        # Model state and metrics
        self.is_initialized = False
        self.is_running = False
        self.initialization_time = None
        self.last_prediction_time = None
        
        # Performance metrics
        self.performance_metrics = {
            "total_predictions": 0,
            "total_errors": 0,
            "average_prediction_time_ms": 0.0,
            "accuracy_score": 0.0,
            "confidence_score": 0.0,
            "model_health": "unknown"
        }
        
        # Configuration
        self.config = {}
        self.features_config = {}
        
        self.logger.info(f"Initialized AI model: {model_name} v{version}")
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID based on name, version and timestamp"""
        unique_string = f"{self.model_name}_{self.version}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    @abstractmethod
    async def start(self):
        """Start the AI model service - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the AI model service - must be implemented by subclasses"""
        pass
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AI model with configuration"""
        try:
            self.logger.info(f"Initializing AI model: {self.model_name}")
            start_time = datetime.now()
            
            if config:
                self.config.update(config)
            
            # Set default configuration
            self._set_default_config()
            
            # Perform model-specific initialization
            await self._model_specific_initialization()
            
            self.is_initialized = True
            self.initialization_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"AI model {self.model_name} initialized in {self.initialization_time:.3f}s")
            
        except Exception as e:
            self.error_system.handle_error(e, "initialize")
            raise
    
    async def _model_specific_initialization(self):
        """Override this method for model-specific initialization logic"""
        pass
    
    def _set_default_config(self):
        """Set default configuration values"""
        default_config = {
            "max_prediction_time_ms": 1000,
            "min_confidence_threshold": 0.5,
            "enable_performance_tracking": True,
            "enable_health_monitoring": True,
            "log_predictions": False
        }
        
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    async def predict(self, input_data: Any) -> Dict[str, Any]:
        """
        Make a prediction using the AI model
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate input
            if not self.is_initialized:
                raise ValueError("Model not initialized. Call initialize() first.")
            
            # Perform prediction
            result = await self._make_prediction(input_data)
            
            # Calculate performance metrics
            prediction_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_performance_metrics(prediction_time_ms, result)
            
            # Add metadata to result
            result.update({
                "model_id": self.model_id,
                "model_name": self.model_name,
                "version": self.version,
                "prediction_time_ms": prediction_time_ms,
                "timestamp": datetime.now().isoformat()
            })
            
            self.last_prediction_time = datetime.now()
            
            if self.config.get("log_predictions", False):
                self.logger.debug(f"Prediction made in {prediction_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            self.performance_metrics["total_errors"] += 1
            self.error_system.handle_error(e, "predict", {"input_data": str(input_data)[:200]})
            return {
                "error": str(e),
                "model_id": self.model_id,
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
    
    @abstractmethod
    async def _make_prediction(self, input_data: Any) -> Dict[str, Any]:
        """Override this method to implement actual prediction logic"""
        pass
    
    def _update_performance_metrics(self, prediction_time_ms: float, result: Dict[str, Any]):
        """Update performance metrics after each prediction"""
        self.performance_metrics["total_predictions"] += 1
        
        # Update average prediction time
        total_predictions = self.performance_metrics["total_predictions"]
        current_avg = self.performance_metrics["average_prediction_time_ms"]
        new_avg = ((current_avg * (total_predictions - 1)) + prediction_time_ms) / total_predictions
        self.performance_metrics["average_prediction_time_ms"] = new_avg
        
        # Update confidence score if available
        if "confidence" in result:
            confidence = result["confidence"]
            current_confidence = self.performance_metrics["confidence_score"]
            new_confidence = ((current_confidence * (total_predictions - 1)) + confidence) / total_predictions
            self.performance_metrics["confidence_score"] = new_confidence
        
        # Update model health
        self._update_model_health()
    
    def _update_model_health(self):
        """Update overall model health status"""
        error_rate = self.performance_metrics["total_errors"] / max(1, self.performance_metrics["total_predictions"])
        avg_time = self.performance_metrics["average_prediction_time_ms"]
        confidence = self.performance_metrics["confidence_score"]
        
        if error_rate < 0.01 and avg_time < 100 and confidence > 0.8:
            self.performance_metrics["model_health"] = "excellent"
        elif error_rate < 0.05 and avg_time < 500 and confidence > 0.6:
            self.performance_metrics["model_health"] = "good"
        elif error_rate < 0.10 and avg_time < 1000 and confidence > 0.4:
            self.performance_metrics["model_health"] = "fair"
        else:
            self.performance_metrics["model_health"] = "poor"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "description": self.description,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "initialization_time": self.initialization_time,
            "last_prediction_time": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            "performance_metrics": self.performance_metrics.copy(),
            "config": self.config.copy()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        return {
            "model_name": self.model_name,
            "model_health": self.performance_metrics["model_health"],
            "total_predictions": self.performance_metrics["total_predictions"],
            "total_errors": self.performance_metrics["total_errors"],
            "error_rate": self.performance_metrics["total_errors"] / max(1, self.performance_metrics["total_predictions"]),
            "average_prediction_time_ms": self.performance_metrics["average_prediction_time_ms"],
            "confidence_score": self.performance_metrics["confidence_score"],
            "last_prediction_time": self.last_prediction_time.isoformat() if self.last_prediction_time else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        try:
            # Basic health indicators
            health_status = {
                "model_name": self.model_name,
                "is_initialized": self.is_initialized,
                "is_running": self.is_running,
                "health": self.performance_metrics["model_health"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Additional health checks
            if self.is_initialized:
                # Check if model is responsive
                test_result = await self._perform_health_test()
                health_status.update(test_result)
            
            return health_status
            
        except Exception as e:
            self.error_system.handle_error(e, "health_check")
            return {
                "model_name": self.model_name,
                "health": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _perform_health_test(self) -> Dict[str, Any]:
        """Override this method to implement model-specific health tests"""
        return {
            "response_test": "passed",
            "performance_test": "passed" if self.performance_metrics["average_prediction_time_ms"] < 1000 else "warning"
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            "total_predictions": 0,
            "total_errors": 0,
            "average_prediction_time_ms": 0.0,
            "accuracy_score": 0.0,
            "confidence_score": 0.0,
            "model_health": "unknown"
        }
        self.logger.info(f"Performance metrics reset for {self.model_name}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update model configuration"""
        self.config.update(new_config)
        self.logger.info(f"Configuration updated for {self.model_name}")


class MLModelMixin:
    """
    Mixin class providing additional ML-specific functionality
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.feature_importance = {}
    
    async def train(self, training_data: Any, validation_data: Optional[Any] = None) -> Dict[str, Any]:
        """Train the ML model - override in subclasses"""
        raise NotImplementedError("Training method must be implemented by subclass")
    
    async def evaluate(self, test_data: Any) -> Dict[str, Any]:
        """Evaluate model performance - override in subclasses"""
        raise NotImplementedError("Evaluation method must be implemented by subclass")
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        raise NotImplementedError("Save method must be implemented by subclass")
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        raise NotImplementedError("Load method must be implemented by subclass")
