"""
Platform3 Model Registry

Centralized registry for all AI/ML models in the trading platform.
Provides unified model discovery, loading, versioning, and management.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import importlib.util

logger = logging.getLogger(__name__)

# COMPLETE PROFESSIONAL TRADING MODEL ECOSYSTEM
# Existing Models (Enhanced) + New Genius Models = Complete Trading Intelligence

EXISTING_MODELS = {
    # Core Trading Models (KEEP + ENHANCE)
    'scalping_lstm': 'M1-M5 price prediction LSTM (507 lines - ENHANCED)',
    'tick_classifier': 'Next tick direction prediction (641 lines - ENHANCED)', 
    'spread_predictor': 'Bid/ask spread forecasting for optimal entry timing',
    'noise_filter': 'Market noise filtering for clean signals',
    'scalping_ensemble': 'Multi-model ensemble for scalping decisions',
    'swing_trading': 'H4 swing pattern recognition for 1-5 day trades',
    'sentiment_analyzer': 'Real-time market sentiment analysis',
    'elliott_wave': 'Automated Elliott wave pattern detection',
    'autoencoder_features': 'Deep feature extraction and anomaly detection',
    'currency_pair_intelligence': 'Pair-specific trading characteristics',
    'online_learning': 'Continuous model adaptation to market changes',
    'model_deployment': 'Production model serving and monitoring'
}

NEW_GENIUS_MODELS = {
    # Professional Trading Intelligence (NEW)
    'indicator_expert': 'GENIUS: Professional indicator selection for each pair/timeframe',
    'strategy_expert': 'GENIUS: Strategy development based on price action observation',
    'simulation_expert': 'GENIUS: Historical backtesting and side development',
    'decision_master': 'GENIUS: Professional trading decision making',
    'risk_genius': 'GENIUS: Advanced risk management and position sizing',
    'pair_specialist': 'GENIUS: Individual pair personality analysis',
    'session_expert': 'GENIUS: Session-specific trading optimization',
    'pattern_master': 'GENIUS: Advanced pattern recognition and completion',
    'execution_expert': 'GENIUS: Optimal trade execution and timing'
}

# Combined Professional Model Types
ESSENTIAL_MODEL_TYPES = {**EXISTING_MODELS, **NEW_GENIUS_MODELS}

# Trading Performance Categories for Daily Profit Focus
TRADING_CATEGORIES = {
    'scalping': ['scalping_lstm', 'tick_classifier', 'spread_predictor', 'noise_filter', 'scalping_ensemble'],
    'day_trading': ['swing_trading', 'sentiment_analyzer', 'elliott_wave'],
    'risk_management': ['autoencoder_features', 'currency_pair_intelligence'],
    'infrastructure': ['online_learning', 'model_deployment']
}

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    name: str
    version: str
    model_type: str
    description: str
    created_at: datetime
    updated_at: datetime
    performance_metrics: Dict[str, float]
    dependencies: List[str]
    input_shape: Optional[tuple]
    output_shape: Optional[tuple]
    training_data_version: Optional[str]


class BaseModel(ABC):
    """Base interface that all models must implement"""
    
    def __init__(self, model_path: Path, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.metadata = self._load_metadata()
        self._model = None
        
    @abstractmethod
    def load(self) -> None:
        """Load the trained model weights"""
        pass
        
    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Make predictions on input data"""
        pass
        
    @abstractmethod
    def train(self, training_data: Any, validation_data: Any = None) -> Dict[str, float]:
        """Train the model and return performance metrics"""
        pass
        
    @abstractmethod
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
        
    def _load_metadata(self) -> ModelMetadata:
        """Load model metadata from metadata.json"""
        metadata_path = self.model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return ModelMetadata(**data)
        else:
            # Return default metadata if file doesn't exist
            return ModelMetadata(
                name=self.model_path.name,
                version="1.0.0",
                model_type="unknown",
                description="No description available",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                performance_metrics={},
                dependencies=[],
                input_shape=None,
                output_shape=None,
                training_data_version=None
            )
    
    def save_metadata(self) -> None:
        """Save model metadata to metadata.json"""
        metadata_path = self.model_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.__dict__, f, indent=2, default=str)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.copy()
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update model performance metrics"""
        self.metadata.performance_metrics.update(metrics)
        self.metadata.updated_at = datetime.now()
        self.save_metadata()


class ModelRegistry:
    """
    Central registry for all Platform3 models.
    
    Provides model discovery, loading, versioning, and management capabilities.
    """
    
    def __init__(self, models_root: Optional[Path] = None):
        self.models_root = models_root or Path(__file__).parent
        self._loaded_models: Dict[str, BaseModel] = {}
        self._model_cache: Dict[str, Type[BaseModel]] = {}
        
        # Discover all available models
        self._discover_models()
    
    def _discover_models(self) -> None:
        """Discover all available models in the models directory"""
        logger.info("Discovering models...")
        
        for model_dir in self.models_root.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                try:
                    self._register_model(model_dir)
                except Exception as e:
                    logger.warning(f"Failed to register model {model_dir.name}: {e}")
    
    def _register_model(self, model_path: Path) -> None:
        """Register a model from its directory"""
        model_name = model_path.name
        
        # Check if model has required files
        if not (model_path / "__init__.py").exists():
            logger.warning(f"Model {model_name} missing __init__.py")
            return
            
        if not (model_path / "model.py").exists():
            logger.warning(f"Model {model_name} missing model.py")
            return
        
        logger.info(f"Registered model: {model_name}")
    
    def list_models(self) -> List[str]:
        """List all available models"""
        models = []
        for model_dir in self.models_root.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                if (model_dir / "__init__.py").exists():
                    models.append(model_dir.name)
        return sorted(models)
    
    def get_model_info(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model"""
        model_path = self.models_root / model_name
        if not model_path.exists():
            return None
            
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return ModelMetadata(**data)
        return None
    
    def load_model(self, model_name: str, version: str = "latest", force_reload: bool = False) -> BaseModel:
        """
        Load a model by name and version
        
        Args:
            model_name: Name of the model to load
            version: Version to load (default: "latest")
            force_reload: Force reload even if already loaded
            
        Returns:
            Loaded model instance
        """
        cache_key = f"{model_name}:{version}"
        
        # Return cached model if available and not forcing reload
        if cache_key in self._loaded_models and not force_reload:
            return self._loaded_models[cache_key]
        
        model_path = self.models_root / model_name
        if not model_path.exists():
            raise ValueError(f"Model {model_name} not found")
        
        # Load model configuration
        config = self._load_model_config(model_path)
        
        # Dynamically import and instantiate the model
        model_instance = self._instantiate_model(model_path, config)
        
        # Load the trained weights
        model_instance.load()
        
        # Cache the loaded model
        self._loaded_models[cache_key] = model_instance
        
        logger.info(f"Loaded model: {model_name} (version: {version})")
        return model_instance
    
    def _load_model_config(self, model_path: Path) -> Dict[str, Any]:
        """Load model configuration from config.yaml"""
        config_path = model_path / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _instantiate_model(self, model_path: Path, config: Dict[str, Any]) -> BaseModel:
        """Dynamically instantiate a model class"""
        model_name = model_path.name
        
        # Import the model module
        spec = importlib.util.spec_from_file_location(
            f"models.{model_name}.model",
            model_path / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the model class (should inherit from BaseModel)
        model_classes = [
            cls for name, cls in module.__dict__.items()
            if isinstance(cls, type) and issubclass(cls, BaseModel) and cls != BaseModel
        ]
        
        if not model_classes:
            raise ValueError(f"No model class found in {model_name}/model.py")
        
        if len(model_classes) > 1:
            logger.warning(f"Multiple model classes found in {model_name}, using the first one")
        
        model_class = model_classes[0]
        return model_class(model_path, config)
    
    def unload_model(self, model_name: str, version: str = "latest") -> None:
        """Unload a model from memory"""
        cache_key = f"{model_name}:{version}"
        if cache_key in self._loaded_models:
            del self._loaded_models[cache_key]
            logger.info(f"Unloaded model: {model_name} (version: {version})")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all loaded models"""
        health_status = {
            "healthy": True,
            "models": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for cache_key, model in self._loaded_models.items():
            model_name = cache_key.split(':')[0]
            try:
                # Basic health check - model should be loaded
                is_healthy = model._model is not None
                health_status["models"][model_name] = {
                    "healthy": is_healthy,
                    "metadata": model.metadata.__dict__
                }
                if not is_healthy:
                    health_status["healthy"] = False
            except Exception as e:
                health_status["models"][model_name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_status["healthy"] = False
        
        return health_status
    
    def get_model_performance(self, model_name: str) -> Dict[str, float]:
        """Get performance metrics for a model"""
        model_info = self.get_model_info(model_name)
        if model_info:
            return model_info.performance_metrics
        return {}
    
    def compare_models(self, model_names: List[str], metric: str = "accuracy") -> Dict[str, float]:
        """Compare models by a specific metric"""
        comparison = {}
        for model_name in model_names:
            performance = self.get_model_performance(model_name)
            if metric in performance:
                comparison[model_name] = performance[metric]
        return comparison


# Global model registry instance
_registry = None

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
