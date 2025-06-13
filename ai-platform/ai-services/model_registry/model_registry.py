"""
Platform3 AI Model Registry
Enterprise-grade model management and coordination system
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import importlib.util
import sys

class ModelType(Enum):
    TRADING_SCALPING = "trading_scalping"
    TRADING_DAYTRADING = "trading_daytrading"
    TRADING_SWING = "trading_swing"
    INTELLIGENT_AGENT = "intelligent_agent"
    MARKET_ANALYSIS = "market_analysis"
    ADAPTIVE_LEARNING = "adaptive_learning"

class ModelStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    DEPRECATED = "deprecated"
    ERROR = "error"

@dataclass
class ModelInfo:
    """Model information structure"""
    model_id: str
    name: str
    model_type: ModelType
    path: str
    description: str
    version: str
    status: ModelStatus
    performance_metrics: Dict[str, float]
    dependencies: List[str]
    created_at: datetime
    updated_at: datetime
    last_used: Optional[datetime] = None
    config: Optional[Dict[str, Any]] = None

class AIModelRegistry:
    """
    Centralized registry for all AI models in Platform3
    Provides discovery, management, and coordination capabilities
    """
    
    def __init__(self, ai_platform_path: str = None):
        if ai_platform_path is None:
            ai_platform_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "ai-platform")
        self.ai_platform_path = Path(ai_platform_path)
        self.models_path = self.ai_platform_path / "ai-models"
        self.registry_file = self.ai_platform_path / "model_registry.json"
        self.models: Dict[str, ModelInfo] = {}
        self.logger = logging.getLogger(__name__)
        
        # Auto-discover and register models
        self._discover_models()
    
    def _discover_models(self):
        """Auto-discover all models in the AI platform"""
        try:
            # Discover trading models
            self._discover_trading_models()
            
            # Discover intelligent agents
            self._discover_intelligent_agents()
            
            # Discover market analysis models
            self._discover_market_analysis_models()
            
            # Discover adaptive learning models
            self._discover_adaptive_learning_models()
            
            self.logger.info(f"Discovered {len(self.models)} AI models")
            
        except Exception as e:
            self.logger.error(f"Error discovering models: {e}")
    
    def _discover_trading_models(self):
        """Discover trading models"""
        trading_path = self.models_path / "trading-models"
        
        # Scalping models
        scalping_path = trading_path / "scalping"
        if scalping_path.exists():
            for model_dir in scalping_path.iterdir():
                if model_dir.is_dir():
                    self._register_model_from_path(
                        model_dir, 
                        ModelType.TRADING_SCALPING,
                        f"scalping_{model_dir.name}"
                    )
        
        # Day trading models
        daytrading_path = trading_path / "daytrading"
        if daytrading_path.exists():
            for model_file in daytrading_path.glob("*.py"):
                if not model_file.name.startswith("__"):
                    self._register_model_from_file(
                        model_file,
                        ModelType.TRADING_DAYTRADING,
                        f"daytrading_{model_file.stem}"
                    )
        
        # Swing trading models
        swing_path = trading_path / "swing"
        if swing_path.exists():
            for model_file in swing_path.glob("*.py"):
                if not model_file.name.startswith("__"):
                    self._register_model_from_file(
                        model_file,
                        ModelType.TRADING_SWING,
                        f"swing_{model_file.stem}"
                    )
    
    def _discover_intelligent_agents(self):
        """Discover intelligent agent models"""
        agents_path = self.models_path / "intelligent-agents"
        
        if agents_path.exists():
            for agent_dir in agents_path.iterdir():
                if agent_dir.is_dir():
                    self._register_model_from_path(
                        agent_dir,
                        ModelType.INTELLIGENT_AGENT,
                        f"agent_{agent_dir.name.replace('-', '_')}"
                    )
    
    def _discover_market_analysis_models(self):
        """Discover market analysis models"""
        analysis_path = self.models_path / "market-analysis"
        
        if analysis_path.exists():
            for model_dir in analysis_path.iterdir():
                if model_dir.is_dir():
                    self._register_model_from_path(
                        model_dir,
                        ModelType.MARKET_ANALYSIS,
                        f"analysis_{model_dir.name.replace('-', '_')}"
                    )
    
    def _discover_adaptive_learning_models(self):
        """Discover adaptive learning models"""
        learning_path = self.models_path / "adaptive-learning"
        
        if learning_path.exists():
            for model_dir in learning_path.iterdir():
                if model_dir.is_dir():
                    self._register_model_from_path(
                        model_dir,
                        ModelType.ADAPTIVE_LEARNING,
                        f"learning_{model_dir.name.replace('-', '_')}"
                    )
    
    def _register_model_from_path(self, model_path: Path, model_type: ModelType, model_id: str):
        """Register a model from a directory path"""
        try:
            # Look for main model file
            main_files = ["model.py", "main.py", "__init__.py"]
            model_file = None
            
            for file_name in main_files:
                candidate = model_path / file_name
                if candidate.exists():
                    model_file = candidate
                    break
            
            if not model_file:
                # Look for any Python file
                py_files = list(model_path.glob("*.py"))
                if py_files:
                    model_file = py_files[0]
            
            if model_file:
                self._register_model_from_file(model_file, model_type, model_id)
            
        except Exception as e:
            self.logger.warning(f"Could not register model from {model_path}: {e}")
    
    def _register_model_from_file(self, model_file: Path, model_type: ModelType, model_id: str):
        """Register a model from a Python file"""
        try:
            # Extract model information
            description = self._extract_model_description(model_file)
            
            model_info = ModelInfo(
                model_id=model_id,
                name=model_file.stem.replace('_', ' ').title(),
                model_type=model_type,
                path=str(model_file),
                description=description,
                version="1.0.0",
                status=ModelStatus.ACTIVE,
                performance_metrics={},
                dependencies=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.models[model_id] = model_info
            
        except Exception as e:
            self.logger.warning(f"Could not register model from {model_file}: {e}")
    
    def _extract_model_description(self, model_file: Path) -> str:
        """Extract description from model file docstring"""
        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for docstring
            lines = content.split('\n')
            in_docstring = False
            description_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    if in_docstring:
                        break
                    in_docstring = True
                    # Extract text after opening quotes
                    desc_part = line[3:].strip()
                    if desc_part and not desc_part.endswith('"""') and not desc_part.endswith("'''"):
                        description_lines.append(desc_part)
                elif in_docstring:
                    if line.endswith('"""') or line.endswith("'''"):
                        # Extract text before closing quotes
                        desc_part = line[:-3].strip()
                        if desc_part:
                            description_lines.append(desc_part)
                        break
                    else:
                        description_lines.append(line)
            
            if description_lines:
                return ' '.join(description_lines)
            else:
                return f"AI model: {model_file.stem}"
                
        except Exception:
            return f"AI model: {model_file.stem}"
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID"""
        return self.models.get(model_id)
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelInfo]:
        """List all models, optionally filtered by type"""
        if model_type:
            return [model for model in self.models.values() if model.model_type == model_type]
        return list(self.models.values())
    
    def get_models_by_status(self, status: ModelStatus) -> List[ModelInfo]:
        """Get models by status"""
        return [model for model in self.models.values() if model.status == status]
    
    def update_model_status(self, model_id: str, status: ModelStatus):
        """Update model status"""
        if model_id in self.models:
            self.models[model_id].status = status
            self.models[model_id].updated_at = datetime.now()
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Update model performance metrics"""
        if model_id in self.models:
            self.models[model_id].performance_metrics.update(metrics)
            self.models[model_id].updated_at = datetime.now()
    
    def load_model(self, model_id: str):
        """Dynamically load a model"""
        model_info = self.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(model_id, model_info.path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[model_id] = module
            spec.loader.exec_module(module)
            
            # Update last used time
            self.models[model_id].last_used = datetime.now()
            
            return module
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {e}")
            self.update_model_status(model_id, ModelStatus.ERROR)
            raise
    
    def save_registry(self):
        """Save registry to file"""
        try:
            registry_data = {
                model_id: {
                    **asdict(model_info),
                    'model_type': model_info.model_type.value,
                    'status': model_info.status.value,
                    'created_at': model_info.created_at.isoformat(),
                    'updated_at': model_info.updated_at.isoformat(),
                    'last_used': model_info.last_used.isoformat() if model_info.last_used else None
                }
                for model_id, model_info in self.models.items()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
    
    def load_registry(self):
        """Load registry from file"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for model_id, model_data in registry_data.items():
                    model_data['model_type'] = ModelType(model_data['model_type'])
                    model_data['status'] = ModelStatus(model_data['status'])
                    model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                    model_data['updated_at'] = datetime.fromisoformat(model_data['updated_at'])
                    if model_data['last_used']:
                        model_data['last_used'] = datetime.fromisoformat(model_data['last_used'])
                    
                    self.models[model_id] = ModelInfo(**model_data)
                
        except Exception as e:
            self.logger.error(f"Error loading registry: {e}")
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of registry"""
        summary = {
            'total_models': len(self.models),
            'models_by_type': {},
            'models_by_status': {},
            'active_models': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        for model in self.models.values():
            # Count by type
            type_key = model.model_type.value
            summary['models_by_type'][type_key] = summary['models_by_type'].get(type_key, 0) + 1
            
            # Count by status
            status_key = model.status.value
            summary['models_by_status'][status_key] = summary['models_by_status'].get(status_key, 0) + 1
            
            if model.status == ModelStatus.ACTIVE:
                summary['active_models'] += 1
        
        return summary

# Global registry instance
_registry = None

def get_registry() -> AIModelRegistry:
    """Get global model registry instance"""
    global _registry
    if _registry is None:
        _registry = AIModelRegistry()
    return _registry

if __name__ == "__main__":
    # Test the registry
    registry = AIModelRegistry()
    summary = registry.get_registry_summary()
    print(f"AI Model Registry Summary:")
    print(f"Total Models: {summary['total_models']}")
    print(f"Active Models: {summary['active_models']}")
    print(f"Models by Type: {summary['models_by_type']}")
    print(f"Models by Status: {summary['models_by_status']}")
    
    # Save registry
    registry.save_registry()
