"""
Model Registry Service - Humanitarian Trading Platform
Central registry for all AI models serving our charitable mission

This service manages all AI models that generate profits for:
- Emergency medical aid for the poor
- Children's surgical procedures 
- Global poverty alleviation
- Food security for struggling families

Author: Platform3 Humanitarian AI Team
Version: 1.0.0 - Dedicated to healing the suffering
"""

import asyncio
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
import hashlib
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of AI models in our humanitarian platform"""
    SCALPING_LSTM = "scalping_lstm"
    SCALPING_ENSEMBLE = "scalping_ensemble"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    PATTERN_RECOGNITION = "pattern_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    ADAPTIVE_LEARNING = "adaptive_learning"
    MARKET_REGIME = "market_regime"
    VOLUME_ANALYSIS = "volume_analysis"

class ModelStatus(Enum):
    """Model deployment status"""
    TRAINING = "training"
    READY = "ready"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ERROR = "error"

class DeploymentStage(Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    deployment_stage: DeploymentStage
    created_at: datetime
    updated_at: datetime
    trained_at: Optional[datetime] = None
    last_prediction: Optional[datetime] = None
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    
    # Trading-specific metrics
    profit_factor: Optional[float] = None
    win_rate: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: int = 0
    
    # Humanitarian impact metrics
    charitable_profit_generated: float = 0.0
    medical_procedures_funded: int = 0
    families_helped: int = 0
    
    # Technical details
    file_path: Optional[str] = None
    model_size_mb: Optional[float] = None
    input_features: List[str] = field(default_factory=list)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Resource requirements
    cpu_usage: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_required: bool = False
    avg_inference_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif isinstance(value, Enum):
                data[key] = value.value
        return data

class ModelRegistry:
    """
    🏥 HUMANITARIAN AI MODEL REGISTRY
    
    Central registry for all AI models serving our charitable mission.
    Tracks model performance and humanitarian impact metrics.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            db_path = os.path.join(project_root, "data", "model_registry.db")
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for fast access
        self._model_cache: Dict[str, ModelMetadata] = {}
        self._load_models_to_cache()
        
        self.logger.info("🚀 Model Registry initialized for humanitarian trading mission")
    
    def _init_database(self):
        """Initialize SQLite database for model storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    deployment_stage TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    trained_at TEXT,
                    last_prediction TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    mse REAL,
                    mae REAL,
                    profit_factor REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER DEFAULT 0,
                    charitable_profit_generated REAL DEFAULT 0.0,
                    medical_procedures_funded INTEGER DEFAULT 0,
                    families_helped INTEGER DEFAULT 0,
                    file_path TEXT,
                    model_size_mb REAL,
                    input_features TEXT,
                    output_schema TEXT,
                    dependencies TEXT,
                    configuration TEXT,
                    cpu_usage REAL,
                    memory_usage_mb REAL,
                    gpu_required INTEGER DEFAULT 0,
                    avg_inference_time_ms REAL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_type ON models(model_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON models(status)
            """)
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _load_models_to_cache(self):
        """Load all models from database to memory cache"""
        with self._get_db_connection() as conn:
            cursor = conn.execute("SELECT * FROM models")
            for row in cursor.fetchall():
                model = self._row_to_model(row)
                self._model_cache[model.model_id] = model
    
    def _row_to_model(self, row: sqlite3.Row) -> ModelMetadata:
        """Convert database row to ModelMetadata"""
        return ModelMetadata(
            model_id=row['model_id'],
            name=row['name'],
            version=row['version'],
            model_type=ModelType(row['model_type']),
            status=ModelStatus(row['status']),
            deployment_stage=DeploymentStage(row['deployment_stage']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            trained_at=datetime.fromisoformat(row['trained_at']) if row['trained_at'] else None,
            last_prediction=datetime.fromisoformat(row['last_prediction']) if row['last_prediction'] else None,
            accuracy=row['accuracy'],
            precision=row['precision'],
            recall=row['recall'],
            f1_score=row['f1_score'],
            mse=row['mse'],
            mae=row['mae'],
            profit_factor=row['profit_factor'],
            win_rate=row['win_rate'],
            sharpe_ratio=row['sharpe_ratio'],
            max_drawdown=row['max_drawdown'],
            total_trades=row['total_trades'] or 0,
            charitable_profit_generated=row['charitable_profit_generated'] or 0.0,
            medical_procedures_funded=row['medical_procedures_funded'] or 0,
            families_helped=row['families_helped'] or 0,
            file_path=row['file_path'],
            model_size_mb=row['model_size_mb'],
            input_features=json.loads(row['input_features']) if row['input_features'] else [],
            output_schema=json.loads(row['output_schema']) if row['output_schema'] else {},
            dependencies=json.loads(row['dependencies']) if row['dependencies'] else [],
            configuration=json.loads(row['configuration']) if row['configuration'] else {},
            cpu_usage=row['cpu_usage'],
            memory_usage_mb=row['memory_usage_mb'],
            gpu_required=bool(row['gpu_required']),
            avg_inference_time_ms=row['avg_inference_time_ms']
        )
    
    def register_model(self, 
                      name: str,
                      model_type: ModelType,
                      version: str = "1.0.0",
                      file_path: Optional[str] = None,
                      configuration: Optional[Dict[str, Any]] = None,
                      dependencies: Optional[List[str]] = None) -> str:
        """
        Register new AI model for humanitarian trading mission
        
        Returns: model_id for tracking
        """
        with self._lock:
            model_id = f"{model_type.value}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            model = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                model_type=model_type,
                status=ModelStatus.READY,
                deployment_stage=DeploymentStage.DEVELOPMENT,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                file_path=file_path,
                configuration=configuration or {},
                dependencies=dependencies or []
            )
            
            # Calculate model size if file exists
            if file_path and Path(file_path).exists():
                model.model_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            
            # Save to database
            self._save_model_to_db(model)
            
            # Update cache
            self._model_cache[model_id] = model
            
            self.logger.info(f"✅ Registered model {name} ({model_id}) for humanitarian mission")
            return model_id
    
    def _save_model_to_db(self, model: ModelMetadata):
        """Save model metadata to database"""
        with self._get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models (
                    model_id, name, version, model_type, status, deployment_stage,
                    created_at, updated_at, trained_at, last_prediction,
                    accuracy, precision, recall, f1_score, mse, mae,
                    profit_factor, win_rate, sharpe_ratio, max_drawdown, total_trades,
                    charitable_profit_generated, medical_procedures_funded, families_helped,
                    file_path, model_size_mb, input_features, output_schema,
                    dependencies, configuration, cpu_usage, memory_usage_mb,
                    gpu_required, avg_inference_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.model_id, model.name, model.version, model.model_type.value,
                model.status.value, model.deployment_stage.value,
                model.created_at.isoformat(), model.updated_at.isoformat(),
                model.trained_at.isoformat() if model.trained_at else None,
                model.last_prediction.isoformat() if model.last_prediction else None,
                model.accuracy, model.precision, model.recall, model.f1_score,
                model.mse, model.mae, model.profit_factor, model.win_rate,
                model.sharpe_ratio, model.max_drawdown, model.total_trades,
                model.charitable_profit_generated, model.medical_procedures_funded,
                model.families_helped, model.file_path, model.model_size_mb,
                json.dumps(model.input_features), json.dumps(model.output_schema),
                json.dumps(model.dependencies), json.dumps(model.configuration),
                model.cpu_usage, model.memory_usage_mb, int(model.gpu_required),
                model.avg_inference_time_ms
            ))
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self._model_cache.get(model_id)
    
    def list_models(self, 
                   model_type: Optional[ModelType] = None,
                   status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List all models with optional filtering"""
        models = list(self._model_cache.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return sorted(models, key=lambda m: m.updated_at, reverse=True)
    
    def update_model_performance(self,
                               model_id: str,
                               performance_metrics: Dict[str, Any],
                               humanitarian_impact: Optional[Dict[str, Any]] = None):
        """Update model performance metrics and humanitarian impact"""
        with self._lock:
            model = self._model_cache.get(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            # Update performance metrics
            for key, value in performance_metrics.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            # Update humanitarian impact
            if humanitarian_impact:
                model.charitable_profit_generated += humanitarian_impact.get('profit_generated', 0.0)
                model.medical_procedures_funded += humanitarian_impact.get('procedures_funded', 0)
                model.families_helped += humanitarian_impact.get('families_helped', 0)
            
            model.updated_at = datetime.now()
            model.last_prediction = datetime.now()
            
            # Save to database
            self._save_model_to_db(model)
            
            self.logger.info(f"📊 Updated performance for model {model_id} - Charitable impact: ${model.charitable_profit_generated:.2f}")
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get comprehensive registry summary for humanitarian mission"""
        models = list(self._model_cache.values())
        
        total_charitable_profit = sum(m.charitable_profit_generated for m in models)
        total_procedures_funded = sum(m.medical_procedures_funded for m in models)
        total_families_helped = sum(m.families_helped for m in models)
        
        return {
            'total_models': len(models),
            'active_models': len([m for m in models if m.status == ModelStatus.ACTIVE]),
            'models_by_type': {
                model_type.value: len([m for m in models if m.model_type == model_type])
                for model_type in ModelType
            },
            'humanitarian_impact': {
                'total_charitable_profit': total_charitable_profit,
                'medical_procedures_funded': total_procedures_funded,
                'families_helped': total_families_helped,
                'average_profit_per_model': total_charitable_profit / len(models) if models else 0
            },
            'performance_summary': {
                'avg_accuracy': sum(m.accuracy for m in models if m.accuracy) / len([m for m in models if m.accuracy]) if any(m.accuracy for m in models) else 0,
                'avg_profit_factor': sum(m.profit_factor for m in models if m.profit_factor) / len([m for m in models if m.profit_factor]) if any(m.profit_factor for m in models) else 0,
                'total_trades': sum(m.total_trades for m in models)
            }
        }
    
    def activate_model(self, model_id: str) -> bool:
        """Activate model for production humanitarian trading"""
        with self._lock:
            model = self._model_cache.get(model_id)
            if not model:
                return False
            
            model.status = ModelStatus.ACTIVE
            model.deployment_stage = DeploymentStage.PRODUCTION
            model.updated_at = datetime.now()
            
            self._save_model_to_db(model)
            
            self.logger.info(f"🚀 Activated model {model_id} for humanitarian trading mission")
            return True
    
    def deactivate_model(self, model_id: str) -> bool:
        """Deactivate model"""
        with self._lock:
            model = self._model_cache.get(model_id)
            if not model:
                return False
            
            model.status = ModelStatus.INACTIVE
            model.updated_at = datetime.now()
            
            self._save_model_to_db(model)
            
            self.logger.info(f"⏸️ Deactivated model {model_id}")
            return True
