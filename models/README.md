# Platform3 Model Registry

This directory contains all AI/ML models for the Platform3 trading system. Each model is self-contained with its code, trained weights, configuration, and documentation.

## Model Structure

Each model follows this standardized structure:
```
models/
├── model_name/
│   ├── __init__.py              # Model interface and exports
│   ├── model.py                 # Core model implementation
│   ├── trainer.py               # Training logic
│   ├── predictor.py             # Inference/prediction logic
│   ├── config.yaml              # Model configuration
│   ├── metadata.json            # Model metadata (version, performance, etc.)
│   ├── weights/                 # Trained model weights/artifacts
│   │   ├── latest.pt            # Latest trained weights
│   │   ├── best.pt              # Best performing weights
│   │   └── versions/            # Historical versions
│   ├── tests/                   # Model-specific tests
│   │   ├── test_model.py
│   │   ├── test_trainer.py
│   │   └── test_predictor.py
│   └── docs/                    # Model documentation
│       ├── README.md
│       ├── architecture.md
│       └── performance.md
```

## Available Models

### Core Trading Models
- `scalping_lstm/` - LSTM model for scalping strategies (MIGRATED ✅)
- `tick_classifier/` - Tick movement classification (MIGRATED ✅)
- `spread_predictor/` - Spread prediction model (MIGRATED ✅)
- `scalping_ensemble/` - Ensemble scalping strategies (MIGRATED ✅)
- `noise_filter/` - Market noise filtering (MIGRATED ✅)
- `swing_trading/` - Swing trading models with trainers (MIGRATED ✅)
- `currency_pair_intelligence/` - Currency pair specialization (MIGRATED ✅)

### Feature & Analysis Models
- `autoencoder_features/` - Feature extraction and dimensionality reduction (MIGRATED ✅)
- `elliott_wave/` - Elliott Wave pattern recognition (MIGRATED ✅)
- `sentiment_analyzer/` - Market sentiment analysis (MIGRATED ✅)

### Infrastructure Models
- `online_learning/` - Real-time adaptive learning (MIGRATED ✅)
- `model_deployment/` - Model deployment and management (MIGRATED ✅)

### Future Models (Planned)
- `market_regime_detector/` - Market condition detection
- `economic_calendar_ai/` - Economic event impact prediction
- `strategy_evolution/` - Strategy optimization and evolution
- `session_optimizer/` - Trading session optimization
- `risk_manager/` - Risk assessment and management

## Model Registry

The model registry (`registry.py`) provides:
- Automatic model discovery
- Version management
- Performance tracking
- Model loading/unloading
- Health checks
- A/B testing support

## Usage

```python
from models import ModelRegistry

# Load a model
registry = ModelRegistry()
scalping_model = registry.load_model('scalping_lstm', version='latest')

# Get predictions
predictions = scalping_model.predict(market_data)

# List available models
available_models = registry.list_models()
```

## Development Guidelines

1. **Self-Contained**: Each model directory should be completely independent
2. **Standardized Interface**: All models implement the same base interface
3. **Versioned**: All model weights and code are versioned
4. **Tested**: Each model has comprehensive tests
5. **Documented**: Clear documentation for architecture and usage
6. **Configurable**: Model behavior controlled via config files
