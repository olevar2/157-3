# Registry Function and Component Mapping

## Complete Inventory of Registry Components

### Core Classes and Data Structures

1. **IndicatorMetadata** (dataclass)
   - Fields: name, category, description, parameters, input_types, output_type, version, author, created_at, last_updated, is_real, performance_tier
   - Method: __post_init__()

2. **EnhancedIndicatorRegistry** (class)
   - Private attributes: _indicators, _metadata, _aliases, _categories, _failed_imports, _performance_cache
   - Methods:
     - register_indicator()
     - _validate_indicator_interface()
     - _generate_metadata()
     - _determine_category()
     - _update_category_index()
     - get_indicator()
     - load_module_indicators()
     - _is_valid_indicator()
     - get_metadata()
     - get_categories()
     - get_failed_imports()
     - validate_all()

3. **GeniusAgentType** (Enum)
   - Values: RISK_GENIUS, SESSION_EXPERT, PATTERN_MASTER, EXECUTION_EXPERT, PAIR_SPECIALIST, DECISION_MASTER, AI_MODEL_COORDINATOR, MARKET_MICROSTRUCTURE_GENIUS, SENTIMENT_INTEGRATION_GENIUS

4. **GeniusAgentIntegration** (class)
   - Methods: __init__(), get_indicators(), analyze()

### Global Variables

1. **_enhanced_registry** - Instance of EnhancedIndicatorRegistry
2. **INDICATOR_REGISTRY** - Dict, legacy compatibility interface
3. **AI_AGENTS_REGISTRY** - Dict containing all 9 AI agents

### Core Functions

1. **load_real_indicators()** - Main indicator loading orchestration
2. **load_momentum_indicators()** - Special momentum loading with stub detection
3. **validate_registry()** - Quality control and dummy detection
4. **get_indicator(name)** - Safe indicator retrieval
5. **get_indicator_categories()** - Dynamic categorization
6. **get_ai_agent(agent_name)** - AI agent retrieval
7. **list_ai_agents()** - AI agent listing with metadata
8. **validate_ai_agents()** - AI agent validation

### Import Blocks and Indicator Groups

1. **Pattern Indicators Import Block** (lines ~340-400)
   - Direct imports: DarkCloudCoverPattern, PiercingLinePattern, TweezerPatterns
   - Complete pattern indicators from pattern_indicators_complete

2. **Technical Indicators Import Block** (lines ~350-450)
   - RSI, MACD, Bollinger Bands, Stochastic, CCI
   - Moving Averages: Simple, Exponential, Weighted
   - Donchian Channels

3. **Volume Indicators Import Block**
   - Module loading from volume_indicators_real
   - Volume-weighted calculations

4. **Trend Indicators Import Block**
   - Real implementations: ATR, Parabolic SAR, DMI, Aroon
   - Module loading from trend_indicators_real

5. **Category Module Loading** (lines ~500-600)
   - channel_indicators, statistical_indicators
   - fractal_indicators_complete, real_gann_indicators
   - divergence_indicators_complete, cycle_indicators_complete
   - sentiment_indicators_complete, ml_advanced_indicators_complete
   - elliott_wave_indicators_complete, pivot_indicators_complete
   - fibonacci_indicators_complete

6. **Momentum Indicators Processing** (lines ~650-750)
   - 17 momentum indicators with stub detection
   - Special processing for signal indicators

7. **Pattern Indicators Complete** (lines ~750-850)
   - 24 pattern types with dual naming conventions
   - Both snake_case and PascalCase registration

8. **Fibonacci Indicators** (lines ~850-900)
   - 6 fibonacci indicators auto-registration

9. **Dynamic Category Loading** (lines ~900-950)
   - Loop through category modules
   - Special handling for real_gann_indicators

### AI Agents Registry Structure

Each agent contains:
- type: GeniusAgentType enum value
- class: GeniusAgentIntegration
- model: Specific model version
- max_tokens: Token limit
- description: Agent purpose
- specialization: Area of expertise
- indicators_used: Number of indicators
- status: "active"

### Critical Validation Logic

1. **Anti-Dummy Protection**
   - Check for "dummy" in __name__
   - Prevent trading inaccuracies
   - Error on dummy detection

2. **Interface Validation**
   - Callable verification
   - Calculate method presence for classes
   - Type checking

3. **Registry Consistency**
   - Duplicate detection
   - Alias management
   - Category organization

4. **Error Handling**
   - Import failure tracking
   - Graceful degradation
   - Comprehensive logging

### Sub-Indicator Dependencies

1. **Pattern Sub-Indicators**: 24 pattern types
2. **Momentum Sub-Indicators**: 17 momentum calculations
3. **Technical Sub-Indicators**: Core technical analysis tools
4. **Statistical Sub-Indicators**: 10+ statistical calculations
5. **Fractal Sub-Indicators**: 5 result data classes
6. **Fibonacci Sub-Indicators**: 6 fibonacci tools
7. **Volume Sub-Indicators**: Volume analysis variants
8. **Trend Sub-Indicators**: Trend identification tools

### Naming Conventions

1. **Dual Registration**: Both snake_case and PascalCase
2. **Alias Management**: Alternative names supported
3. **Category Prefixes**: Some indicators use category prefixes
4. **Signal Suffixes**: Special handling for signal indicators

## Preservation Strategy

### Must Not Change:
- AI_AGENTS_REGISTRY structure
- INDICATOR_REGISTRY dict interface
- validate_registry() anti-dummy logic
- All existing import blocks
- Alias mappings
- Error handling patterns

### Can Be Extended:
- EnhancedIndicatorRegistry methods
- Dynamic loading capabilities
- Metadata management
- Performance monitoring

### Implementation Approach:
- Add file-based discovery to existing system
- Maintain all legacy compatibility
- Preserve all validation logic
- Keep all naming conventions
- Extend rather than replace