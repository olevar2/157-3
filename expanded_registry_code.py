# Expanded Registry Code for adaptive_indicator_bridge.py
# Replace _build_indicator_registry return statement with this:

return {
            # ====== AI_ENHANCEMENT INDICATORS (42 indicators) ======
            'adaptives': {
                'module': 'engines.ai_enhancement.adaptive_indicators',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'AdaptiveIndicators'
            },
            'performance': {
                'module': 'engines.ai_enhancement.adaptive_indicators',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'IndicatorPerformance'
            },
            'adaptivebridge': {
                'module': 'engines.ai_enhancement.adaptive_indicator_bridge',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'AdaptiveIndicatorBridge'
            },
            'geniusagenttype': {
                'module': 'engines.ai_enhancement.adaptive_indicator_bridge',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'GeniusAgentType'
            },
            'package': {
                'module': 'engines.ai_enhancement.adaptive_indicator_bridge',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'IndicatorPackage'
            },
            'adaptivecoordinator': {
                'module': 'engines.ai_enhancement.adaptive_indicator_coordinator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'AdaptiveIndicatorCoordinator'
            },
            'enhancedadaptivecoordinator': {
                'module': 'engines.ai_enhancement.enhanced_adaptive_coordinator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'EnhancedAdaptiveCoordinator'
            },
            'config': {
                'module': 'engines.ai_enhancement.enhanced_adaptive_coordinator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'IndicatorConfig'
            },
            'performancetracker': {
                'module': 'engines.ai_enhancement.enhanced_adaptive_coordinator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'IndicatorPerformanceTracker'
            },
            'marketregime': {
                'module': 'engines.ai_enhancement.enhanced_adaptive_coordinator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MarketRegime'
            },
            'marketregimedetector': {
                'module': 'engines.ai_enhancement.enhanced_adaptive_coordinator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MarketRegimeDetector'
            },
            'aimodelcoordinatorinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'AIModelCoordinatorInterface'
            },
            'baseagentinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'BaseAgentInterface'
            },
            'decisionmasterinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'DecisionMasterInterface'
            },
            'executionexpertinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'ExecutionExpertInterface'
            },
            'geniusagentintegration': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'GeniusAgentIntegration'
            },
            'marketmicrostructureinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MarketMicrostructureInterface'
            },
            'pairspecialistinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'PairSpecialistInterface'
            },
            'patternmasterinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'PatternMasterInterface'
            },
            'riskgeniusinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'RiskGeniusInterface'
            },
            'sentimentintegrationinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'SentimentIntegrationInterface'
            },
            'sessionexpertinterface': {
                'module': 'engines.ai_enhancement.genius_agent_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'SessionExpertInterface'
            },
            'marketmicrostructure': {
                'module': 'engines.ai_enhancement.market_microstructure_analysis',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MarketMicrostructureAnalysis'
            },
            'microstructurepattern': {
                'module': 'engines.ai_enhancement.market_microstructure_analysis',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MicrostructurePattern'
            },
            'microstructuresignal': {
                'module': 'engines.ai_enhancement.market_microstructure_analysis',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MicrostructureSignal'
            },
            'mlsignal': {
                'module': 'engines.ai_enhancement.ml_signal_generator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MLSignal'
            },
            'mlsignalgenerator': {
                'module': 'engines.ai_enhancement.ml_signal_generator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MLSignalGenerator'
            },
            'modelconfig': {
                'module': 'engines.ai_enhancement.ml_signal_generator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'ModelConfig'
            },
            'signaltype': {
                'module': 'engines.ai_enhancement.ml_signal_generator',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'SignalType'
            },
            'correlationsignal': {
                'module': 'engines.ai_enhancement.multi_asset_correlation',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'CorrelationSignal'
            },
            'multiassetcorrelation': {
                'module': 'engines.ai_enhancement.multi_asset_correlation',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MultiAssetCorrelation'
            },
            'pattern': {
                'module': 'engines.ai_enhancement.pattern_recognition_ai',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'Pattern'
            },
            'patternrecognitionai': {
                'module': 'engines.ai_enhancement.pattern_recognition_ai',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'PatternRecognitionAI'
            },
            'patternsignal': {
                'module': 'engines.ai_enhancement.pattern_recognition_ai',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'PatternSignal'
            },
            'marketregime': {
                'module': 'engines.ai_enhancement.regime_detection_ai',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'MarketRegime'
            },
            'regimedetectionai': {
                'module': 'engines.ai_enhancement.regime_detection_ai',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'RegimeDetectionAI'
            },
            'regimesignal': {
                'module': 'engines.ai_enhancement.regime_detection_ai',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'RegimeSignal'
            },
            'riskassessmentai': {
                'module': 'engines.ai_enhancement.risk_assessment_ai',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'RiskAssessmentAI'
            },
            'sentimentintegration': {
                'module': 'engines.ai_enhancement.sentiment_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'SentimentIntegration'
            },
            'sentimentsignal': {
                'module': 'engines.ai_enhancement.sentiment_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'SentimentSignal'
            },
            'sentimentsource': {
                'module': 'engines.ai_enhancement.sentiment_integration',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'SentimentSource'
            },
            'signalconfidenceai': {
                'module': 'engines.ai_enhancement.signal_confidence_ai',
                'category': 'ai_enhancement',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS],
                'priority': 2,
                'class_name': 'SignalConfidenceAI'
            },

            # ====== CORE_MOMENTUM INDICATORS (13 indicators) ======
            'macd': {
                'module': 'engines.core_momentum.MACD',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MACD'
            },
            'macddata': {
                'module': 'engines.core_momentum.MACD',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MACDData'
            },
            'macdresult': {
                'module': 'engines.core_momentum.MACD',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MACDResult'
            },
            'macdsignal': {
                'module': 'engines.core_momentum.MACD',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MACDSignal'
            },
            'rsi': {
                'module': 'engines.core_momentum.RSI',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'RSI'
            },
            'rsiresult': {
                'module': 'engines.core_momentum.RSI',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'RSIResult'
            },
            'rsisignal': {
                'module': 'engines.core_momentum.RSI',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'RSISignal'
            },
            'smoothingmethod': {
                'module': 'engines.core_momentum.RSI',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'SmoothingMethod'
            },
            'stochastic': {
                'module': 'engines.core_momentum.Stochastic',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'Stochastic'
            },
            'stochasticdata': {
                'module': 'engines.core_momentum.Stochastic',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'StochasticData'
            },
            'stochasticresult': {
                'module': 'engines.core_momentum.Stochastic',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'StochasticResult'
            },
            'stochasticsignal': {
                'module': 'engines.core_momentum.Stochastic',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'StochasticSignal'
            },
            'stochastictype': {
                'module': 'engines.core_momentum.Stochastic',
                'category': 'core_momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'StochasticType'
            },

            # ====== CORE_TREND INDICATORS (25 indicators) ======
            'adx': {
                'module': 'engines.core_trend.ADX',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'ADX'
            },
            'adxresult': {
                'module': 'engines.core_trend.ADX',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'ADXResult'
            },
            'adxsignal': {
                'module': 'engines.core_trend.ADX',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'ADXSignal'
            },
            'adxsignaltype': {
                'module': 'engines.core_trend.ADX',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'ADXSignalType'
            },
            'trenddirection': {
                'module': 'engines.core_trend.ADX',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'TrendDirection'
            },
            'trendstrength': {
                'module': 'engines.core_trend.ADX',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'TrendStrength'
            },
            'exponentialmovingaverage': {
                'module': 'engines.core_trend.ExponentialMovingAverage',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'ExponentialMovingAverage'
            },
            'cloudcolor': {
                'module': 'engines.core_trend.Ichimoku',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'CloudColor'
            },
            'cloudposition': {
                'module': 'engines.core_trend.Ichimoku',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'CloudPosition'
            },
            'ichimoku': {
                'module': 'engines.core_trend.Ichimoku',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'Ichimoku'
            },
            'ichimokuresult': {
                'module': 'engines.core_trend.Ichimoku',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'IchimokuResult'
            },
            'ichimokusignal': {
                'module': 'engines.core_trend.Ichimoku',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'IchimokuSignal'
            },
            'ichimokusignaltype': {
                'module': 'engines.core_trend.Ichimoku',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'IchimokuSignalType'
            },
            'simplemovingaverage': {
                'module': 'engines.core_trend.SimpleMovingAverage',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'SimpleMovingAverage'
            },
            'madata': {
                'module': 'engines.core_trend.SMA_EMA',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'MAData'
            },
            'maresult': {
                'module': 'engines.core_trend.SMA_EMA',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'MAResult'
            },
            'masignal': {
                'module': 'engines.core_trend.SMA_EMA',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'MASignal'
            },
            'matype': {
                'module': 'engines.core_trend.SMA_EMA',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'MAType'
            },
            'movingaverages': {
                'module': 'engines.core_trend.SMA_EMA',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'MovingAverages'
            },
            'supertrend': {
                'module': 'engines.core_trend.SuperTrend',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'SuperTrend'
            },
            'supertrenddata': {
                'module': 'engines.core_trend.SuperTrend',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'SuperTrendData'
            },
            'supertrendresult': {
                'module': 'engines.core_trend.SuperTrend',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'SuperTrendResult'
            },
            'supertrendsignal': {
                'module': 'engines.core_trend.SuperTrend',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'SuperTrendSignal'
            },
            'trenddirection': {
                'module': 'engines.core_trend.SuperTrend',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'TrendDirection'
            },
            'weightedmovingaverage': {
                'module': 'engines.core_trend.WeightedMovingAverage',
                'category': 'core_trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'WeightedMovingAverage'
            },

            # ====== CYCLE INDICATORS (17 indicators) ======
            'alligator': {
                'module': 'engines.cycle.Alligator',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'Alligator'
            },
            'alligatorsignal': {
                'module': 'engines.cycle.Alligator',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'AlligatorSignal'
            },
            'alligatorsignaltype': {
                'module': 'engines.cycle.Alligator',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'AlligatorSignalType'
            },
            'alligatortrend': {
                'module': 'engines.cycle.Alligator',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'AlligatorTrend'
            },
            'cycleperiodidentification': {
                'module': 'engines.cycle.cycle_period_identification',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'CyclePeriodIdentification'
            },
            'dominantcycle': {
                'module': 'engines.cycle.dominant_cycle_analysis',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DominantCycleAnalysis'
            },
            'fishersignal': {
                'module': 'engines.cycle.FisherTransform',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FisherSignal'
            },
            'fishersignaltype': {
                'module': 'engines.cycle.FisherTransform',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FisherSignalType'
            },
            'fishertransform': {
                'module': 'engines.cycle.FisherTransform',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FisherTransform'
            },
            'fishertrend': {
                'module': 'engines.cycle.FisherTransform',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FisherTrend'
            },
            'hurstexponent': {
                'module': 'engines.cycle.HurstExponent',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HurstExponent'
            },
            'hurstsignal': {
                'module': 'engines.cycle.HurstExponent',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HurstSignal'
            },
            'hurstsignaltype': {
                'module': 'engines.cycle.HurstExponent',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HurstSignalType'
            },
            'marketregime': {
                'module': 'engines.cycle.HurstExponent',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MarketRegime'
            },
            'hurstexponent': {
                'module': 'engines.cycle.hurst_exponent',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HurstExponent'
            },
            'marketregimedetection': {
                'module': 'engines.cycle.market_regime_detection',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MarketRegimeDetection'
            },
            'phase': {
                'module': 'engines.cycle.phase_analysis',
                'category': 'cycle',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PhaseAnalysis'
            },

            # ====== DIVERGENCE INDICATORS (5 indicators) ======
            'hiddendivergencedetector': {
                'module': 'engines.divergence.hidden_divergence_detector',
                'category': 'divergence',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'HiddenDivergenceDetector'
            },
            'momentumdivergencescanner': {
                'module': 'engines.divergence.momentum_divergence_scanner',
                'category': 'divergence',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MomentumDivergenceScanner'
            },
            'multitimeframedivergence': {
                'module': 'engines.divergence.multi_timeframe_divergence',
                'category': 'divergence',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MultiTimeframeDivergenceAnalyzer'
            },
            'timeframeconfig': {
                'module': 'engines.divergence.multi_timeframe_divergence',
                'category': 'divergence',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'TimeframeConfig'
            },
            'pricevolumedivergence': {
                'module': 'engines.divergence.price_volume_divergence',
                'category': 'divergence',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'PriceVolumeDivergence'
            },

            # ====== ELLIOTT_WAVE INDICATORS (16 indicators) ======
            'fibonaccilevel': {
                'module': 'engines.elliott_wave.fibonacci_wave_projections',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'FibonacciLevel'
            },
            'fibonacciwaveprojections': {
                'module': 'engines.elliott_wave.fibonacci_wave_projections',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'FibonacciWaveProjections'
            },
            'projectioncluster': {
                'module': 'engines.elliott_wave.fibonacci_wave_projections',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'ProjectionCluster'
            },
            'projectionmethod': {
                'module': 'engines.elliott_wave.fibonacci_wave_projections',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'ProjectionMethod'
            },
            'waveprojection': {
                'module': 'engines.elliott_wave.fibonacci_wave_projections',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WaveProjection'
            },
            'wavetype': {
                'module': 'engines.elliott_wave.fibonacci_wave_projections',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WaveType'
            },
            'impulsivecorrectiveclassifier': {
                'module': 'engines.elliott_wave.impulse_corrective_classifier',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'ImpulsiveCorrectiveClassifier'
            },
            'wave': {
                'module': 'engines.elliott_wave.impulse_corrective_classifier',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WaveAnalysis'
            },
            'waveclassification': {
                'module': 'engines.elliott_wave.impulse_corrective_classifier',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WaveClassification'
            },
            'wavedegree': {
                'module': 'engines.elliott_wave.impulse_corrective_classifier',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WaveDegree'
            },
            'wavepoint': {
                'module': 'engines.elliott_wave.impulse_corrective_classifier',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WavePoint'
            },
            'wavesegment': {
                'module': 'engines.elliott_wave.impulse_corrective_classifier',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WaveSegment'
            },
            'wavestructure': {
                'module': 'engines.elliott_wave.impulse_corrective_classifier',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WaveStructure'
            },
            'elliottwavepattern': {
                'module': 'engines.elliott_wave.wave_count_calculator',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'ElliottWavePattern'
            },
            'enhancedelliottwave': {
                'module': 'engines.elliott_wave.wave_count_calculator',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'EnhancedElliottWaveCalculator'
            },
            'wavepoint': {
                'module': 'engines.elliott_wave.wave_count_calculator',
                'category': 'elliott_wave',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WavePoint'
            },

            # ====== FIBONACCI INDICATORS (23 indicators) ======
            'confluencearea': {
                'module': 'engines.fibonacci.ConfluenceDetector',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ConfluenceArea'
            },
            'confluencedetector': {
                'module': 'engines.fibonacci.ConfluenceDetector',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ConfluenceDetector'
            },
            'confluenceresult': {
                'module': 'engines.fibonacci.ConfluenceDetector',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ConfluenceResult'
            },
            'confluencesignal': {
                'module': 'engines.fibonacci.ConfluenceDetector',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ConfluenceSignal'
            },
            'extensionlevel': {
                'module': 'engines.fibonacci.FibonacciExtension',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ExtensionLevel'
            },
            'extensionresult': {
                'module': 'engines.fibonacci.FibonacciExtension',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ExtensionResult'
            },
            'extensiontarget': {
                'module': 'engines.fibonacci.FibonacciExtension',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ExtensionTarget'
            },
            'fibonacciextension': {
                'module': 'engines.fibonacci.FibonacciExtension',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciExtension'
            },
            'fanline': {
                'module': 'engines.fibonacci.FibonacciFan',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FanLine'
            },
            'fanzone': {
                'module': 'engines.fibonacci.FibonacciFan',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FanZone'
            },
            'fibonaccifan': {
                'module': 'engines.fibonacci.FibonacciFan',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciFanIndicator'
            },
            'fibonaccilevel': {
                'module': 'engines.fibonacci.FibonacciRetracement',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciLevel'
            },
            'fibonacciretracement': {
                'module': 'engines.fibonacci.FibonacciRetracement',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciRetracement'
            },
            'fibonaccizone': {
                'module': 'engines.fibonacci.FibonacciRetracement',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciZone'
            },
            'retracementresult': {
                'module': 'engines.fibonacci.FibonacciRetracement',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'RetracementResult'
            },
            'fibonacciarc': {
                'module': 'engines.fibonacci.ProjectionArcCalculator',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciArc'
            },
            'fibonacciprojection': {
                'module': 'engines.fibonacci.ProjectionArcCalculator',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciProjection'
            },
            'projectionarc': {
                'module': 'engines.fibonacci.ProjectionArcCalculator',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ProjectionArcCalculator'
            },
            'projectionresult': {
                'module': 'engines.fibonacci.ProjectionArcCalculator',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ProjectionResult'
            },
            'timeprediction': {
                'module': 'engines.fibonacci.TimeZoneAnalysis',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TimePrediction'
            },
            'timezone': {
                'module': 'engines.fibonacci.TimeZoneAnalysis',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TimeZone'
            },
            'timezone': {
                'module': 'engines.fibonacci.TimeZoneAnalysis',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TimeZoneAnalysis'
            },
            'timezoneresult': {
                'module': 'engines.fibonacci.TimeZoneAnalysis',
                'category': 'fibonacci',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.RISK_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TimeZoneResult'
            },

            # ====== FRACTAL INDICATORS (30 indicators) ======
            'attractorpoint': {
                'module': 'engines.fractal.chaos_theory_indicators',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'AttractorPoint'
            },
            'chaossignal': {
                'module': 'engines.fractal.chaos_theory_indicators',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ChaosSignal'
            },
            'chaostheorys': {
                'module': 'engines.fractal.chaos_theory_indicators',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ChaosTheoryIndicators'
            },
            'fractalbreakout': {
                'module': 'engines.fractal.fractal_breakout',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalBreakoutIndicator'
            },
            'base': {
                'module': 'engines.fractal.fractal_breakout',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'IndicatorBase'
            },
            'fractalchannel': {
                'module': 'engines.fractal.fractal_channel',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalChannelIndicator'
            },
            'fractalchaososcillator': {
                'module': 'engines.fractal.fractal_chaos_oscillator',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalChaosOscillator'
            },
            'fractalcorrelationdimension': {
                'module': 'engines.fractal.fractal_correlation_dimension',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalCorrelationDimension'
            },
            'fractaldimension': {
                'module': 'engines.fractal.fractal_dimension_calculator',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalDimensionCalculator'
            },
            'fractalefficiencyratio': {
                'module': 'engines.fractal.fractal_efficiency_ratio',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalEfficiencyRatio'
            },
            'fractalenergy': {
                'module': 'engines.fractal.fractal_energy_indicator',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalEnergyIndicator'
            },
            'fractalmarkethypothesis': {
                'module': 'engines.fractal.fractal_market_hypothesis',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalMarketHypothesis'
            },
            'fractalmarketprofile': {
                'module': 'engines.fractal.fractal_market_profile',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalMarketProfile'
            },
            'fractalmomentumoscillator': {
                'module': 'engines.fractal.fractal_momentum_oscillator',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalMomentumOscillator'
            },
            'base': {
                'module': 'engines.fractal.fractal_momentum_oscillator',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'IndicatorBase'
            },
            'fractalvolume': {
                'module': 'engines.fractal.fractal_volume_analysis',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalVolumeAnalysis'
            },
            'base': {
                'module': 'engines.fractal.fractal_volume_analysis',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'IndicatorBase'
            },
            'fractalwave': {
                'module': 'engines.fractal.fractal_wave_counter',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalWave'
            },
            'fractalwavecounter': {
                'module': 'engines.fractal.fractal_wave_counter',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalWaveCounter'
            },
            'wavedegree': {
                'module': 'engines.fractal.fractal_wave_counter',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'WaveDegree'
            },
            'wavetype': {
                'module': 'engines.fractal.fractal_wave_counter',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'WaveType'
            },
            'fractaladaptivemovingaverage': {
                'module': 'engines.fractal.frama',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalAdaptiveMovingAverage'
            },
            'hurstexponent': {
                'module': 'engines.fractal.hurst_exponent',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HurstExponentCalculator'
            },
            'fractalconfig': {
                'module': 'engines.fractal.implementation_template',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalIndicatorConfig'
            },
            'fractaltemplate': {
                'module': 'engines.fractal.implementation_template',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FractalIndicatorTemplate'
            },
            'mandelbrotfractal': {
                'module': 'engines.fractal.mandelbrot_fractal',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MandelbrotFractalIndicator'
            },
            'multifractaldfa': {
                'module': 'engines.fractal.mfdfa',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MultiFractalDFA'
            },
            'patternsignature': {
                'module': 'engines.fractal.self_similarity_detector',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PatternSignature'
            },
            'selfsimilaritydetector': {
                'module': 'engines.fractal.self_similarity_detector',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SelfSimilarityDetector'
            },
            'selfsimilaritysignal': {
                'module': 'engines.fractal.self_similarity_detector',
                'category': 'fractal',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.PATTERN_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SelfSimilaritySignal'
            },

            # ====== GANN INDICATORS (25 indicators) ======
            'gannangle': {
                'module': 'engines.gann.GannAnglesCalculator',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannAngle'
            },
            'gannangles': {
                'module': 'engines.gann.GannAnglesCalculator',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannAnglesCalculator'
            },
            'ganncalculationresult': {
                'module': 'engines.gann.GannAnglesCalculator',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannCalculationResult'
            },
            'ganngrid': {
                'module': 'engines.gann.GannGrid',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannGridIndicator'
            },
            'gridline': {
                'module': 'engines.gann.GannGrid',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GridLine'
            },
            'gridnode': {
                'module': 'engines.gann.GannGrid',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GridNode'
            },
            'gridzone': {
                'module': 'engines.gann.GannGrid',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GridZone'
            },
            'gannpattern': {
                'module': 'engines.gann.GannPatternDetector',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannPattern'
            },
            'gannpatterndetector': {
                'module': 'engines.gann.GannPatternDetector',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannPatternDetector'
            },
            'patternresult': {
                'module': 'engines.gann.GannPatternDetector',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PatternResult'
            },
            'patternsignal': {
                'module': 'engines.gann.GannPatternDetector',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PatternSignal'
            },
            'gannangle': {
                'module': 'engines.gann.gann_fan_lines',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannAngle'
            },
            'gannfan': {
                'module': 'engines.gann.gann_fan_lines',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannFanAnalysis'
            },
            'gannfanlines': {
                'module': 'engines.gann.gann_fan_lines',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannFanLines'
            },
            'gannline': {
                'module': 'engines.gann.gann_fan_lines',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannLine'
            },
            'gannsquareofnine': {
                'module': 'engines.gann.gann_square_of_nine',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannSquareOfNine'
            },
            'square': {
                'module': 'engines.gann.gann_square_of_nine',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SquareAnalysis'
            },
            'squaredirection': {
                'module': 'engines.gann.gann_square_of_nine',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SquareDirection'
            },
            'squarelevel': {
                'module': 'engines.gann.gann_square_of_nine',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SquareLevel'
            },
            'squarepoint': {
                'module': 'engines.gann.gann_square_of_nine',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SquarePoint'
            },
            'timesquare': {
                'module': 'engines.gann.gann_square_of_nine',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TimeSquare'
            },
            'ganntimecycles': {
                'module': 'engines.gann.gann_time_cycles',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannTimeCycles'
            },
            'ganntimesignal': {
                'module': 'engines.gann.gann_time_cycles',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannTimeSignal'
            },
            'timecycle': {
                'module': 'engines.gann.gann_time_cycles',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TimeCycle'
            },
            'pricetimerelationships': {
                'module': 'engines.gann.price_time_relationships',
                'category': 'gann',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PriceTimeRelationships'
            },

            # ====== ML_ADVANCED INDICATORS (10 indicators) ======
            'advancedmlengine': {
                'module': 'engines.ml_advanced.advanced_ml_engine',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'AdvancedMLEngine'
            },
            'deeplearningpredictor': {
                'module': 'engines.ml_advanced.advanced_ml_engine',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'DeepLearningPredictor'
            },
            'ensemblemodel': {
                'module': 'engines.ml_advanced.advanced_ml_engine',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'EnsembleModel'
            },
            'featureengineer': {
                'module': 'engines.ml_advanced.advanced_ml_engine',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'FeatureEngineer'
            },
            'inferencecache': {
                'module': 'engines.ml_advanced.advanced_ml_engine',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'InferenceCache'
            },
            'modelmonitor': {
                'module': 'engines.ml_advanced.advanced_ml_engine',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'ModelMonitor'
            },
            'compositesignal': {
                'module': 'engines.ml_advanced.custom_ai_composite_indicator',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'CompositeSignal'
            },
            'customaicomposite': {
                'module': 'engines.ml_advanced.custom_ai_composite_indicator',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'CustomAICompositeIndicator'
            },
            'geneticalgorithmoptimizer': {
                'module': 'engines.ml_advanced.genetic_algorithm_optimizer',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'GeneticAlgorithmOptimizer'
            },
            'neuralnetworkpredictor': {
                'module': 'engines.ml_advanced.neural_network_predictor',
                'category': 'ml_advanced',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.RISK_GENIUS],
                'priority': 2,
                'class_name': 'NeuralNetworkPredictor'
            },

            # ====== MOMENTUM INDICATORS (24 indicators) ======
            'awesomeoscillator': {
                'module': 'engines.momentum.awesome_oscillator',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'AwesomeOscillator'
            },
            'commoditychannelindex': {
                'module': 'engines.momentum.cci',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'CommodityChannelIndex'
            },
            'chandemomentumoscillator': {
                'module': 'engines.momentum.chande_momentum_oscillator',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'ChandeMomentumOscillator'
            },
            'commoditychannelindex': {
                'module': 'engines.momentum.commodity_channel_index',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'CommodityChannelIndex'
            },
            'correlationmatrix': {
                'module': 'engines.momentum.correlation_momentum',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'CorrelationMatrix'
            },
            'dynamiccorrelation': {
                'module': 'engines.momentum.correlation_momentum',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'DynamicCorrelationIndicator'
            },
            'momentummetrics': {
                'module': 'engines.momentum.correlation_momentum',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MomentumMetrics'
            },
            'relativemomentum': {
                'module': 'engines.momentum.correlation_momentum',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'RelativeMomentumIndicator'
            },
            'detrendedpriceoscillator': {
                'module': 'engines.momentum.detrended_price_oscillator',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'DetrendedPriceOscillator'
            },
            'knowsurething': {
                'module': 'engines.momentum.know_sure_thing',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'KnowSureThing'
            },
            'movingaverageconvergencedivergence': {
                'module': 'engines.momentum.macd',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MovingAverageConvergenceDivergence'
            },
            'mfi': {
                'module': 'engines.momentum.mfi',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'Mfi'
            },
            'moneyflowindex': {
                'module': 'engines.momentum.mfi',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MoneyFlowIndex'
            },
            'momentum': {
                'module': 'engines.momentum.momentum',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'MomentumIndicator'
            },
            'percentagepriceoscillator': {
                'module': 'engines.momentum.percentage_price_oscillator',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'PercentagePriceOscillator'
            },
            'rateofchange': {
                'module': 'engines.momentum.roc',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'RateOfChange'
            },
            'relativestrengthindex': {
                'module': 'engines.momentum.rsi',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'RelativeStrengthIndex'
            },
            'stochasticoscillator': {
                'module': 'engines.momentum.stochastic',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'StochasticOscillator'
            },
            'trix': {
                'module': 'engines.momentum.trix',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'TRIX'
            },
            'trix': {
                'module': 'engines.momentum.trix_new',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'TRIX'
            },
            'trix': {
                'module': 'engines.momentum.trix_old',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'TRIX'
            },
            'truestrengthindex': {
                'module': 'engines.momentum.true_strength_index',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'TrueStrengthIndex'
            },
            'ultimateoscillator': {
                'module': 'engines.momentum.ultimate_oscillator',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'UltimateOscillator'
            },
            'williamsr': {
                'module': 'engines.momentum.williams_r',
                'category': 'momentum',
                'agents': [GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'WilliamsR'
            },

            # ====== PATTERN INDICATORS (88 indicators) ======
            'abandonedbabypatternengine': {
                'module': 'engines.pattern.abandoned_baby_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'AbandonedBabyPatternEngine'
            },
            'abandonedbabysignal': {
                'module': 'engines.pattern.abandoned_baby_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'AbandonedBabySignal'
            },
            'beltholdpattern': {
                'module': 'engines.pattern.belt_hold_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'BeltHoldPattern'
            },
            'beltholdresult': {
                'module': 'engines.pattern.belt_hold_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'BeltHoldResult'
            },
            'beltholdtype': {
                'module': 'engines.pattern.belt_hold_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'BeltHoldType'
            },
            'candledata': {
                'module': 'engines.pattern.belt_hold_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'CandleData'
            },
            'candledata': {
                'module': 'engines.pattern.dark_cloud_cover_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'CandleData'
            },
            'darkcloudcoverpattern': {
                'module': 'engines.pattern.dark_cloud_cover_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DarkCloudCoverPattern'
            },
            'darkcloudresult': {
                'module': 'engines.pattern.dark_cloud_cover_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DarkCloudResult'
            },
            'darkcloudtype': {
                'module': 'engines.pattern.dark_cloud_cover_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DarkCloudType'
            },
            'dojirecognitionengine': {
                'module': 'engines.pattern.doji_recognition',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DojiRecognitionEngine'
            },
            'dojirecognitionresult': {
                'module': 'engines.pattern.doji_recognition',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DojiRecognitionResult'
            },
            'dojisignal': {
                'module': 'engines.pattern.doji_recognition',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DojiSignal'
            },
            'dojitype': {
                'module': 'engines.pattern.doji_recognition',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DojiType'
            },
            'dojirecognitionengine': {
                'module': 'engines.pattern.doji_recognition_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DojiRecognitionEngine'
            },
            'dojirecognitionresult': {
                'module': 'engines.pattern.doji_recognition_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DojiRecognitionResult'
            },
            'dojisignal': {
                'module': 'engines.pattern.doji_recognition_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DojiSignal'
            },
            'dojitype': {
                'module': 'engines.pattern.doji_recognition_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'DojiType'
            },
            'elliottwave': {
                'module': 'engines.pattern.elliott_wave_analysis',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ElliottWaveAnalysis'
            },
            'elliottwavepattern': {
                'module': 'engines.pattern.elliott_wave_analysis',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ElliottWavePattern'
            },
            'wavedegree': {
                'module': 'engines.pattern.elliott_wave_analysis',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'WaveDegree'
            },
            'wavepoint': {
                'module': 'engines.pattern.elliott_wave_analysis',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'WavePoint'
            },
            'wavetype': {
                'module': 'engines.pattern.elliott_wave_analysis',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'WaveType'
            },
            'candlestickdata': {
                'module': 'engines.pattern.engulfing_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'CandlestickData'
            },
            'engulfingpatternresult': {
                'module': 'engines.pattern.engulfing_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'EngulfingPatternResult'
            },
            'engulfingpatternscanner': {
                'module': 'engines.pattern.engulfing_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'EngulfingPatternScanner'
            },
            'engulfingsignal': {
                'module': 'engines.pattern.engulfing_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'EngulfingSignal'
            },
            'engulfingtype': {
                'module': 'engines.pattern.engulfing_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'EngulfingType'
            },
            'candlestickdata': {
                'module': 'engines.pattern.engulfing_pattern_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'CandlestickData'
            },
            'engulfingpatternresult': {
                'module': 'engines.pattern.engulfing_pattern_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'EngulfingPatternResult'
            },
            'engulfingpatternscanner': {
                'module': 'engines.pattern.engulfing_pattern_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'EngulfingPatternScanner'
            },
            'engulfingsignal': {
                'module': 'engines.pattern.engulfing_pattern_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'EngulfingSignal'
            },
            'engulfingtype': {
                'module': 'engines.pattern.engulfing_pattern_fixed',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'EngulfingType'
            },
            'fibonacci': {
                'module': 'engines.pattern.fibonacci_retracement_extension',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciAnalysis'
            },
            'fibonaccilevel': {
                'module': 'engines.pattern.fibonacci_retracement_extension',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciLevel'
            },
            'fibonacciretracementextension': {
                'module': 'engines.pattern.fibonacci_retracement_extension',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciRetracementExtension'
            },
            'fibonaccitype': {
                'module': 'engines.pattern.fibonacci_retracement_extension',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'FibonacciType'
            },
            'swingpoint': {
                'module': 'engines.pattern.fibonacci_retracement_extension',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SwingPoint'
            },
            'trenddirection': {
                'module': 'engines.pattern.fibonacci_retracement_extension',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TrendDirection'
            },
            'gannanglestimecycles': {
                'module': 'engines.pattern.gann_angles_time_cycles',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'GannAnglesTimeCycles'
            },
            'hammerdetectorresult': {
                'module': 'engines.pattern.hammer_hanging_man',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HammerDetectorResult'
            },
            'hammerhangingmandetector': {
                'module': 'engines.pattern.hammer_hanging_man',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HammerHangingManDetector'
            },
            'hammersignal': {
                'module': 'engines.pattern.hammer_hanging_man',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HammerSignal'
            },
            'hammertype': {
                'module': 'engines.pattern.hammer_hanging_man',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HammerType'
            },
            'haramicandlestick': {
                'module': 'engines.pattern.harami_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HaramiCandlestick'
            },
            'haramipatternidentifier': {
                'module': 'engines.pattern.harami_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HaramiPatternIdentifier'
            },
            'haramipatternresult': {
                'module': 'engines.pattern.harami_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HaramiPatternResult'
            },
            'haramisignal': {
                'module': 'engines.pattern.harami_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HaramiSignal'
            },
            'haramitype': {
                'module': 'engines.pattern.harami_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HaramiType'
            },
            'harmonicpattern': {
                'module': 'engines.pattern.harmonic_pattern_detector',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HarmonicPattern'
            },
            'harmonicpatterndetector': {
                'module': 'engines.pattern.harmonic_pattern_detector',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HarmonicPatternDetector'
            },
            'harmonicpoint': {
                'module': 'engines.pattern.harmonic_pattern_detector',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HarmonicPoint'
            },
            'highwavecandledetector': {
                'module': 'engines.pattern.high_wave_candle',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HighWaveCandleDetector'
            },
            'highwavecandlepattern': {
                'module': 'engines.pattern.high_wave_candle',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HighWaveCandlePattern'
            },
            'invertedhammershootingstardetector': {
                'module': 'engines.pattern.inverted_hammer_shooting_star',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'InvertedHammerShootingStarDetector'
            },
            'invertedhammershootingstarpattern': {
                'module': 'engines.pattern.inverted_hammer_shooting_star',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'InvertedHammerShootingStarPattern'
            },
            'candledata': {
                'module': 'engines.pattern.japanese_candlestick_patterns',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'CandleData'
            },
            'japanesecandlestickpatterns': {
                'module': 'engines.pattern.japanese_candlestick_patterns',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'JapaneseCandlestickPatterns'
            },
            'patternresult': {
                'module': 'engines.pattern.japanese_candlestick_patterns',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PatternResult'
            },
            'patterntype': {
                'module': 'engines.pattern.japanese_candlestick_patterns',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PatternType'
            },
            'kickingpatternengine': {
                'module': 'engines.pattern.kicking_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'KickingPatternEngine'
            },
            'kickingsignal': {
                'module': 'engines.pattern.kicking_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'KickingSignal'
            },
            'longleggeddojidetector': {
                'module': 'engines.pattern.long_legged_doji',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'LongLeggedDojiDetector'
            },
            'longleggeddojipattern': {
                'module': 'engines.pattern.long_legged_doji',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'LongLeggedDojiPattern'
            },
            'marubozudetector': {
                'module': 'engines.pattern.marubozu',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MarubozuDetector'
            },
            'marubozupattern': {
                'module': 'engines.pattern.marubozu',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MarubozuPattern'
            },
            'matchingpatternengine': {
                'module': 'engines.pattern.matching_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MatchingPatternEngine'
            },
            'matchingsignal': {
                'module': 'engines.pattern.matching_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MatchingSignal'
            },
            'candledata': {
                'module': 'engines.pattern.piercing_line_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'CandleData'
            },
            'piercinglinepattern': {
                'module': 'engines.pattern.piercing_line_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PiercingLinePattern'
            },
            'piercinglineresult': {
                'module': 'engines.pattern.piercing_line_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PiercingLineResult'
            },
            'piercinglinetype': {
                'module': 'engines.pattern.piercing_line_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'PiercingLineType'
            },
            'soldierspatternengine': {
                'module': 'engines.pattern.soldiers_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SoldiersPatternEngine'
            },
            'soldierssignal': {
                'module': 'engines.pattern.soldiers_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SoldiersSignal'
            },
            'spinningtopdetector': {
                'module': 'engines.pattern.spinning_top',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SpinningTopDetector'
            },
            'spinningtoppattern': {
                'module': 'engines.pattern.spinning_top',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SpinningTopPattern'
            },
            'starpatternengine': {
                'module': 'engines.pattern.star_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'StarPatternEngine'
            },
            'starsignal': {
                'module': 'engines.pattern.star_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'StarSignal'
            },
            'threeinsidepatternengine': {
                'module': 'engines.pattern.three_inside_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ThreeInsidePatternEngine'
            },
            'threeinsidesignal': {
                'module': 'engines.pattern.three_inside_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ThreeInsideSignal'
            },
            'threelinestrikepatternengine': {
                'module': 'engines.pattern.three_line_strike_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ThreeLineStrikePatternEngine'
            },
            'threelinestrikesignal': {
                'module': 'engines.pattern.three_line_strike_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ThreeLineStrikeSignal'
            },
            'threeoutsidepatternengine': {
                'module': 'engines.pattern.three_outside_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ThreeOutsidePatternEngine'
            },
            'threeoutsidesignal': {
                'module': 'engines.pattern.three_outside_pattern',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ThreeOutsideSignal'
            },
            'candledata': {
                'module': 'engines.pattern.tweezer_patterns',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'CandleData'
            },
            'tweezerpatterns': {
                'module': 'engines.pattern.tweezer_patterns',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TweezerPatterns'
            },
            'tweezerresult': {
                'module': 'engines.pattern.tweezer_patterns',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TweezerResult'
            },
            'tweezertype': {
                'module': 'engines.pattern.tweezer_patterns',
                'category': 'pattern',
                'agents': [GeniusAgentType.PATTERN_MASTER, GeniusAgentType.DECISION_MASTER, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'TweezerType'
            },

            # ====== PIVOT INDICATORS (5 indicators) ======
            'pivotlevel': {
                'module': 'engines.pivot.PivotPointCalculator',
                'category': 'pivot',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'PivotLevel'
            },
            'pivotpoint': {
                'module': 'engines.pivot.PivotPointCalculator',
                'category': 'pivot',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'PivotPointCalculator'
            },
            'pivotpointresult': {
                'module': 'engines.pivot.PivotPointCalculator',
                'category': 'pivot',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'PivotPointResult'
            },
            'pivottype': {
                'module': 'engines.pivot.PivotPointCalculator',
                'category': 'pivot',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'PivotType'
            },
            'timeframe': {
                'module': 'engines.pivot.PivotPointCalculator',
                'category': 'pivot',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.EXECUTION_EXPERT, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'TimeFrame'
            },

            # ====== SENTIMENT INDICATORS (9 indicators) ======
            'newsarticle': {
                'module': 'engines.sentiment.NewsScraper',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'NewsArticle'
            },
            'newsscraper': {
                'module': 'engines.sentiment.NewsScraper',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'NewsScraper'
            },
            'scraperconfig': {
                'module': 'engines.sentiment.NewsScraper',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'ScraperConfig'
            },
            'sentiment': {
                'module': 'engines.sentiment.SentimentAnalyzer',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'SentimentAnalyzer'
            },
            'sentimentconfig': {
                'module': 'engines.sentiment.SentimentAnalyzer',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'SentimentConfig'
            },
            'sentimentdata': {
                'module': 'engines.sentiment.SentimentAnalyzer',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'SentimentData'
            },
            'sentimentscore': {
                'module': 'engines.sentiment.SentimentAnalyzer',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'SentimentScore'
            },
            'sentimentsource': {
                'module': 'engines.sentiment.SentimentAnalyzer',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'SentimentSource'
            },
            'sentimenttype': {
                'module': 'engines.sentiment.SentimentAnalyzer',
                'category': 'sentiment',
                'agents': [GeniusAgentType.SENTIMENT_INTEGRATION_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.DECISION_MASTER],
                'priority': 2,
                'class_name': 'SentimentType'
            },

            # ====== STATISTICAL INDICATORS (14 indicators) ======
            'autocorrelation': {
                'module': 'engines.statistical.autocorrelation',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'AutocorrelationIndicator'
            },
            'betacoefficient': {
                'module': 'engines.statistical.beta_coefficient',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'BetaCoefficientIndicator'
            },
            'cointegration': {
                'module': 'engines.statistical.cointegration',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'CointegrationIndicator'
            },
            'correlation': {
                'module': 'engines.statistical.correlation_analysis',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'CorrelationAnalysis'
            },
            'correlationcoefficient': {
                'module': 'engines.statistical.correlation_coefficient',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'CorrelationCoefficientIndicator'
            },
            'linearregression': {
                'module': 'engines.statistical.linear_regression',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'LinearRegressionIndicator'
            },
            'linearregressionchannels': {
                'module': 'engines.statistical.linear_regression_channels',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'LinearRegressionChannels'
            },
            'rsquared': {
                'module': 'engines.statistical.r_squared',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'RSquaredIndicator'
            },
            'kurtosis': {
                'module': 'engines.statistical.skewness_kurtosis',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'KurtosisIndicator'
            },
            'skewness': {
                'module': 'engines.statistical.skewness_kurtosis',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'SkewnessIndicator'
            },
            'standarddeviation': {
                'module': 'engines.statistical.standard_deviation',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'StandardDeviationIndicator'
            },
            'standarddeviationchannels': {
                'module': 'engines.statistical.standard_deviation_channels',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'StandardDeviationChannels'
            },
            'varianceratio': {
                'module': 'engines.statistical.variance_ratio',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'VarianceRatioIndicator'
            },
            'zscore': {
                'module': 'engines.statistical.z_score',
                'category': 'statistical',
                'agents': [GeniusAgentType.AI_MODEL_COORDINATOR, GeniusAgentType.RISK_GENIUS, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'ZScoreIndicator'
            },

            # ====== TREND INDICATORS (20 indicators) ======
            'aroon': {
                'module': 'engines.trend.aroon_indicator',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'AroonIndicator'
            },
            'averagetruerange': {
                'module': 'engines.trend.average_true_range',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'AverageTrueRange'
            },
            'bollingerbands': {
                'module': 'engines.trend.bollinger_bands',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'BollingerBands'
            },
            'directionalmovementsystem': {
                'module': 'engines.trend.directional_movement_system',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'DirectionalMovementSystem'
            },
            'donchianchannels': {
                'module': 'engines.trend.donchian_channels',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'DonchianChannels'
            },
            'keltner': {
                'module': 'engines.trend.keltner_channels',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'KeltnerAnalysis'
            },
            'keltnerchanneldata': {
                'module': 'engines.trend.keltner_channels',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'KeltnerChannelData'
            },
            'keltnerchannelstate': {
                'module': 'engines.trend.keltner_channels',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'KeltnerChannelState'
            },
            'keltnerchannels': {
                'module': 'engines.trend.keltner_channels',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'KeltnerChannels'
            },
            'keltnerconfig': {
                'module': 'engines.trend.keltner_channels',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'KeltnerConfig'
            },
            'keltnertrenddirection': {
                'module': 'engines.trend.keltner_channels',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'KeltnerTrendDirection'
            },
            'movingaveragetype': {
                'module': 'engines.trend.keltner_channels',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'MovingAverageType'
            },
            'parabolicsar': {
                'module': 'engines.trend.parabolic_sar',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'ParabolicSar'
            },
            'vortex': {
                'module': 'engines.trend.vortex_indicator',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'VortexAnalysis'
            },
            'vortexconfig': {
                'module': 'engines.trend.vortex_indicator',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'VortexConfig'
            },
            'vortexdata': {
                'module': 'engines.trend.vortex_indicator',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'VortexData'
            },
            'vortex': {
                'module': 'engines.trend.vortex_indicator',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'VortexIndicator'
            },
            'vortexmomentum': {
                'module': 'engines.trend.vortex_indicator',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'VortexMomentum'
            },
            'vortexsignaltype': {
                'module': 'engines.trend.vortex_indicator',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'VortexSignalType'
            },
            'vortextrendstate': {
                'module': 'engines.trend.vortex_indicator',
                'category': 'trend',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.DECISION_MASTER, GeniusAgentType.PAIR_SPECIALIST],
                'priority': 2,
                'class_name': 'VortexTrendState'
            },

            # ====== VOLATILITY INDICATORS (9 indicators) ======
            'chaikinvolatility': {
                'module': 'engines.volatility.chaikin_volatility',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'ChaikinVolatility'
            },
            'historicalvolatility': {
                'module': 'engines.volatility.historical_volatility',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'HistoricalVolatility'
            },
            'keltnerchannels': {
                'module': 'engines.volatility.keltner_channels',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'KeltnerChannels'
            },
            'massindex': {
                'module': 'engines.volatility.mass_index',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'MassIndex'
            },
            'relativevolatilityindex': {
                'module': 'engines.volatility.rvi',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'RelativeVolatilityIndex'
            },
            'sdchannelresult': {
                'module': 'engines.volatility.standard_deviation_channels',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SDChannelResult'
            },
            'sdchannelsignal': {
                'module': 'engines.volatility.standard_deviation_channels',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'SDChannelSignal'
            },
            'standarddeviationchannels': {
                'module': 'engines.volatility.standard_deviation_channels',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'StandardDeviationChannels'
            },
            'volatilityindex': {
                'module': 'engines.volatility.volatility_index',
                'category': 'volatility',
                'agents': [GeniusAgentType.RISK_GENIUS, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.AI_MODEL_COORDINATOR],
                'priority': 2,
                'class_name': 'VolatilityIndex'
            },

            # ====== VOLUME INDICATORS (48 indicators) ======
            'accumulationdistributionline': {
                'module': 'engines.volume.accumulation_distribution',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'AccumulationDistributionLine'
            },
            'accumulationdistributionsignal': {
                'module': 'engines.volume.accumulation_distribution',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'AccumulationDistributionSignal'
            },
            'chaikinmoneyflow': {
                'module': 'engines.volume.chaikin_money_flow',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'ChaikinMoneyFlow'
            },
            'chaikinmoneyflowsignal': {
                'module': 'engines.volume.chaikin_money_flow',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'ChaikinMoneyFlowSignal'
            },
            'easeofmovement': {
                'module': 'engines.volume.ease_of_movement',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'EaseOfMovement'
            },
            'forceindex': {
                'module': 'engines.volume.force_index',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'ForceIndex'
            },
            'institutionalflowdetector': {
                'module': 'engines.volume.institutional_flow_detector',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'InstitutionalFlowDetector'
            },
            'institutionalflowsignal': {
                'module': 'engines.volume.institutional_flow_detector',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'InstitutionalFlowSignal'
            },
            'klingeroscillator': {
                'module': 'engines.volume.klinger_oscillator',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'KlingerOscillator'
            },
            'liquidityflow': {
                'module': 'engines.volume.liquidity_flow_indicator',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'LiquidityFlowIndicator'
            },
            'liquidityflowsignal': {
                'module': 'engines.volume.liquidity_flow_indicator',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'LiquidityFlowSignal'
            },
            'marketmicrostructure': {
                'module': 'engines.volume.market_microstructure_indicator',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'MarketMicrostructureIndicator'
            },
            'marketmicrostructuresignal': {
                'module': 'engines.volume.market_microstructure_indicator',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'MarketMicrostructureSignal'
            },
            'negativevolumeindex': {
                'module': 'engines.volume.negative_volume_index',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'NegativeVolumeIndex'
            },
            'onbalancevolume': {
                'module': 'engines.volume.obv',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'OnBalanceVolume'
            },
            'orderflowimbalance': {
                'module': 'engines.volume.OrderFlowImbalance',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'Orderflowimbalance'
            },
            'volumedata': {
                'module': 'engines.volume.OrderFlowImbalance',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeData'
            },
            'blocktradesignal': {
                'module': 'engines.volume.order_flow_block_trade_detector',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'BlockTradeSignal'
            },
            'orderflowblocktradedetector': {
                'module': 'engines.volume.order_flow_block_trade_detector',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'OrderFlowBlockTradeDetector'
            },
            'orderflowsequence': {
                'module': 'engines.volume.order_flow_sequence_analyzer',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'OrderFlowSequenceAnalyzer'
            },
            'orderflowsequencesignal': {
                'module': 'engines.volume.order_flow_sequence_analyzer',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'OrderFlowSequenceSignal'
            },
            'positivevolumeindex': {
                'module': 'engines.volume.positive_volume_index',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'PositiveVolumeIndex'
            },
            'pricevolumerank': {
                'module': 'engines.volume.price_volume_rank',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'PriceVolumeRank'
            },
            'smartmoneys': {
                'module': 'engines.volume.SmartMoneyIndicators',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'Smartmoneyindicators'
            },
            'volumedata': {
                'module': 'engines.volume.SmartMoneyIndicators',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeData'
            },
            'tickvolumes': {
                'module': 'engines.volume.TickVolumeIndicators',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'Tickvolumeindicators'
            },
            'volumedata': {
                'module': 'engines.volume.TickVolumeIndicators',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeData'
            },
            'tickvolume': {
                'module': 'engines.volume.tick_volume_analyzer',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'TickVolumeAnalyzer'
            },
            'tickvolumesignal': {
                'module': 'engines.volume.tick_volume_analyzer',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'TickVolumeSignal'
            },
            'volumedata': {
                'module': 'engines.volume.VolumeProfiles',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeData'
            },
            'volumeprofiles': {
                'module': 'engines.volume.VolumeProfiles',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'Volumeprofiles'
            },
            'volumedata': {
                'module': 'engines.volume.VolumeSpreadAnalysis',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeData'
            },
            'volumespread': {
                'module': 'engines.volume.VolumeSpreadAnalysis',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'Volumespreadanalysis'
            },
            'volumebreakoutdetector': {
                'module': 'engines.volume.volume_breakout_detector',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeBreakoutDetector'
            },
            'volumebreakoutsignal': {
                'module': 'engines.volume.volume_breakout_detector',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeBreakoutSignal'
            },
            'volumedelta': {
                'module': 'engines.volume.volume_delta_indicator',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeDeltaIndicator'
            },
            'volumedeltasignal': {
                'module': 'engines.volume.volume_delta_indicator',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeDeltaSignal'
            },
            'volumeoscillator': {
                'module': 'engines.volume.volume_oscillator',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeOscillator'
            },
            'vpt': {
                'module': 'engines.volume.volume_price_trend',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VPTAnalysis'
            },
            'vptconfig': {
                'module': 'engines.volume.volume_price_trend',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VPTConfig'
            },
            'vptsignaltype': {
                'module': 'engines.volume.volume_price_trend',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VPTSignalType'
            },
            'vpttrendstate': {
                'module': 'engines.volume.volume_price_trend',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VPTTrendState'
            },
            'vptvolumeflow': {
                'module': 'engines.volume.volume_price_trend',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VPTVolumeFlow'
            },
            'volumepricetrend': {
                'module': 'engines.volume.volume_price_trend',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumePriceTrend'
            },
            'volumerateofchange': {
                'module': 'engines.volume.volume_rate_of_change',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeRateOfChange'
            },
            'marketdepthsignal': {
                'module': 'engines.volume.volume_weighted_market_depth',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'MarketDepthSignal'
            },
            'volumeweightedmarketdepth': {
                'module': 'engines.volume.volume_weighted_market_depth',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeWeightedMarketDepthIndicator'
            },
            'volumeweightedaverageprice': {
                'module': 'engines.volume.vwap',
                'category': 'volume',
                'agents': [GeniusAgentType.SESSION_EXPERT, GeniusAgentType.MARKET_MICROSTRUCTURE_GENIUS, GeniusAgentType.EXECUTION_EXPERT],
                'priority': 2,
                'class_name': 'VolumeWeightedAveragePrice'
            },

        }
