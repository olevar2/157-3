# --- START OF FILE indicator_mappings.py ---

"""
This file contains the master mapping of indicators to each of the 9 Genius Agents.
It is imported by the AdaptiveIndicatorBridge to know which indicators to calculate for each agent.
"""

AGENT_INDICATOR_MAPPINGS = {
    "risk_genius": {
        "Physics": ["ThermodynamicEntropyEngine", "QuantumMomentumOracle"],
        "Risk_Statistical": ["BetaCoefficientIndicator", "CorrelationAnalysis", "CorrelationCoefficientIndicator", "Variance", "VarianceRatioIndicator", "ZScoreIndicator", "StandardDeviationIndicator", "SkewnessIndicator", "HurstExponent", "CointegrationIndicator", "AutocorrelationIndicator", "RSquaredIndicator"],
        "Volatility": ["ChaikinVolatility", "HistoricalVolatility", "RelativeVolatilityIndex", "VolatilityIndex", "AverageTrueRange"],
        "Channels": ["BollingerBands", "KeltnerChannels", "StandardDeviationChannels", "DonchianChannels", "LinearRegressionChannels"]
    },
    "session_expert": {
        "Physics": ["BiorhythmMarketSynth", "CrystallographicLatticeDetector"],
        "Time_Cycles": ["GannTimeCycleIndicator", "CyclePeriodIdentification", "DominantCycleAnalysis", "FibonacciTimeZoneIndicator", "MarketRegimeDetection"],
        "Gann_Tools": ["GannAnglesIndicator", "GannFanIndicator", "GannPriceTimeIndicator", "GannSquareIndicator"],
        "Trend_Direction": ["ParabolicSAR", "SuperTrend", "IchimokuIndicator", "AroonIndicator"]
    },
    "pattern_master": {
        "Physics": ["ChaosGeometryPredictor", "NeuralHarmonicResonance"],
        "Candlestick": ["AbandonedBabySignal", "BeltHoldType", "DarkCloudCoverPattern", "DojiPattern", "EngulfingPattern", "HammerPattern", "HaramiType", "HighWaveCandlePattern", "InvertedHammerShootingStarPattern", "KickingSignal", "LongLeggedDojiPattern", "MarubozuPattern", "MatchingSignal", "PiercingLinePattern", "SoldiersSignal", "SpinningTopPattern", "StarSignal", "ThreeInsideSignal", "ThreeLineStrikeSignal", "ThreeOutsideSignal", "TweezerPatterns"],
        "Fractal_Chaos": ["FractalAdaptiveMovingAverage", "FractalBreakoutIndicator", "FractalChannelIndicator", "FractalChaosOscillator", "FractalDimensionIndicator", "FractalEnergyIndicator", "FractalVolumeIndicator", "MandelbrotFractalIndicator"],
        "Fibonacci": ["FibonacciArcIndicator", "FibonacciChannelIndicator", "FibonacciExtensionIndicator", "FibonacciFanIndicator", "FibonacciRetracementIndicator", "FibonacciTimeZoneIndicator"],
        "Wave_Analysis": ["WavePoint", "WaveStructure", "WaveType"]
    },
    "execution_expert": {
        "Physics": ["PhotonicWavelengthAnalyzer", "QuantumMomentumOracle"],
        "Microstructure": ["BidAskSpreadAnalyzer", "LiquidityFlowSignal", "MarketDepthIndicator", "MarketMicrostructureSignal", "OrderFlowImbalance", "OrderFlowSequenceSignal"],
        "Volume_Execution": ["BlockTradeSignal", "InstitutionalFlowSignal", "VolumeDeltaSignal", "VolumeBreakoutSignal"],
        "Price_Action": ["VolumeWeightedAveragePrice", "VWAPIndicator", "EaseOfMovement", "ForceIndex"],
        "Execution_Tools": ["MomentumDivergenceScanner", "PhaseAnalysis", "ConfluenceArea"]
    },
    "pair_specialist": {
        "Physics": ["CrystallographicLatticeDetector", "ThermodynamicEntropyEngine"],
        "Correlation": ["CorrelationMatrixIndicator", "CorrelationAnalysis", "CorrelationCoefficientIndicator", "BetaCoefficientIndicator", "CointegrationIndicator"],
        "Pair_Specific": ["CommodityChannelIndex", "DirectionalMovementSystem", "ADXIndicator", "CCIIndicator"],
        "Volume_Correlation": ["KlingerOscillator", "AccumulationDistribution", "PositiveVolumeIndex"]
    },
    "decision_master": {
        "Physics": ["NeuralHarmonicResonance", "ChaosGeometryPredictor"],
        "Machine_Learning": ["AdvancedMLEngine", "GeneticAlgorithmOptimizer", "NeuralNetworkPredictor"],
        "Confluence_Tools": ["CompositeSignal", "ConfluenceArea", "AttractorPoint"],
        "Decision_Support": ["MarketRegimeDetection", "ZScoreIndicator"]
    },
    "ai_model_coordinator": {
        "Physics": ["QuantumMomentumOracle", "PhotonicWavelengthAnalyzer"],
        "ML_Integration": ["AdvancedMLEngine", "GeneticAlgorithmOptimizer", "NeuralNetworkPredictor"],
        "System_Performance": ["CorrelationMatrixIndicator", "RSquaredIndicator", "VarianceRatioIndicator"],
        "Model_Sync": ["CompositeSignal", "PhaseAnalysis", "FractalAdaptiveMovingAverage"]
    },
    "market_microstructure_genius": {
        "Physics": ["PhotonicWavelengthAnalyzer", "CrystallographicLatticeDetector"],
        "Microstructure_Core": ["BidAskSpreadAnalyzer", "LiquidityFlowSignal", "MarketDepthIndicator", "MarketMicrostructureSignal", "OrderFlowImbalance", "OrderFlowSequenceSignal"],
        "Institutional_Flow": ["BlockTradeSignal", "InstitutionalFlowSignal", "VolumeDeltaSignal"],
        "Order_Book": ["PriceVolumeDivergence", "PriceVolumeRank", "VolumeBreakoutSignal"],
        "High_Frequency": ["TickVolumeSignal", "MassIndex", "FractalMarketProfile"]
    },
    "sentiment_integration_genius": {
        "Physics": ["BiorhythmMarketSynth", "ThermodynamicEntropyEngine"],
        "Sentiment_Core": ["NewsArticle", "SocialMediaPostIndicator"],
        "Behavioral_Finance": ["FisherTransform", "AwesomeOscillatorIndicator"],
        "Market_Psychology": ["DeMarkerIndicator", "BearsPowerIndicator", "BullsPowerIndicator"],
        "Sentiment_Cycles": ["CyclePeriodIdentification", "DominantCycleAnalysis", "GannTimeCycleIndicator"]
    }
}

# --- END OF FILE indicator_mappings.py ---