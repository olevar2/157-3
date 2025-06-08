"""
Fix the registry by creating a proper combined registry
"""

# Get the existing working registry mappings 
current_mapping = '''    # Trend indicators
    'adaptive_rsi': AdaptiveRSI,
    'regime_adaptive_ma': RegimeAdaptiveMA,
    'smart_money_index': SmartMoneyIndex,
    'correlation_matrix': CorrelationMatrix,
    'dynamic_correlation': DynamicCorrelation,
    
    # Volatility indicators
    'chaikin_volatility': ChaikinVolatility,
    'historical_volatility': HistoricalVolatility,
    'relative_volatility_index': RelativeVolatilityIndex,
    'volatility_index': VolatilityIndex,
    'mass_index': MassIndex,
    
    # Channel indicators
    'sd_channel_signal': SDChannelSignal,
    'keltner_channels': KeltnerChannels,
    'linear_regression_channels': LinearRegressionChannels,
    'standard_deviation_channels': StandardDeviationChannels,
    
    # Statistical indicators
    'autocorrelation_indicator': AutocorrelationIndicator,
    'beta_coefficient_indicator': BetaCoefficientIndicator,
    'correlation_coefficient_indicator': CorrelationCoefficientIndicator,
    'cointegration_indicator': CointegrationIndicator,
    'linear_regression_indicator': LinearRegressionIndicator,
    'r_squared_indicator': RSquaredIndicator,
    'skewness_indicator': SkewnessIndicator,
    'standard_deviation_indicator': StandardDeviationIndicator,
    'variance_ratio_indicator': VarianceRatioIndicator,
    'z_score_indicator': ZScoreIndicator,
    'chaos_fractal_dimension': ChaosFractalDimension,
    
    # Other original indicators that were working
    'moving_average': SimpleMovingAverage,
    'rsi': RelativeStrengthIndex,
    'macd': MACD,
    'bollinger_bands': BollingerBands,
    'stochastic': StochasticOscillator,
    'adx': DirectionalMovementIndex,
    'cci': CommodityChannelIndex,
    'williams_r': WilliamsR,
    'roc': RateOfChange,
    'obv': OnBalanceVolume,
    'ad': AccumulationDistribution,
    'cmf': ChaikinMoneyFlow,
    'mfi': MoneyFlowIndex,
    'atr': AverageTrueRange,
    'parabolic_sar': ParabolicSAR,
    'trix': TRIX,
    'ultimate_oscillator': UltimateOscillator,
    'detrended_price_oscillator': DetrendedPriceOscillator,
    'ease_of_movement': EaseOfMovement,
    'negative_volume_index': NegativeVolumeIndex,
    'positive_volume_index': PositiveVolumeIndex,
    'price_volume_trend': PriceVolumeTrend,
    'volume_oscillator': VolumeOscillator,
    'chande_momentum_oscillator': ChandeMomentumOscillator,
    'know_sure_thing': KnowSureThing,
    'percentage_price_oscillator': PercentagePriceOscillator,
    'true_strength_index': TrueStrengthIndex,
    'vortex': VortexIndicator,
    'klinger_oscillator': KlingerOscillator,
    'elder_ray': ElderRay,
    'force_index': ForceIndex,
    'fractal_efficiency_ratio': FractalEfficiencyRatio,
    'attractor_point': AttractorPoint,
    'fractal_correlation_dimension': FractalCorrelationDimension,
    'hurst_exponent_calculator': HurstExponentCalculator,
    'multi_fractal_dfa': MultiFractalDFA,
    'self_similarity_signal': SelfSimilaritySignal,
    'fractal_adaptive_moving_average': FractalAdaptiveMovingAverage,
    'fibonacci_retracement': FibonacciRetracement,
    'fibonacci_extension': FibonacciExtension,
    'gann_fan': GannFan,
    'gann_square': GannSquare,
    'elliott_wave': ElliottWave,
    'abandoned_baby_signal': AbandonedBabySignal,
    'belt_hold_type': BeltHoldType,
    'dark_cloud_type': DarkCloudType,
    'doji_type': DojiType,
    'engulfing_type': EngulfingType,
    'hammer_type': HammerType,
    'harami_type': HaramiType,
    'piercing_line_type': PiercingLineType,
    'shooting_star_type': ShootingStarType,
    'spinning_top_type': SpinningTopType,
    'gann_angles_time_cycles': GannAnglesTimeCycles,
    'advanced_ml_engine': AdvancedMLEngine,
    'composite_signal': CompositeSignal,
    'genetic_algorithm_optimizer': GeneticAlgorithmOptimizer,
    'neural_network_predictor': NeuralNetworkPredictor'''

print("Current mapping prepared. Now need to create the complete registry structure.")
print("The fix will involve:")
print("1. Defining INDICATOR_REGISTRY with original working indicators")
print("2. Adding new indicators to reach 157 total")
print("3. Ensuring all imports work correctly")
