#!/usr/bin/env python3
"""
Find Missing Indicators Analysis
Compare documented indicators vs. registered indicators to identify missing ones
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

try:
    from engines.ai_enhancement.registry import INDICATOR_REGISTRY, get_enhanced_registry
    registry = get_enhanced_registry()
    
    print("=== MISSING INDICATORS ANALYSIS ===")
    print(f"Registry count: {len(registry._indicators)}")
    print(f"Documented count: 167")
    print(f"Missing: {167 - len(registry._indicators)}")
    print()
    
    # Get all documented indicators from the file paths
    documented_indicators = {
        # Physics indicators (7)
        'quantummomentumOracle', 'neuralharmonicresonance', 'chaosgeometrypredictor',
        'biorhythmmarketsyNth', 'photonicwavelengthanalyzer', 'thermodynamicentropyengine',
        'crystallographiclatticedetector',
        
        # Channel indicators (6)
        'bollingerbands', 'donchianchannels', 'keltnerchannels', 'linearregressionchannels',
        'sdchannelsignal', 'standarddeviationchannels',
        
        # Fibonacci indicators (6)
        'fibonacciarcindicator', 'fibonaccichannelindicator', 'fibonacciextensionindicator',
        'fibonaccifanindicator', 'fibonacciretracementindicator', 'fibonaccitimezoneindicator',
        
        # Fractal indicators (8)
        'fractaladaptivemovingaverage', 'fractalbreakoutindicator', 'fractalchannelindicator',
        'fractalchaosobillator', 'fractaldimensionindicator', 'fractalenergyindicator',
        'fractalvolumeindicator', 'mandelbrotfractalindicator',
        
        # Gann indicators (5)
        'gannanglesindicator', 'gannfanindicator', 'gannpricetimeindicator',
        'gannsquareindicator', 'ganntimecycleindicator',
        
        # Machine Learning indicators (3)
        'advancedmlengine', 'geneticalgorithmoptimizer', 'neuralnetworkpredictor',
        
        # Microstructure indicators (6)
        'bidaskspreadanalyzer', 'liquidityflowsignal', 'marketdepthindicator',
        'marketmicrostructuresignal', 'orderflowimbalance', 'orderflowsequencesignal',
        
        # Wave indicators (3)
        'wavepoint', 'wavestructure', 'wavetype',
        
        # Momentum indicators (26)
        'accelerationdecelerationindicator', 'awesomeoscillatorindicator', 'bearspowerindicator',
        'bullspowerindicator', 'chaikinoscillator', 'chandemomentumoscillatorindicator',
        'commoditychannelindex', 'correlationmatrixindicator', 'demarkerindicator',
        'detrendedpriceoscillatorindicator', 'fishertransform', 'knowsurethingIndicator',
        'macdsignalindicator', 'momentumindicator', 'moneyflowIndexindicator',
        'movingaverageconvergencedivergence', 'movingaverageconvergencedivergenceindicator',
        'percentagepriceoscillatorindicator', 'rateofchangeindicator', 'relativestrengthindex',
        'relativestrengthindexindicator', 'relativevigorindexindicator', 'rsisignalindicator',
        'stochasticoscillator', 'trixindicator', 'truestrengthindexindicator',
        'ultimateoscillatorindicator', 'williamsrindicator',
        
        # Pattern indicators (21) - Note: 21, not 20
        'abandonedBabysignal', 'beltholdtype', 'darkcloudcoverpattern', 'dojipattern',
        'engulfingpattern', 'hammerpattern', 'haramitype', 'highwavecandlepattern',
        'invertedhammershootingstarpattern', 'kickingsignal', 'longleggeddojipattern',
        'marubozupattern', 'matchingsignal', 'piercinglinepattern', 'soldierssignal',
        'spinningtoppattern', 'starsignal', 'threeInsidesignal', 'threelineStrikesignal',
        'threeoutsidesignal', 'tweezerpatterns',
        
        # Sentiment indicators (1)
        'newsarticle',
        
        # Social media indicators (1)
        'socialmediapostindicator',
        
        # Statistical indicators (23)
        'autocorrelationindicator', 'betacoefficientindicator', 'chaosfractaldimension',
        'cointegrationindicator', 'correlationanalysis', 'correlationcoefficientindicator',
        'cycleperiodidentification', 'dominantcycleanalysis', 'fractalefficiencyratio',
        'fractalmarkethypothesis', 'hurstexponent', 'linearregressionchannels',
        'linearregressionindicator', 'marketregimedetection', 'multifractaldfa',
        'rsquaredindicator', 'selfsimilaritysignal', 'skewnessindicator',
        'standarddeviationchannels', 'standarddeviationindicator', 'variance',
        'varianceratioindicator', 'zscoreindicator',
        
        # Technical indicators (11)
        'attractorpoint', 'compositesignal', 'confluencearea', 'fractalcorrelationdimension',
        'gridline', 'harmonicpoint', 'hiddendivergencedetector', 'momentumdivergencescanner',
        'phaseanalysis', 'pivotpoint', 'timeframeconfig',
        
        # Trend indicators (16) - Note: 16, not 15
        'adxindicator', 'alligatorindicator', 'aroonindicator', 'averagetruerange',
        'cciindicator', 'directionalmovementsystem', 'exponentialmovingaverage',
        'ichimokuindicator', 'macdindicator', 'parabolicsar', 'rsiindicator',
        'simplemovingaverage', 'stochasticindicator', 'supertrend', 'vortexindicator',
        'weightedmovingaverage',
        
        # Volume indicators (28) - Note: should be 28, not 30
        'accumulationdistribution', 'blocktradesignal', 'chaikinmoneyflow',
        'chaikinmoneyflowsignal', 'chaikinvolatility', 'easeofmovement', 'forceindex',
        'fractalmarketprofile', 'historicalvolatility', 'institutionalflowsignal',
        'klingeroscillator', 'massindex', 'negativevolumeindex', 'onbalancevolume',
        'positivevolumeindex', 'pricevolumedivergence', 'pricevolumerank',
        'pricevolumetrend', 'relativevolatilityindex', 'tickvolumesignal',
        'volatilityindex', 'volumebreakoutsignal', 'volumedeltasignal',
        'volumeoscillator', 'volumerateofchange', 'volumeweightedaverageprice',
        'vpttrendstate', 'vwapindicator'
    }
    
    print(f"Expected documented indicators: {len(documented_indicators)}")
    
    # Get registered indicators (normalize names)
    registered_indicators = {name.lower().replace('_', '') for name in registry._indicators.keys()}
    
    print(f"Registered indicators: {len(registered_indicators)}")
    print()
    
    # Find missing indicators
    missing = documented_indicators - registered_indicators
    extra = registered_indicators - documented_indicators
    
    print("=== MISSING INDICATORS ===")
    for indicator in sorted(missing):
        print(f"- {indicator}")
    print(f"Total missing: {len(missing)}")
    print()
    
    print("=== EXTRA INDICATORS IN REGISTRY ===")
    for indicator in sorted(extra):
        print(f"- {indicator}")
    print(f"Total extra: {len(extra)}")
    print()
    
    print("=== SUMMARY ===")
    print(f"Documented: {len(documented_indicators)}")
    print(f"Registered: {len(registered_indicators)}")
    print(f"Missing: {len(missing)}")
    print(f"Extra: {len(extra)}")
    print(f"Expected final count after fixing: {len(registered_indicators) + len(missing) - len(extra)}")

except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()