#!/usr/bin/env python3
"""
Identify and Fix Missing 11 Indicators
Compare documented vs registered indicators and fix the loading issues
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

try:
    from engines.ai_enhancement.registry import get_enhanced_registry
    registry = get_enhanced_registry()
    
    print("=== IDENTIFY MISSING 11 INDICATORS ===")
    print(f"Current registry count: {len(registry._indicators)}")
    print(f"Target count: 167")
    print(f"Missing: {167 - len(registry._indicators)}")
    print()
    
    # Get all registered indicators (normalize names)
    registered_indicators = {name.lower().replace('_', '') for name in registry._indicators.keys()}
    
    # Get all documented indicators from COMPLETE_INDICATOR_REGISTRY.md
    documented_indicators = {
        # Physics indicators (7)
        'quantummomentumoracle', 'neuralharmonicresonance', 'chaosgeometrypredictor',
        'biorhythmmarketsynth', 'photonicwavelengthanalyzer', 'thermodynamicentropyengine',
        'crystallographiclatticedetector',
        
        # Channel indicators (6)
        'bollingerbands', 'donchianchannels', 'keltnerchannels', 'linearregressionchannels',
        'sdchannelsignal', 'standarddeviationchannels',
        
        # Fibonacci indicators (6)
        'fibonacciarcindicator', 'fibonaccichannelindicator', 'fibonacciextensionindicator',
        'fibonaccifanindicator', 'fibonacciretracementindicator', 'fibonaccitimezoneindicator',
        
        # Fractal indicators (8)
        'fractaladaptivemovingaverage', 'fractalbreakoutindicator', 'fractalchannelindicator',
        'fractalchaososcillator', 'fractaldimensionindicator', 'fractalenergyindicator',
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
        'detrendedpriceoscillatorindicator', 'fishertransform', 'knowsurethingindicator',
        'macdsignalindicator', 'momentumindicator', 'moneyflowindexindicator',
        'movingaverageconvergencedivergence', 'movingaverageconvergencedivergenceindicator',
        'percentagepriceoscillatorindicator', 'rateofchangeindicator', 'relativestrengthindex',
        'relativestrengthindexindicator', 'relativevigorindexindicator', 'rsisignalindicator',
        'stochasticoscillator', 'trixindicator', 'truestrengthindexindicator',
        'ultimateoscillatorindicator', 'williamsrindicator',
        
        # Pattern indicators (21)
        'abandonedbabysignal', 'beltholdtype', 'darkcloudcoverpattern', 'dojipattern',
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
        'fractalmarkethypothesis', 'hurstexponent', 'linearregressionindicator', 'marketregimedetection', 
        'multifractaldfa', 'rsquaredindicator', 'selfsimilaritysignal', 'skewnessindicator',
        'standarddeviationindicator', 'variance', 'varianceratioindicator', 'zscoreindicator',
        'fractalmarketprofile', 'correlationmatrixindicator',
        
        # Technical indicators (11)
        'attractorpoint', 'compositesignal', 'confluencearea', 'fractalcorrelationdimension',
        'gridline', 'harmonicpoint', 'hiddendivergencedetector', 'momentumdivergencescanner',
        'phaseanalysis', 'pivotpoint', 'timeframeconfig',
        
        # Trend indicators (16)
        'adxindicator', 'alligatorindicator', 'aroonindicator', 'averagetruerange',
        'cciindicator', 'directionalmovementsystem', 'exponentialmovingaverage',
        'ichimokuindicator', 'macdindicator', 'parabolicsar', 'rsiindicator',
        'simplemovingaverage', 'stochasticindicator', 'supertrend', 'vortexindicator',
        'weightedmovingaverage',
        
        # Volume indicators (28)
        'accumulationdistribution', 'blocktradesignal', 'chaikinmoneyflow',
        'chaikinmoneyflowsignal', 'chaikinvolatility', 'easeofmovement', 'forceindex',
        'historicalvolatility', 'institutionalflowsignal', 'klingeroscillator', 'massindex', 
        'negativevolumeindex', 'onbalancevolume', 'positivevolumeindex', 'pricevolumedivergence', 
        'pricevolumerank', 'pricevolumetrend', 'relativevolatilityindex', 'tickvolumesignal',
        'volatilityindex', 'volumebreakoutsignal', 'volumedeltasignal', 'volumeoscillator', 
        'volumerateofchange', 'volumeweightedaverageprice', 'vpttrendstate', 'vwapindicator',
        'fractalmarketprofile'
    }
    
    print(f"Expected documented indicators: {len(documented_indicators)}")
    
    # Find missing indicators
    missing = documented_indicators - registered_indicators
    
    print("=== MISSING INDICATORS ===")
    for i, indicator in enumerate(sorted(missing), 1):
        print(f"{i:2d}. {indicator}")
    print(f"Total missing: {len(missing)}")
    print()
    
    # Check if these indicators have actual files
    print("=== CHECKING IF MISSING INDICATORS HAVE FILES ===")
    indicators_base_path = Path("engines/ai_enhancement/indicators")
    
    file_exists_count = 0
    missing_files = []
    
    for indicator in sorted(missing):
        # Try to find the file
        found_file = None
        
        # Search in all subdirectories
        for pattern_dir in indicators_base_path.rglob("*.py"):
            file_stem = pattern_dir.stem.lower().replace('_', '')
            if file_stem == indicator:
                found_file = pattern_dir
                break
        
        if found_file:
            file_exists_count += 1
            print(f"[FILE EXISTS] {indicator} -> {found_file}")
        else:
            missing_files.append(indicator)
            print(f"[NO FILE] {indicator}")
    
    print(f"\nSummary:")
    print(f"Missing indicators with files: {file_exists_count}")
    print(f"Missing indicators without files: {len(missing_files)}")
    
    if missing_files:
        print(f"\nIndicators that need file creation:")
        for indicator in missing_files:
            print(f"  - {indicator}")

except Exception as e:
    print(f"Error during analysis: {e}")
    import traceback
    traceback.print_exc()