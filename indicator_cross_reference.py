#!/usr/bin/env python3
"""
Platform3 Indicator Cross-Reference Analysis
Compares documented indicators (indicator_implementation_priority.md) 
with actually loadable indicators (from comprehensive_indicator_loader_fixed.py output)
"""

from typing import Dict, List, Set
import re

class IndicatorCrossReference:
    def __init__(self):
        # Actual loadable indicators from comprehensive_indicator_loader_fixed.py output
        self.loadable_indicators = {
            'volume': ['ForceIndex', 'KlingerOscillator', 'NegativeVolumeIndex', 'PositiveVolumeIndex', 'VolumeOscillator', 'VolumePriceTrend'],
            'volatility': ['MassIndex', 'RelativeVolatilityIndex', 'VolatilityIndex'],
            'trend': ['AroonIndicator', 'DirectionalMovementSystem', 'KeltnerAnalysis', 'ParabolicSAR', 'SuperTrend'],
            'statistical': ['AutocorrelationIndicator', 'BetaCoefficientIndicator', 'CointegrationIndicator', 'CorrelationCoefficientIndicator', 'LinearRegressionIndicator', 'RSquaredIndicator', 'SkewnessKurtosisIndicator', 'StandardDeviationIndicator', 'VarianceRatioIndicator', 'ZScoreIndicator', 'AdvancedZScore'],
            'pattern': ['BeltHoldPattern', 'DarkCloudCoverPattern', 'DojiRecognitionEngine', 'ElliottWavePattern', 'EngulfingPattern', 'HaramiPattern', 'HarmonicPatternDetector', 'HighWavePattern', 'InvertedHammerPattern', 'MauribosuPattern', 'PiercingLinePattern', 'ShootingStarPattern', 'SpinningTopPattern', 'TweezerPatternsDetector', 'HammerPattern', 'AbandonedBabyPattern', 'MorningStarPattern', 'EveningStarPattern'],
            'momentum': ['AwesomeOscillator', 'CommodityChannelIndex', 'ChandeMomentumOscillator', 'KnowSureThingOscillator', 'MACDIndicator', 'PercentagePriceOscillator', 'RelativeStrengthIndex', 'StochasticOscillator', 'TrueStrengthIndex', 'UltimateOscillator'],
            'indicators': ['MomentumIndicators'],
            'fractal': ['ChaosTheoryIndicators', 'FractalChannelIndicator', 'FractalChaosOscillator', 'FractalDimensionCalculator', 'FractalEnergyIndicator', 'FractalMarketProfile', 'MandelbrotFractalIndicator', 'SelfSimilarityDetector', 'MultiFractalDetrendedFluctuationAnalysis'],
            'gann': ['GannPattern', 'GannPatternDetector', 'GannFanAnalysis', 'GannTimeCycles'],
            'elliott_wave': ['WaveAnalysis', 'EnhancedElliottWaveCalculator'],
            'fibonacci': ['ConfluenceDetector', 'ProjectionArcCalculator', 'TimeZoneAnalysis'],
            'cycle': ['DominantCycleAnalysis', 'PhaseAnalysis'],
            'divergence': ['HiddenDivergenceDetector'],
            'ai_enhancement': ['AdaptiveIndicators', 'IndicatorPerformance', 'AdaptiveIndicatorBridge', 'AdaptiveIndicatorCoordinator', 'CustomAICompositeIndicator', 'GeniusAgentIndicatorSystem', 'MLSignalGenerator', 'MarketMicrostructureAnalysis', 'SentimentIntegration', 'MultiAssetCorrelation', 'RegimeDetectionAI', 'EnhancedAdaptiveCoordinator']
        }
        
        # Expected indicators from indicator_implementation_priority.md (115+ documented as implemented)
        self.documented_indicators = {
            # Fractal Geometry (15)
            'fractal': [
                'FractalDimensionCalculator', 'MandelbrotFractal', 'FRAMA', 'FractalChannel', 'MFDFA',
                'FractalMarketHypothesis', 'FractalEfficiencyRatio', 'FractalBreakout', 'FractalMomentumOscillator',
                'FractalVolumeAnalysis', 'FractalCorrelationDimension', 'FractalEnergyIndicator',
                'FractalChaosOscillator', 'FractalWaveCounter', 'FractalMarketProfile'
            ],
            
            # Candlestick Patterns (25)
            'pattern': [
                'DojiStandard', 'DojiDragonfly', 'DojiGravestone', 'DojiLongLegged',
                'Hammer', 'HangingMan', 'InvertedHammer', 'ShootingStar',
                'Marubozu', 'SpinningTop', 'HighWaveCandle',
                'BullishEngulfing', 'BearishEngulfing', 'BullishHarami', 'BearishHarami',
                'PiercingLine', 'DarkCloudCover', 'TweezerTops', 'TweezerBottoms',
                'BeltHold', 'KickingPattern', 'MorningStar', 'EveningStar',
                'ThreeWhiteSoldiers', 'ThreeBlackCrows', 'ThreeInsideUp', 'ThreeInsideDown',
                'ThreeOutsideUp', 'ThreeOutsideDown', 'AbandonedBaby',
                'ThreeLineStrike', 'MatchingLow', 'MatchingHigh'
            ],
            
            # Core Technical - Momentum (15)
            'momentum': [
                'RSI', 'MACD', 'StochasticOscillator', 'WilliamsR', 'CCI',
                'ROC', 'TSI', 'UltimateOscillator', 'AwesomeOscillator',
                'PPO', 'DPO', 'CMO', 'KST', 'TRIX', 'MomentumIndicator'
            ],
            
            # Core Technical - Trend (15) - NOTE: These are in core_trend but not loading
            'trend': [
                'SMA', 'EMA', 'WMA', 'TEMA', 'DEMA', 'HMA', 'KAMA',
                'ADX', 'AroonIndicator', 'IchimokuCloud', 'ParabolicSAR',
                'SuperTrend', 'VWMA', 'McGinleyDynamic', 'ZeroLagEMA'
            ],
            
            # Core Technical - Volatility (10)
            'volatility': [
                'BollingerBands', 'ATR', 'KeltnerChannels', 'DonchianChannels',
                'StandardDeviationChannels', 'VolatilityIndex', 'HistoricalVolatility',
                'ChaikinVolatility', 'MassIndex', 'RVI'
            ],
            
            # Volume & Market Structure (15)
            'volume': [
                'OBV', 'MFI', 'VWAP', 'VolumeProfile', 'ChaikinMoneyFlow',
                'AccumulationDistribution', 'EaseOfMovement', 'VolumePriceTrend',
                'NegativeVolumeIndex', 'PositiveVolumeIndex', 'VolumeRateOfChange',
                'PriceVolumeRank', 'VolumeOscillator', 'KlingerOscillator', 'ForceIndex'
            ],
            
            # Advanced - Statistical (10)
            'statistical': [
                'LinearRegression', 'StandardDeviation', 'CorrelationCoefficient',
                'ZScore', 'BetaCoefficient', 'RSquared', 'VarianceRatio',
                'SkewnessKurtosis', 'Cointegration', 'Autocorrelation'
            ],
            
            # Advanced - Fibonacci (5)
            'fibonacci': [
                'FibonacciRetracement', 'FibonacciExtension', 'FibonacciTimeZones',
                'FibonacciArc', 'FibonacciFan'
            ],
            
            # Advanced - Gann (5)
            'gann': [
                'GannFanLines', 'GannSquareOfNine', 'GannTimeCycles',
                'PriceTimeRelationships', 'GannGrid'
            ],
            
            # Bonus (5+)
            'bonus': [
                'ElliottWaveCounter', 'HarmonicPatternDetector', 'MarketProfile',
                'PivotPoints', 'CustomAIComposite'
            ]
        }

    def normalize_name(self, name: str) -> str:
        """Normalize indicator names for comparison"""
        # Convert to lowercase and remove common suffixes
        name = name.lower()
        suffixes = ['indicator', 'oscillator', 'analysis', 'detector', 'calculator', 'engine', 'pattern', 'index']
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.strip('_')

    def find_matches(self, documented_name: str, loadable_list: List[str]) -> List[str]:
        """Find potential matches for a documented indicator in loadable list"""
        normalized_doc = self.normalize_name(documented_name)
        matches = []
        
        for loadable in loadable_list:
            normalized_load = self.normalize_name(loadable)
            
            # Direct match
            if normalized_doc == normalized_load:
                matches.append(loadable)
                continue
            
            # Partial match
            if normalized_doc in normalized_load or normalized_load in normalized_doc:
                matches.append(loadable)
                continue
            
            # Common abbreviations
            abbreviations = {
                'rsi': 'relativestrength',
                'macd': 'movingaverageconvergencedivergence',
                'cci': 'commoditychannel',
                'atr': 'averagetrue',
                'obv': 'onbalancevolume',
                'mfi': 'moneyflow',
                'vwap': 'volumeweightedaverage',
                'adx': 'averagedirectional',
                'roc': 'rateofchange',
                'tsi': 'truestrength',
                'ppo': 'percentageprice',
                'dpo': 'detrendedprice',
                'cmo': 'chandemomentum',
                'kst': 'knowsure',
                'williams': 'williamsr',
                'bollinger': 'bollingerband',
                'keltner': 'keltner',
                'donchian': 'donchian'
            }
            
            for abbr, full in abbreviations.items():
                if (normalized_doc == abbr and abbr in normalized_load) or \
                   (normalized_load == abbr and abbr in normalized_doc) or \
                   (normalized_doc == full and abbr in normalized_load) or \
                   (normalized_load == full and abbr in normalized_doc):
                    matches.append(loadable)
                    break
        
        return matches

    def analyze_coverage(self):
        """Analyze coverage of documented indicators by loadable indicators"""
        print("="*80)
        print("PLATFORM3 INDICATOR CROSS-REFERENCE ANALYSIS")
        print("="*80)
        
        total_documented = 0
        total_found = 0
        total_missing = 0
        
        all_missing = []
        all_found = []
        
        # Flatten all loadable indicators
        all_loadable = []
        for category, indicators in self.loadable_indicators.items():
            all_loadable.extend(indicators)
        
        for category, documented_list in self.documented_indicators.items():
            print(f"\n📋 CATEGORY: {category.upper()}")
            print("-" * 40)
            
            category_total = len(documented_list)
            category_found = 0
            category_missing = []
            
            for doc_indicator in documented_list:
                # Look for matches in corresponding category first
                loadable_list = self.loadable_indicators.get(category, [])
                matches = self.find_matches(doc_indicator, loadable_list)
                
                # If no matches in category, search all indicators
                if not matches:
                    matches = self.find_matches(doc_indicator, all_loadable)
                
                if matches:
                    print(f"  ✅ {doc_indicator} → {matches[0]}")
                    category_found += 1
                    all_found.append(doc_indicator)
                else:
                    print(f"  ❌ {doc_indicator} → NOT FOUND")
                    category_missing.append(doc_indicator)
                    all_missing.append(doc_indicator)
            
            coverage = (category_found / category_total * 100) if category_total > 0 else 0
            print(f"  📊 Coverage: {category_found}/{category_total} ({coverage:.1f}%)")
            
            total_documented += category_total
            total_found += category_found
            total_missing += len(category_missing)
        
        overall_coverage = (total_found / total_documented * 100) if total_documented > 0 else 0
        
        print(f"\n📊 OVERALL SUMMARY")
        print("-" * 40)
        print(f"Total Documented: {total_documented}")
        print(f"Total Found: {total_found}")
        print(f"Total Missing: {total_missing}")
        print(f"Overall Coverage: {overall_coverage:.1f}%")
        
        # Show critical missing indicators
        if all_missing:
            print(f"\n❌ CRITICAL MISSING INDICATORS ({len(all_missing)}):")
            print("-" * 40)
            # Group by category for better readability
            for category, documented_list in self.documented_indicators.items():
                category_missing = [ind for ind in documented_list if ind in all_missing]
                if category_missing:
                    print(f"\n{category.upper()} ({len(category_missing)} missing):")
                    for indicator in category_missing:
                        print(f"  └── {indicator}")
        
        # Analyze what we have that's not documented
        documented_flat = []
        for indicators in self.documented_indicators.values():
            documented_flat.extend(indicators)
        
        extra_indicators = []
        for loadable in all_loadable:
            found = False
            for documented in documented_flat:
                if self.find_matches(documented, [loadable]):
                    found = True
                    break
            if not found:
                extra_indicators.append(loadable)
        
        if extra_indicators:
            print(f"\n➕ EXTRA LOADABLE INDICATORS ({len(extra_indicators)}):")
            print("-" * 40)
            for indicator in extra_indicators:
                print(f"  └── {indicator}")
        
        print("\n" + "="*80)
        
        return {
            'total_documented': total_documented,
            'total_found': total_found,
            'total_missing': total_missing,
            'coverage_rate': overall_coverage,
            'missing_indicators': all_missing,
            'extra_indicators': extra_indicators
        }

def main():
    print("🚀 Platform3 Indicator Cross-Reference Analysis Starting...")
    
    analyzer = IndicatorCrossReference()
    results = analyzer.analyze_coverage()
    
    print(f"\n✅ Analysis complete!")
    print(f"📈 Coverage: {results['coverage_rate']:.1f}% ({results['total_found']}/{results['total_documented']} documented indicators found)")
    print(f"🔍 Missing: {results['total_missing']} indicators need attention")
    print(f"➕ Extra: {len(results['extra_indicators'])} indicators not documented")

if __name__ == "__main__":
    main()
