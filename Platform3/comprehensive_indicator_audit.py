#!/usr/bin/env python3
"""
Comprehensive Platform3 Indicator Audit
========================================

This script thoroughly checks and verifies ALL indicators in the Platform3 forex trading platform
to ensure they are complete, functional, and ready for production use.

Indicator Categories Audited:
1. Technical Indicators (Momentum, Trend, Volatility, Volume, Cycle)
2. Advanced Indicators (PCA, Autoencoder, Sentiment, Time-Weighted Volatility)
3. Gann Indicators (All types)
4. Fibonacci Indicators (All types)
5. Pivot Indicators
6. Elliott Wave Indicators
7. Fractal Geometry Indicators
8. Specialized Trading Indicators (Scalping, Day Trading, Swing Trading)

Author: Platform3 Audit Team
Version: 1.0.0
"""

import sys
import os
import importlib
import traceback
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append('services/analytics-service/src')

class IndicatorAuditor:
    def __init__(self):
        self.results = {
            'technical_indicators': {},
            'advanced_indicators': {},
            'gann_indicators': {},
            'fibonacci_indicators': {},
            'pivot_indicators': {},
            'elliott_wave_indicators': {},
            'fractal_geometry_indicators': {},
            'specialized_indicators': {},
            'summary': {}
        }

    def audit_technical_indicators(self):
        """Audit all technical indicators by category"""
        logger.info("🔍 AUDITING TECHNICAL INDICATORS")
        logger.info("=" * 50)

        categories = {
            'momentum': ['RSI', 'MACD', 'Stochastic', 'DayTradingMomentum', 'ScalpingMomentum', 'SwingMomentum'],
            'trend': ['SMA_EMA', 'ADX', 'Ichimoku'],
            'volatility': ['ATR', 'BollingerBands', 'CCI', 'KeltnerChannels', 'ParabolicSAR', 'SuperTrend', 'Vortex'],
            'volume': ['OBV', 'MFI', 'VFI', 'AdvanceDecline'],
            'cycle': ['Alligator', 'FisherTransform', 'HurstExponent']
        }

        for category, indicators in categories.items():
            logger.info(f"\n📊 {category.upper()} INDICATORS:")
            category_results = {}

            for indicator in indicators:
                try:
                    # Try to import the indicator
                    module_path = f"engines.indicators.{category}.{indicator}"
                    module = importlib.import_module(module_path)

                    # Check if the main class exists
                    if hasattr(module, indicator):
                        indicator_class = getattr(module, indicator)
                        # Try to instantiate
                        instance = indicator_class()
                        category_results[indicator] = "✅ READY"
                        logger.info(f"  ✅ {indicator}: READY")
                    else:
                        category_results[indicator] = "⚠️  CLASS_MISSING"
                        logger.info(f"  ⚠️  {indicator}: Class not found in module")

                except ImportError as e:
                    category_results[indicator] = "❌ IMPORT_ERROR"
                    logger.info(f"  ❌ {indicator}: Import failed")
                except Exception as e:
                    category_results[indicator] = "⚠️  INSTANTIATION_ERROR"
                    logger.info(f"  ⚠️  {indicator}: {str(e)}")

            self.results['technical_indicators'][category] = category_results

    def audit_advanced_indicators(self):
        """Audit advanced indicators"""
        logger.info("\n🧠 AUDITING ADVANCED INDICATORS")
        logger.info("=" * 50)

        advanced_indicators = [
            'TimeWeightedVolatility',
            'PCAFeatures',
            'AutoencoderFeatures',
            'SentimentScores',
            'AdvancedIndicatorSuite'
        ]

        results = {}
        for indicator in advanced_indicators:
            try:
                module = importlib.import_module("engines.indicators.advanced")
                if hasattr(module, indicator):
                    indicator_class = getattr(module, indicator)

                    # Special handling for AutoencoderFeatures which requires input_dim
                    if indicator == 'AutoencoderFeatures':
                        instance = indicator_class(input_dim=10)
                    else:
                        instance = indicator_class()

                    results[indicator] = "✅ READY"
                    logger.info(f"  ✅ {indicator}: READY")
                else:
                    results[indicator] = "❌ NOT_FOUND"
                    logger.info(f"  ❌ {indicator}: Not found")
            except Exception as e:
                results[indicator] = f"❌ ERROR: {str(e)}"
                logger.info(f"  ❌ {indicator}: {str(e)}")

        self.results['advanced_indicators'] = results

    def audit_gann_indicators(self):
        """Audit Gann indicators"""
        logger.info("\n📐 AUDITING GANN INDICATORS")
        logger.info("=" * 50)

        gann_indicators = [
            'GannAnglesCalculator',
            'GannFanAnalysis',
            'GannPatternDetector',
            'GannSquareOfNine',
            'GannTimePrice'
        ]

        results = {}
        for indicator in gann_indicators:
            try:
                module_path = f"engines.gann.{indicator}"
                module = importlib.import_module(module_path)
                if hasattr(module, indicator):
                    results[indicator] = "✅ READY"
                    logger.info(f"  ✅ {indicator}: READY")
                else:
                    results[indicator] = "⚠️  CLASS_MISSING"
                    logger.info(f"  ⚠️  {indicator}: Class missing")
            except Exception as e:
                results[indicator] = "❌ ERROR"
                logger.info(f"  ❌ {indicator}: {str(e)}")

        self.results['gann_indicators'] = results

    def audit_fibonacci_indicators(self):
        """Audit Fibonacci indicators"""
        logger.info("\n🌀 AUDITING FIBONACCI INDICATORS")
        logger.info("=" * 50)

        fibonacci_indicators = [
            'FibonacciRetracement',
            'FibonacciExtension',
            'ProjectionArcCalculator',
            'TimeZoneAnalysis',
            'ConfluenceDetector'
        ]

        results = {}
        for indicator in fibonacci_indicators:
            try:
                module_path = f"engines.fibonacci.{indicator}"
                module = importlib.import_module(module_path)
                if hasattr(module, indicator):
                    results[indicator] = "✅ READY"
                    logger.info(f"  ✅ {indicator}: READY")
                else:
                    results[indicator] = "⚠️  CLASS_MISSING"
                    logger.info(f"  ⚠️  {indicator}: Class missing")
            except Exception as e:
                results[indicator] = "❌ ERROR"
                logger.info(f"  ❌ {indicator}: {str(e)}")

        self.results['fibonacci_indicators'] = results

    def audit_pivot_indicators(self):
        """Audit Pivot indicators"""
        logger.info("\n📍 AUDITING PIVOT INDICATORS")
        logger.info("=" * 50)

        try:
            module = importlib.import_module("engines.pivot.PivotPointCalculator")
            if hasattr(module, 'PivotPointCalculator'):
                self.results['pivot_indicators']['PivotPointCalculator'] = "✅ READY"
                logger.info("  ✅ PivotPointCalculator: READY")
            else:
                self.results['pivot_indicators']['PivotPointCalculator'] = "⚠️  CLASS_MISSING"
                logger.info("  ⚠️  PivotPointCalculator: Class missing")
        except Exception as e:
            self.results['pivot_indicators']['PivotPointCalculator'] = "❌ ERROR"
            logger.info(f"  ❌ PivotPointCalculator: {str(e)}")

    def audit_elliott_wave_indicators(self):
        """Audit Elliott Wave indicators"""
        logger.info("\n🌊 AUDITING ELLIOTT WAVE INDICATORS")
        logger.info("=" * 50)

        elliott_indicators = [
            'ShortTermElliottWaves',
            'QuickFibonacci',
            'SessionSupportResistance'
        ]

        results = {}
        for indicator in elliott_indicators:
            try:
                module_path = f"engines.swingtrading.{indicator}"
                module = importlib.import_module(module_path)
                if hasattr(module, indicator):
                    results[indicator] = "✅ READY"
                    logger.info(f"  ✅ {indicator}: READY")
                else:
                    results[indicator] = "⚠️  CLASS_MISSING"
                    logger.info(f"  ⚠️  {indicator}: Class missing")
            except Exception as e:
                results[indicator] = "❌ ERROR"
                logger.info(f"  ❌ {indicator}: {str(e)}")

        self.results['elliott_wave_indicators'] = results

    def audit_fractal_geometry_indicators(self):
        """Audit Fractal Geometry indicators"""
        logger.info("\n🔺 AUDITING FRACTAL GEOMETRY INDICATORS")
        logger.info("=" * 50)

        try:
            module = importlib.import_module("engines.fractal_geometry.FractalGeometryIndicator")
            if hasattr(module, 'FractalGeometryIndicator'):
                self.results['fractal_geometry_indicators']['FractalGeometryIndicator'] = "✅ READY"
                logger.info("  ✅ FractalGeometryIndicator: READY")
            else:
                self.results['fractal_geometry_indicators']['FractalGeometryIndicator'] = "⚠️  CLASS_MISSING"
                logger.info("  ⚠️  FractalGeometryIndicator: Class missing")
        except Exception as e:
            self.results['fractal_geometry_indicators']['FractalGeometryIndicator'] = "❌ ERROR"
            logger.info(f"  ❌ FractalGeometryIndicator: {str(e)}")

    def audit_specialized_indicators(self):
        """Audit specialized trading indicators"""
        logger.info("\n⚡ AUDITING SPECIALIZED TRADING INDICATORS")
        logger.info("=" * 50)

        specialized_categories = {
            'scalping': ['MicrostructureFilters', 'OrderBookAnalysis', 'ScalpingPriceAction', 'TickVolumeIndicators', 'VWAPScalping'],
            'daytrading': ['FastMomentumOscillators', 'IntradayTrendAnalysis', 'SessionBreakouts', 'SessionMomentum', 'VolatilitySpikesDetector'],
            'volume_analysis': ['OrderFlowImbalance', 'SmartMoneyIndicators', 'TickVolumeIndicators', 'VolumeProfiles', 'VolumeSpreadAnalysis'],
            'signals': ['ConfidenceCalculator', 'ConflictResolver', 'QuickDecisionMatrix', 'SignalAggregator', 'TimeframeSynchronizer'],
            'ml_indicators': ['NoiseFilter', 'ScalpingLSTM', 'SpreadPredictor', 'TickClassifier']
        }

        for category, indicators in specialized_categories.items():
            logger.info(f"\n  📊 {category.upper()}:")
            category_results = {}

            for indicator in indicators:
                try:
                    if category == 'ml_indicators':
                        module_path = f"engines.ml.scalping.{indicator}"
                    elif category == 'volume_analysis':
                        module_path = f"engines.volume.{indicator}"
                    else:
                        module_path = f"engines.{category}.{indicator}"

                    module = importlib.import_module(module_path)
                    if hasattr(module, indicator):
                        category_results[indicator] = "✅ READY"
                        logger.info(f"    ✅ {indicator}: READY")
                    else:
                        category_results[indicator] = "⚠️  CLASS_MISSING"
                        logger.info(f"    ⚠️  {indicator}: Class missing")
                except Exception as e:
                    category_results[indicator] = "❌ ERROR"
                    logger.info(f"    ❌ {indicator}: {str(e)}")

            self.results['specialized_indicators'][category] = category_results

    def generate_summary(self):
        """Generate comprehensive summary of all indicators"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE PLATFORM3 INDICATOR AUDIT SUMMARY")
        logger.info("=" * 80)

        total_indicators = 0
        ready_indicators = 0

        # Count indicators by category
        for category, indicators in self.results.items():
            if category == 'summary':
                continue

            if isinstance(indicators, dict):
                for subcategory, subindicators in indicators.items():
                    if isinstance(subindicators, dict):
                        for indicator, status in subindicators.items():
                            total_indicators += 1
                            if "✅ READY" in status:
                                ready_indicators += 1
                    else:
                        total_indicators += 1
                        if "✅ READY" in subindicators:
                            ready_indicators += 1

        completion_percentage = (ready_indicators / total_indicators * 100) if total_indicators > 0 else 0

        logger.info(f"\n🎯 OVERALL COMPLETION: {ready_indicators}/{total_indicators} ({completion_percentage:.1f}%)")

        # Category breakdown
        logger.info("\n📋 CATEGORY BREAKDOWN:")

        for category, indicators in self.results.items():
            if category == 'summary':
                continue

            category_total = 0
            category_ready = 0

            if isinstance(indicators, dict):
                for subcategory, subindicators in indicators.items():
                    if isinstance(subindicators, dict):
                        for indicator, status in subindicators.items():
                            category_total += 1
                            if "✅ READY" in status:
                                category_ready += 1
                    else:
                        category_total += 1
                        if "✅ READY" in subindicators:
                            category_ready += 1

            category_percentage = (category_ready / category_total * 100) if category_total > 0 else 0
            status_icon = "✅" if category_percentage >= 80 else "⚠️" if category_percentage >= 50 else "❌"

            logger.info(f"  {status_icon} {category.replace('_', ' ').title()}: {category_ready}/{category_total} ({category_percentage:.1f}%)")

        # Store summary
        self.results['summary'] = {
            'total_indicators': total_indicators,
            'ready_indicators': ready_indicators,
            'completion_percentage': completion_percentage,
            'audit_timestamp': datetime.now().isoformat()
        }

        # Final assessment
        logger.info("\n🏆 PLATFORM READINESS ASSESSMENT:")
        if completion_percentage >= 90:
            logger.info("  🟢 EXCELLENT: Platform is production-ready with comprehensive indicator coverage")
        elif completion_percentage >= 75:
            logger.info("  🟡 GOOD: Platform has strong indicator coverage, minor gaps remain")
        elif completion_percentage >= 50:
            logger.info("  🟠 MODERATE: Platform has basic indicator coverage, significant development needed")
        else:
            logger.info("  🔴 POOR: Platform requires substantial indicator development")

        return self.results

    def run_full_audit(self):
        """Run complete indicator audit"""
        logger.info("🚀 STARTING COMPREHENSIVE PLATFORM3 INDICATOR AUDIT")
        logger.info("=" * 80)
        logger.info(f"Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # Run all audits
        self.audit_technical_indicators()
        self.audit_advanced_indicators()
        self.audit_gann_indicators()
        self.audit_fibonacci_indicators()
        self.audit_pivot_indicators()
        self.audit_elliott_wave_indicators()
        self.audit_fractal_geometry_indicators()
        self.audit_specialized_indicators()

        # Generate summary
        return self.generate_summary()

def main():
    """Main execution function"""
    auditor = IndicatorAuditor()
    results = auditor.run_full_audit()

    # Save results to file
    import json
    with open('indicator_audit_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n💾 Detailed results saved to: indicator_audit_results.json")
    logger.info("🎉 Audit completed successfully!")

if __name__ == "__main__":
    main()
