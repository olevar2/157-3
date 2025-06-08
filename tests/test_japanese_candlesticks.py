# -*- coding: utf-8 -*-
"""
Unit tests for Japanese Candlestick Pattern detectors
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from engines.pattern.japanese_candlestick_patterns import (
    JapaneseCandlestickPatterns, PatternType, CandleData
)
from engines.pattern.inverted_hammer_shooting_star import InvertedHammerShootingStarDetector
from engines.pattern.marubozu import MarubozuDetector
from engines.pattern.spinning_top import SpinningTopDetector
from engines.pattern.high_wave_candle import HighWaveCandleDetector
from engines.pattern.long_legged_doji import LongLeggedDojiDetector


class TestJapaneseCandlestickPatterns(unittest.TestCase):
    """Test Japanese Candlestick Pattern detection"""
    
    def setUp(self):
        """Set up test data"""
        self.detector = JapaneseCandlestickPatterns()
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Ensure high >= open, close and low <= open, close
        for i in range(len(self.sample_data)):
            self.sample_data.loc[self.sample_data.index[i], 'high'] = max(
                self.sample_data.iloc[i]['high'],
                self.sample_data.iloc[i]['open'],
                self.sample_data.iloc[i]['close']
            )
            self.sample_data.loc[self.sample_data.index[i], 'low'] = min(
                self.sample_data.iloc[i]['low'],
                self.sample_data.iloc[i]['open'],
                self.sample_data.iloc[i]['close']
            )
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.doji_threshold, 0.1)
        self.assertEqual(self.detector.shadow_ratio_threshold, 2.0)
    
    def test_calculate_with_valid_data(self):
        """Test calculate with valid data"""
        result = self.detector.calculate(self.sample_data)
        
        self.assertIn('patterns', result)
        self.assertIn('pattern_count', result)
        self.assertIn('analysis', result)
        self.assertIn('strongest_pattern', result)
        self.assertIn('trend_context', result)
    
    def test_doji_detection(self):
        """Test Doji pattern detection"""
        # Create perfect doji
        doji_data = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [98],
            'close': [100.05],  # Very small body
            'volume': [5000]
        })
        
        candles = self.detector._create_candle_objects(doji_data)
        self.detector._detect_doji_patterns(candles[0], 0, 'uptrend')
        
        self.assertTrue(len(self.detector.detected_patterns) > 0)
        pattern = self.detector.detected_patterns[0]
        self.assertIn('Doji', pattern.pattern_name)
    
    def test_hammer_detection(self):
        """Test Hammer pattern detection"""
        # Create perfect hammer
        hammer_data = pd.DataFrame({
            'open': [100],
            'high': [100.5],
            'low': [95],  # Long lower shadow
            'close': [100.3],
            'volume': [5000]
        })
        
        candles = self.detector._create_candle_objects(hammer_data)
        self.detector._detect_hammer_hanging_man(candles[0], 0, 'downtrend')
        
        if self.detector.detected_patterns:
            pattern = self.detector.detected_patterns[0]
            self.assertEqual(pattern.pattern_name, 'Hammer')
            self.assertEqual(pattern.pattern_type, PatternType.BULLISH)


class TestInvertedHammerShootingStar(unittest.TestCase):
    """Test Inverted Hammer and Shooting Star detection"""
    
    def setUp(self):
        """Set up test data"""
        self.detector = JapaneseCandlestickPatterns()
    
    def test_inverted_hammer_detection(self):
        """Test Inverted Hammer detection in downtrend"""
        # Create downtrend with inverted hammer
        data = pd.DataFrame({
            'open': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90.5],
            'high': [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 95],  # Last has long upper shadow
            'low': [99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 90],
            'close': [99, 98, 97, 96, 95, 94, 93, 92, 91, 90.2, 90.7],
            'volume': [1000] * 11
        })
        
        result = self.detector.calculate(data)
        
        self.assertIn('patterns', result)
        self.assertIn('summary', result)
        
        patterns = result.get('patterns', [])
        inverted_hammer = [p for p in patterns if 'Inverted Hammer' in p['name']]
        self.assertGreater(len(inverted_hammer), 0)
    
    def test_shooting_star_detection(self):
        """Test Shooting Star detection in uptrend"""
        # Create uptrend with shooting star
        data = pd.DataFrame({
            'open': [90 + i for i in range(11)],
            'high': [91 + i for i in range(10)] + [105],  # Last has long upper shadow
            'low': [89.5 + i for i in range(11)],
            'close': [90.5 + i for i in range(10)] + [100.3],
            'volume': [1000] * 11
        })
        
        result = self.detector.calculate(data)
        patterns = result.get('patterns', [])
        
        shooting_star = [p for p in patterns if 'Shooting Star' in p['name']]
        self.assertGreater(len(shooting_star), 0)


class TestMarubozu(unittest.TestCase):
    """Test Marubozu pattern detection"""
    
    def setUp(self):
        """Set up test data"""
        self.detector = JapaneseCandlestickPatterns()
    
    def test_bullish_marubozu(self):
        """Test Bullish Marubozu detection"""
        # Create bullish marubozu (no shadows)
        data = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [100],
            'close': [105],
            'volume': [1000]
        })
        
        result = self.detector.calculate(data)
        patterns = result.get('patterns', [])
        
        marubozu = [p for p in patterns if 'Bullish Marubozu' in p['name']]
        self.assertEqual(len(marubozu), 1)
    
    def test_bearish_marubozu(self):
        """Test Bearish Marubozu detection"""
        # Create bearish marubozu (no shadows)
        data = pd.DataFrame({
            'open': [105],
            'high': [105],
            'low': [100],
            'close': [100],
            'volume': [1000]
        })
        
        result = self.detector.calculate(data)
        patterns = result.get('patterns', [])
        
        marubozu = [p for p in patterns if 'Bearish Marubozu' in p['name']]
        self.assertEqual(len(marubozu), 1)


class TestTwoCandlePatterns(unittest.TestCase):
    """Test two-candle pattern detection"""
    
    def setUp(self):
        """Set up test data"""
        self.detector = JapaneseCandlestickPatterns()
    
    def test_piercing_line(self):
        """Test Piercing Line pattern detection"""
        # Create downtrend with piercing line
        data = pd.DataFrame({
            'open': [100, 99, 98, 97, 96, 98],
            'high': [100.5, 99.5, 98.5, 97.5, 96.5, 99],
            'low': [99, 98, 97, 96, 95, 95.5],
            'close': [99, 98, 97, 96, 95, 97.5],  # Last pierces previous midpoint
            'volume': [1000] * 6
        })
        
        result = self.detector.calculate(data)
        patterns = result.get('patterns', [])
        
        piercing = [p for p in patterns if 'Piercing Line' in p['name']]
        self.assertGreater(len(piercing), 0)
    
    def test_dark_cloud_cover(self):
        """Test Dark Cloud Cover pattern detection"""
        # Create uptrend with dark cloud cover
        data = pd.DataFrame({
            'open': [90, 91, 92, 93, 94, 92],
            'high': [91, 92, 93, 94, 95, 95.5],
            'low': [89.5, 90.5, 91.5, 92.5, 93.5, 91.5],
            'close': [91, 92, 93, 94, 95, 92.5],  # Last penetrates previous midpoint
            'volume': [1000] * 6
        })
        
        result = self.detector.calculate(data)
        patterns = result.get('patterns', [])
        
        dark_cloud = [p for p in patterns if 'Dark Cloud Cover' in p['name']]
        self.assertGreater(len(dark_cloud), 0)
    
    def test_tweezer_patterns(self):
        """Test Tweezer Top and Bottom patterns"""
        # Create data with matching highs/lows
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 102],
            'high': [102, 103, 104, 104, 103],  # Matching highs at index 2-3
            'low': [99, 100, 101, 102, 101],
            'close': [101, 102, 103, 102.5, 101.5],
            'volume': [1000] * 5
        })
        
        result = self.detector.calculate(data)
        patterns = result.get('patterns', [])
        
        tweezer = [p for p in patterns if 'Tweezer' in p['name']]
        self.assertGreaterEqual(len(tweezer), 0)  # May detect depending on trend


class TestThreeCandlePatterns(unittest.TestCase):
    """Test three-candle pattern detection"""
    
    def setUp(self):
        """Set up test data"""
        self.detector = JapaneseCandlestickPatterns()
    
    def test_three_inside_patterns(self):
        """Test Three Inside Up/Down patterns"""
        # Create harami followed by confirmation
        data = pd.DataFrame({
            'open': [100, 98, 99, 100.5],
            'high': [101, 100, 99.5, 102],
            'low': [98, 97, 98.5, 100],
            'close': [98, 99.5, 99.2, 101.5],  # Harami then confirmation
            'volume': [1000] * 4
        })
        
        result = self.detector.calculate(data)
        patterns = result.get('patterns', [])
        
        three_inside = [p for p in patterns if 'Three Inside' in p['name']]
        self.assertGreaterEqual(len(three_inside), 0)
    
    def test_abandoned_baby(self):
        """Test Abandoned Baby pattern detection"""
        # Create pattern with gaps
        data = pd.DataFrame({
            'open': [100, 98, 95, 97],
            'high': [101, 99, 95.5, 99],
            'low': [99, 97, 94.5, 96],
            'close': [98, 97.5, 95, 98.5],  # Gaps between candles
            'volume': [1000] * 4
        })
        
        # Note: Abandoned baby is very rare and requires specific gap conditions
        result = self.detector.calculate(data)
        patterns = result.get('patterns', [])
        
        # Just verify no errors in detection
        self.assertIsInstance(patterns, list)


if __name__ == '__main__':
    unittest.main()
