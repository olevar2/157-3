# Platform3 Remaining Issues Implementation Plan

## 1. Pattern Indicator Parameter Fixes

### For each pattern indicator with syntax errors:

#### 1.1 Doji Recognition Fix

```python
# Current problematic signature
def update(self, open_price: float, high: float, low: float, close: float, 
           volume: float = 0, timestamp: Optional[pd.Timestamp] = None,
           calculation_time_ms: float = 0, doji_type: DojiType) -> DojiRecognitionResult:
    # ...

# Fixed signature - move non-default doji_type before parameters with defaults
def update(self, open_price: float, high: float, low: float, close: float, 
           doji_type: DojiType, volume: float = 0, 
           timestamp: Optional[pd.Timestamp] = None,
           calculation_time_ms: float = 0) -> DojiRecognitionResult:
    # ...
```

#### 1.2 Engulfing Pattern Fix

```python
# Current problematic signature
def update(self, open_price: float, high: float, low: float, close: float, 
           volume: float = 0, timestamp: Optional[pd.Timestamp] = None,
           pattern_type: EngulfingType) -> EngulfingPatternResult:
    # ...

# Fixed signature
def update(self, open_price: float, high: float, low: float, close: float, 
           pattern_type: EngulfingType, volume: float = 0, 
           timestamp: Optional[pd.Timestamp] = None) -> EngulfingPatternResult:
    # ...
```

#### 1.3 Hammer/Hanging Man Fix

```python
# Current problematic signature
def update(self, open_price: float, high: float, low: float, close: float, 
           volume: float = 0, timestamp: Optional[pd.Timestamp] = None,
           pattern_type: HammerType) -> HammerResult:
    # ...

# Fixed signature
def update(self, open_price: float, high: float, low: float, close: float, 
           pattern_type: HammerType, volume: float = 0,
           timestamp: Optional[pd.Timestamp] = None) -> HammerResult:
    # ...
```

#### 1.4 Harami Pattern Fix

```python
# Current problematic signature
def update(self, open_price: float, high: float, low: float, close: float, 
           volume: float = 0, timestamp: Optional[pd.Timestamp] = None,
           pattern_type: HaramiType) -> HaramiResult:
    # ...

# Fixed signature
def update(self, open_price: float, high: float, low: float, close: float, 
           pattern_type: HaramiType, volume: float = 0,
           timestamp: Optional[pd.Timestamp] = None) -> HaramiResult:
    # ...
```

## 2. SentimentAnalyzer Module Implementation

Create a new file: `engines/sentiment/SentimentAnalyzer.py`

```python
# -*- coding: utf-8 -*-
"""
Platform3 Sentiment Analyzer Module
==================================

Core sentiment analysis functionality for social media and news data.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

from engines.indicator_base import IndicatorBase, IndicatorSignal

class SentimentSource(Enum):
    NEWS = auto()
    TWITTER = auto()
    REDDIT = auto()
    STOCKTWITS = auto()
    CUSTOM = auto()

class SentimentScore:
    def __init__(self, 
                 polarity: float, 
                 confidence: float,
                 source: SentimentSource,
                 raw_text: Optional[str] = None,
                 timestamp: Optional[pd.Timestamp] = None):
        self.polarity = polarity  # -1.0 to 1.0
        self.confidence = confidence  # 0.0 to 1.0
        self.source = source
        self.raw_text = raw_text
        self.timestamp = timestamp or pd.Timestamp.now()

class SentimentAnalyzer(IndicatorBase):
    """
    Core sentiment analysis engine for various text data sources.
    Provides standardized sentiment scoring across different sources.
    """
    
    def __init__(self, name: str = "Sentiment Analyzer"):
        """Initialize Sentiment Analyzer with default parameters"""
        super().__init__(name)
        self.sentiment_history: List[SentimentScore] = []
        self.current_sentiment: Optional[SentimentScore] = None
        
    def analyze_text(self, text: str, 
                    source: SentimentSource = SentimentSource.CUSTOM) -> SentimentScore:
        """
        Analyze raw text and return sentiment score
        Uses simple rule-based sentiment analysis as fallback
        """
        # Basic rule-based sentiment analysis
        # In production would use more sophisticated NLP
        positive_words = ['buy', 'bullish', 'up', 'growth', 'profit',
                         'positive', 'strong', 'higher', 'gain',
                         'good', 'great', 'excellent', 'increase']
        
        negative_words = ['sell', 'bearish', 'down', 'decline', 'loss',
                         'negative', 'weak', 'lower', 'fall',
                         'bad', 'terrible', 'poor', 'decrease']
        
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            polarity = 0.0
            confidence = 0.3  # Low confidence for neutral sentiment
        else:
            polarity = (pos_count - neg_count) / (pos_count + neg_count)
            confidence = min(0.7, (pos_count + neg_count) / len(text.split()) * 2)
            
        score = SentimentScore(
            polarity=polarity,
            confidence=confidence,
            source=source,
            raw_text=text
        )
        
        self.current_sentiment = score
        self.sentiment_history.append(score)
        
        return score
        
    def get_aggregate_sentiment(self, 
                              window: int = 20) -> Dict[str, float]:
        """Get aggregate sentiment metrics from recent history"""
        if not self.sentiment_history or window <= 0:
            return {
                'polarity': 0.0,
                'confidence': 0.0,
                'volume': 0,
                'trend': 0.0
            }
            
        recent = self.sentiment_history[-window:]
        
        weighted_polarity = sum(s.polarity * s.confidence for s in recent)
        total_confidence = sum(s.confidence for s in recent)
        
        if total_confidence > 0:
            avg_polarity = weighted_polarity / total_confidence
        else:
            avg_polarity = 0.0
            
        # Calculate trend (change over time)
        if len(recent) > 5:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            
            first_polarity = sum(s.polarity * s.confidence for s in first_half)
            first_conf = sum(s.confidence for s in first_half)
            
            second_polarity = sum(s.polarity * s.confidence for s in second_half)
            second_conf = sum(s.confidence for s in second_half)
            
            if first_conf > 0 and second_conf > 0:
                first_avg = first_polarity / first_conf
                second_avg = second_polarity / second_conf
                trend = second_avg - first_avg
            else:
                trend = 0.0
        else:
            trend = 0.0
            
        return {
            'polarity': avg_polarity,
            'confidence': sum(s.confidence for s in recent) / len(recent),
            'volume': len(recent),
            'trend': trend
        }

    def reset(self) -> None:
        """Reset the analyzer state"""
        super().reset()
        self.sentiment_history = []
        self.current_sentiment = None
```

## 3. Validation Testing Plan

```python
# Testing script for fixed indicators

def test_volume_indicators():
    # Test accumulation_distribution.py
    from engines.volume.accumulation_distribution import AccumulationDistributionLine
    ad = AccumulationDistributionLine()
    
    # Test with sample data
    result = ad.update(open_price=100.0, high=105.0, low=98.0, close=103.0, volume=1000)
    print(f"A/D Line Result: {result}")
    
    # Test chaikin_money_flow.py
    from engines.volume.chaikin_money_flow import ChaikinMoneyFlow
    cmf = ChaikinMoneyFlow()
    
    # Test with sample data
    result = cmf.update(open_price=100.0, high=105.0, low=98.0, close=103.0, volume=1000)
    print(f"CMF Result: {result}")

def test_pattern_indicators():
    # Test each fixed pattern indicator
    from engines.pattern.doji_recognition import DojiRecognitionEngine
    doji = DojiRecognitionEngine()
    
    # Test with sample data (assuming fixed parameter order)
    # result = doji.update(open_price=100.0, high=100.5, low=99.5, close=100.1, doji_type=None)
    # print(f"Doji Result: {result}")

# Execute tests after fixes are applied
test_volume_indicators()
test_pattern_indicators()
```

## 4. Integration Testing Framework

```python
# High-level integration test plan

def test_indicator_to_ai_agent_pipeline():
    # Load key indicators from each category
    from engines.momentum.rsi import RelativeStrengthIndex
    from engines.trend.bollinger_bands import BollingerBands
    from engines.volume.obv import OnBalanceVolume
    
    # Initialize indicators
    rsi = RelativeStrengthIndex()
    bb = BollingerBands()
    obv = OnBalanceVolume()
    
    # Process sample data
    sample_data = [
        # timestamp, open, high, low, close, volume
        (pd.Timestamp.now(), 100.0, 105.0, 98.0, 103.0, 1000),
        (pd.Timestamp.now(), 103.0, 106.0, 101.0, 102.0, 1200),
        (pd.Timestamp.now(), 102.0, 104.0, 100.0, 104.0, 900),
    ]
    
    results = []
    for ts, o, h, l, c, v in sample_data:
        rsi_result = rsi.update(c, timestamp=ts)
        bb_result = bb.update(c, timestamp=ts)
        obv_result = obv.update(c, v, timestamp=ts)
        
        results.append({
            'timestamp': ts,
            'indicators': {
                'rsi': rsi_result,
                'bollinger_bands': bb_result,
                'obv': obv_result
            }
        })
    
    # Verify results can be used by AI agent
    # This would integrate with the actual AI agent components
    return results
```