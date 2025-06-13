# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Hammer/Hanging Man Detector - Japanese Candlestick Pattern Recognition
Platform3 Enhanced Technical Analysis Engine

Detects Hammer and Hanging Man candlestick patterns, which are critical reversal
indicators in technical analysis. These patterns have identical appearance but
different meanings based on trend context.

Pattern Characteristics:
- Small real body at upper end of trading range
- Long lower shadow (at least twice the body size)
- Little to no upper shadow
- Can be bullish (Hammer) or bearish (Hanging Man) depending on context

Key Features:
- Trend context analysis
- Pattern strength measurement
- Volume confirmation
- Reversal probability scoring
- Support/resistance level validation
- Multiple timeframe coordination

Trading Applications:
- Trend reversal identification
- Support level confirmation
- Entry/exit timing optimization
- Risk management enhancement
- Market sentiment analysis

Mathematical Foundation:
- Body Size = |Close - Open|
- Lower Shadow = Min(Open, Close) - Low
- Upper Shadow = High - Max(Open, Close)
- Body Position = (Max(Open, Close) - Low) / (High - Low)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame, IndicatorSignal, SignalType