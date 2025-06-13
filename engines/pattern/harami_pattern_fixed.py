# -*- coding: utf-8 -*-

# Platform3 path management
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

"""
Harami Pattern Identifier - Japanese Candlestick Pattern Recognition
Platform3 Enhanced Technical Analysis Engine

Detects Bullish and Bearish Harami patterns, which are two-candle reversal patterns
where a small candle is contained within the body of the previous larger candle.
The word "harami" means "pregnant" in Japanese, as the pattern resembles a 
pregnant woman.

Pattern Characteristics:
- Two-candle pattern
- First candle has a large body
- Second candle's body is completely contained within the first candle's body
- Opposite colors (usually)
- Indicates potential trend reversal or consolidation

Key Features:
- Bullish and bearish harami detection
- Trend context analysis
- Volume confirmation
- Pattern strength measurement
- Reversal probability scoring
- Support/resistance level validation

Trading Applications:
- Trend reversal identification
- Consolidation phase detection
- Entry/exit timing optimization
- Risk management enhancement
- Market indecision analysis

Mathematical Foundation:
- Containment Condition: Body2 completely within Body1
- Size Ratio: Body2_Size / Body1_Size < threshold
- Trend Context: Direction of preceding trend
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from engines.indicator_base import IndicatorBase, IndicatorResult, IndicatorType, TimeFrame, IndicatorSignal, SignalType