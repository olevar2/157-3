"""
Rapid Trendlines Module
Trend line breaks and continuations for swing trading
Optimized for fast trend line detection and breakout analysis.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import statistics
from scipy import stats


@dataclass
class TrendLine:
    """Trend line definition"""
    start_point: Tuple[float, float]  # (timestamp, price)
    end_point: Tuple[float, float]    # (timestamp, price)
    slope: float
    intercept: float
    line_type: str  # 'support', 'resistance', 'channel_upper', 'channel_lower'
    strength: float  # 0-1 based on touches and length
    touches: int
    creation_time: float
    last_touch_time: float
    r_squared: float  # Correlation coefficient


@dataclass
class TrendLineBreak:
    """Trend line breakout event"""
    trend_line: TrendLine
    break_type: str  # 'bullish_break', 'bearish_break', 'false_break'
    break_time: float
    break_price: float
    break_strength: float
    volume_confirmation: bool
    target_projection: float
    confirmation_candles: int


@dataclass
class TrendChannel:
    """Trend channel formed by parallel trend lines"""
    upper_line: TrendLine
    lower_line: TrendLine
    channel_width: float
    channel_angle: float
    channel_type: str  # 'ascending', 'descending', 'horizontal'
    reliability: float
    breakout_probability: Dict[str, float]


@dataclass
class RapidTrendlinesResult:
    """Rapid trend lines analysis result"""
    symbol: str
    timestamp: float
    timeframe: str
    active_trendlines: List[TrendLine]
    trend_channels: List[TrendChannel]
    recent_breaks: List[TrendLineBreak]
    trend_direction: str
    trend_strength: float
    key_levels: Dict[str, float]
    execution_signals: List[Dict[str, Union[str, float]]]


class RapidTrendlines:
    """
    Rapid Trendlines Engine for Swing Trading
    Provides fast trend line detection and breakout analysis for swing entries
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.ready = False
        
        # Trend line detection parameters
        self.min_touches = 2
        self.max_deviation_pips = 20  # Maximum deviation from trend line
        self.min_line_length_periods = 10
        self.max_line_age_periods = 100
        
        # Breakout detection parameters
        self.breakout_confirmation_pips = 15
        self.false_break_threshold = 0.5  # 50% retracement back to line
        self.min_breakout_strength = 0.6
        
        # Channel detection parameters
        self.channel_parallel_tolerance = 0.1  # 10% slope difference tolerance
        self.min_channel_width_pips = 30
        
        # Performance optimization
        self.trendline_cache: Dict[str, List[TrendLine]] = {}
        self.calculation_cache: Dict[str, deque] = {}

    async def initialize(self) -> bool:
        """Initialize the Rapid Trendlines engine"""
        try:
            self.logger.info("Initializing Rapid Trendlines Engine...")
            
            # Test trend line detection with sample data
            test_data = self._generate_test_data()
            test_result = await self._detect_trendlines(test_data)
            
            if test_result and len(test_result) > 0:
                self.ready = True
                self.logger.info("✅ Rapid Trendlines Engine initialized")
                return True
            else:
                raise Exception("Trend line detection test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Rapid Trendlines Engine initialization failed: {e}")
            return False

    async def analyze_trendlines(self, symbol: str, price_data: List[Dict], 
                               volume_data: Optional[List[Dict]] = None,
                               timeframe: str = 'H4') -> RapidTrendlinesResult:
        """
        Analyze trend lines for rapid breakout detection
        
        Args:
            symbol: Currency pair symbol
            price_data: List of OHLC data dictionaries
            volume_data: Optional volume data
            timeframe: Chart timeframe (default H4)
            
        Returns:
            RapidTrendlinesResult with trend line analysis
        """
        if not self.ready:
            raise Exception("Rapid Trendlines Engine not initialized")
            
        if len(price_data) < 30:
            raise Exception("Insufficient data for trend line analysis (minimum 30 periods)")
            
        try:
            start_time = time.time()
            
            # Extract price data
            closes = [float(data.get('close', 0)) for data in price_data]
            highs = [float(data.get('high', 0)) for data in price_data]
            lows = [float(data.get('low', 0)) for data in price_data]
            timestamps = [float(data.get('timestamp', time.time())) for data in price_data]
            
            # Detect active trend lines
            active_trendlines = await self._detect_trendlines(price_data)
            
            # Detect trend channels
            trend_channels = await self._detect_trend_channels(active_trendlines)
            
            # Detect recent breakouts
            recent_breaks = await self._detect_trendline_breaks(
                price_data, active_trendlines, volume_data
            )
            
            # Analyze overall trend direction and strength
            trend_direction, trend_strength = await self._analyze_trend_direction(closes, timestamps)
            
            # Identify key levels from trend lines
            key_levels = await self._identify_key_levels(active_trendlines, closes[-1])
            
            # Generate execution signals
            execution_signals = await self._generate_execution_signals(
                symbol, closes[-1], active_trendlines, recent_breaks, trend_direction
            )
            
            # Cache results for performance
            self.trendline_cache[symbol] = active_trendlines
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Trend line analysis for {symbol} completed in {execution_time:.2f}ms")
            
            return RapidTrendlinesResult(
                symbol=symbol,
                timestamp=time.time(),
                timeframe=timeframe,
                active_trendlines=active_trendlines,
                trend_channels=trend_channels,
                recent_breaks=recent_breaks,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                key_levels=key_levels,
                execution_signals=execution_signals
            )
            
        except Exception as e:
            self.logger.error(f"Trend line analysis failed for {symbol}: {e}")
            raise

    async def _detect_trendlines(self, price_data: List[Dict]) -> List[TrendLine]:
        """Detect trend lines from price data"""
        trendlines = []
        
        # Extract data arrays
        highs = [float(data.get('high', 0)) for data in price_data]
        lows = [float(data.get('low', 0)) for data in price_data]
        timestamps = [float(data.get('timestamp', time.time())) for data in price_data]
        
        # Find swing points for trend line construction
        swing_highs = await self._find_swing_points(highs, 'high')
        swing_lows = await self._find_swing_points(lows, 'low')
        
        # Detect resistance trend lines from swing highs
        resistance_lines = await self._construct_trendlines_from_points(
            swing_highs, timestamps, 'resistance'
        )
        trendlines.extend(resistance_lines)
        
        # Detect support trend lines from swing lows
        support_lines = await self._construct_trendlines_from_points(
            swing_lows, timestamps, 'support'
        )
        trendlines.extend(support_lines)
        
        # Filter and validate trend lines
        valid_trendlines = await self._validate_trendlines(trendlines, price_data)
        
        return valid_trendlines

    async def _find_swing_points(self, prices: List[float], point_type: str) -> List[Tuple[int, float]]:
        """Find swing highs or lows in price data"""
        swing_points = []
        lookback = 3  # Periods to look back for swing detection
        
        for i in range(lookback, len(prices) - lookback):
            if point_type == 'high':
                # Check for swing high
                is_swing = all(prices[i] >= prices[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            else:  # low
                # Check for swing low
                is_swing = all(prices[i] <= prices[j] for j in range(i - lookback, i + lookback + 1) if j != i)
            
            if is_swing:
                swing_points.append((i, prices[i]))
        
        return swing_points

    async def _construct_trendlines_from_points(self, swing_points: List[Tuple[int, float]], 
                                              timestamps: List[float], 
                                              line_type: str) -> List[TrendLine]:
        """Construct trend lines from swing points"""
        trendlines = []
        
        if len(swing_points) < 2:
            return trendlines
        
        # Try all combinations of swing points to form trend lines
        for i in range(len(swing_points)):
            for j in range(i + 1, len(swing_points)):
                point1_idx, point1_price = swing_points[i]
                point2_idx, point2_price = swing_points[j]
                
                # Skip if points are too close
                if point2_idx - point1_idx < self.min_line_length_periods:
                    continue
                
                # Calculate trend line parameters
                x1, y1 = timestamps[point1_idx], point1_price
                x2, y2 = timestamps[point2_idx], point2_price
                
                # Calculate slope and intercept
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    
                    # Count touches and calculate strength
                    touches = await self._count_trendline_touches(
                        timestamps, [float(data.get('high' if line_type == 'resistance' else 'low', 0)) 
                                   for data in self._get_price_data_from_timestamps(timestamps)],
                        slope, intercept, line_type
                    )
                    
                    if touches >= self.min_touches:
                        # Calculate R-squared for line fit
                        r_squared = await self._calculate_r_squared(
                            swing_points[i:j+1], slope, intercept, timestamps
                        )
                        
                        strength = min(touches / 5.0, 1.0) * r_squared
                        
                        trendlines.append(TrendLine(
                            start_point=(x1, y1),
                            end_point=(x2, y2),
                            slope=slope,
                            intercept=intercept,
                            line_type=line_type,
                            strength=strength,
                            touches=touches,
                            creation_time=x1,
                            last_touch_time=x2,
                            r_squared=r_squared
                        ))
        
        return trendlines

    async def _count_trendline_touches(self, timestamps: List[float], prices: List[float],
                                     slope: float, intercept: float, line_type: str) -> int:
        """Count how many times price touched the trend line"""
        touches = 0
        
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Calculate expected price on trend line
            expected_price = slope * timestamp + intercept
            
            # Calculate deviation in pips
            deviation_pips = abs(price - expected_price) * 10000
            
            # Check if price touched the line within tolerance
            if deviation_pips <= self.max_deviation_pips:
                # Additional check for line type
                if line_type == 'resistance' and price <= expected_price:
                    touches += 1
                elif line_type == 'support' and price >= expected_price:
                    touches += 1
        
        return touches

    async def _calculate_r_squared(self, points: List[Tuple[int, float]], 
                                 slope: float, intercept: float, 
                                 timestamps: List[float]) -> float:
        """Calculate R-squared for trend line fit"""
        if len(points) < 2:
            return 0.0
        
        # Extract actual prices and corresponding timestamps
        actual_prices = [point[1] for point in points]
        point_timestamps = [timestamps[point[0]] for point in points]
        
        # Calculate predicted prices
        predicted_prices = [slope * ts + intercept for ts in point_timestamps]
        
        # Calculate R-squared
        try:
            correlation_matrix = np.corrcoef(actual_prices, predicted_prices)
            correlation = correlation_matrix[0, 1]
            r_squared = correlation ** 2
            return max(0.0, min(1.0, r_squared))
        except:
            return 0.5  # Default value if calculation fails

    async def _validate_trendlines(self, trendlines: List[TrendLine], 
                                 price_data: List[Dict]) -> List[TrendLine]:
        """Validate and filter trend lines"""
        valid_trendlines = []
        current_time = time.time()
        
        for trendline in trendlines:
            # Check age
            age_hours = (current_time - trendline.creation_time) / 3600
            if age_hours > self.max_line_age_periods:
                continue
            
            # Check strength
            if trendline.strength < 0.5:
                continue
            
            # Check if line is still relevant (recent touches)
            time_since_last_touch = (current_time - trendline.last_touch_time) / 3600
            if time_since_last_touch > 48:  # 48 hours
                continue
            
            valid_trendlines.append(trendline)
        
        # Sort by strength and return top lines
        valid_trendlines.sort(key=lambda x: x.strength, reverse=True)
        return valid_trendlines[:10]  # Return top 10 strongest lines

    async def _detect_trend_channels(self, trendlines: List[TrendLine]) -> List[TrendChannel]:
        """Detect trend channels from parallel trend lines"""
        channels = []
        
        # Group trend lines by type
        support_lines = [tl for tl in trendlines if tl.line_type == 'support']
        resistance_lines = [tl for tl in trendlines if tl.line_type == 'resistance']
        
        # Find parallel lines to form channels
        for support_line in support_lines:
            for resistance_line in resistance_lines:
                # Check if lines are roughly parallel
                slope_diff = abs(support_line.slope - resistance_line.slope)
                slope_tolerance = max(abs(support_line.slope), abs(resistance_line.slope)) * self.channel_parallel_tolerance
                
                if slope_diff <= slope_tolerance:
                    # Calculate channel width
                    mid_time = (support_line.start_point[0] + support_line.end_point[0]) / 2
                    support_price = support_line.slope * mid_time + support_line.intercept
                    resistance_price = resistance_line.slope * mid_time + resistance_line.intercept
                    
                    channel_width_pips = abs(resistance_price - support_price) * 10000
                    
                    if channel_width_pips >= self.min_channel_width_pips:
                        # Determine channel type
                        avg_slope = (support_line.slope + resistance_line.slope) / 2
                        if avg_slope > 0.00001:
                            channel_type = 'ascending'
                        elif avg_slope < -0.00001:
                            channel_type = 'descending'
                        else:
                            channel_type = 'horizontal'
                        
                        # Calculate reliability
                        reliability = (support_line.strength + resistance_line.strength) / 2
                        
                        # Calculate breakout probabilities
                        breakout_probability = {
                            'upward': 0.6 if channel_type == 'ascending' else 0.4,
                            'downward': 0.6 if channel_type == 'descending' else 0.4
                        }
                        
                        channels.append(TrendChannel(
                            upper_line=resistance_line,
                            lower_line=support_line,
                            channel_width=channel_width_pips,
                            channel_angle=np.degrees(np.arctan(avg_slope)),
                            channel_type=channel_type,
                            reliability=reliability,
                            breakout_probability=breakout_probability
                        ))
        
        return channels

    async def _detect_trendline_breaks(self, price_data: List[Dict], 
                                     trendlines: List[TrendLine],
                                     volume_data: Optional[List[Dict]]) -> List[TrendLineBreak]:
        """Detect trend line breakouts"""
        breaks = []
        
        if len(price_data) < 5:
            return breaks
        
        recent_data = price_data[-10:]  # Look at recent 10 periods
        
        for trendline in trendlines:
            for i, data_point in enumerate(recent_data):
                timestamp = float(data_point.get('timestamp', time.time()))
                high = float(data_point.get('high', 0))
                low = float(data_point.get('low', 0))
                close = float(data_point.get('close', 0))
                
                # Calculate expected price on trend line
                expected_price = trendline.slope * timestamp + trendline.intercept
                
                # Check for breakout
                breakout_detected = False
                break_type = None
                break_price = None
                
                if trendline.line_type == 'resistance' and high > expected_price:
                    # Bullish breakout of resistance
                    break_strength = (high - expected_price) / expected_price
                    if break_strength * 10000 > self.breakout_confirmation_pips:
                        breakout_detected = True
                        break_type = 'bullish_break'
                        break_price = high
                
                elif trendline.line_type == 'support' and low < expected_price:
                    # Bearish breakout of support
                    break_strength = (expected_price - low) / expected_price
                    if break_strength * 10000 > self.breakout_confirmation_pips:
                        breakout_detected = True
                        break_type = 'bearish_break'
                        break_price = low
                
                if breakout_detected:
                    # Check for false breakout
                    is_false_break = await self._check_false_breakout(
                        recent_data[i:], trendline, break_type
                    )
                    
                    if is_false_break:
                        break_type = 'false_break'
                    
                    # Calculate target projection
                    target_projection = await self._calculate_breakout_target(
                        trendline, break_price, break_type
                    )
                    
                    # Count confirmation candles
                    confirmation_candles = await self._count_confirmation_candles(
                        recent_data[i:], expected_price, break_type
                    )
                    
                    breaks.append(TrendLineBreak(
                        trend_line=trendline,
                        break_type=break_type,
                        break_time=timestamp,
                        break_price=break_price,
                        break_strength=break_strength,
                        volume_confirmation=True,  # Simplified
                        target_projection=target_projection,
                        confirmation_candles=confirmation_candles
                    ))
        
        return breaks

    async def _check_false_breakout(self, subsequent_data: List[Dict], 
                                  trendline: TrendLine, break_type: str) -> bool:
        """Check if breakout is likely a false break"""
        if len(subsequent_data) < 3:
            return False
        
        # Check if price returned to trend line within next few periods
        for data_point in subsequent_data[1:4]:
            timestamp = float(data_point.get('timestamp', time.time()))
            close = float(data_point.get('close', 0))
            expected_price = trendline.slope * timestamp + trendline.intercept
            
            if break_type == 'bullish_break' and close < expected_price * (1 - self.false_break_threshold):
                return True
            elif break_type == 'bearish_break' and close > expected_price * (1 + self.false_break_threshold):
                return True
        
        return False

    async def _calculate_breakout_target(self, trendline: TrendLine, break_price: float, 
                                       break_type: str) -> float:
        """Calculate target price for breakout"""
        # Simple projection based on trend line angle and historical volatility
        line_range = abs(trendline.end_point[1] - trendline.start_point[1])
        
        if break_type == 'bullish_break':
            return break_price + (line_range * 0.5)
        elif break_type == 'bearish_break':
            return break_price - (line_range * 0.5)
        else:
            return break_price

    async def _count_confirmation_candles(self, subsequent_data: List[Dict], 
                                        expected_price: float, break_type: str) -> int:
        """Count confirmation candles after breakout"""
        confirmations = 0
        
        for data_point in subsequent_data[1:4]:  # Check next 3 candles
            close = float(data_point.get('close', 0))
            
            if break_type == 'bullish_break' and close > expected_price:
                confirmations += 1
            elif break_type == 'bearish_break' and close < expected_price:
                confirmations += 1
        
        return confirmations

    async def _analyze_trend_direction(self, closes: List[float], 
                                     timestamps: List[float]) -> Tuple[str, float]:
        """Analyze overall trend direction and strength"""
        if len(closes) < 20:
            return 'neutral', 0.5
        
        # Use linear regression on recent prices
        recent_closes = closes[-20:]
        x_values = list(range(len(recent_closes)))
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, recent_closes)
            
            # Determine direction
            if slope > 0.0001:
                direction = 'bullish'
            elif slope < -0.0001:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            # Calculate strength based on R-squared and slope magnitude
            strength = abs(r_value) * min(abs(slope) * 10000, 1.0)
            
            return direction, max(0.1, min(0.9, strength))
            
        except:
            return 'neutral', 0.5

    async def _identify_key_levels(self, trendlines: List[TrendLine], 
                                 current_price: float) -> Dict[str, float]:
        """Identify key levels from trend lines"""
        key_levels = {}
        current_time = time.time()
        
        # Find nearest support and resistance trend lines
        support_lines = [tl for tl in trendlines if tl.line_type == 'support']
        resistance_lines = [tl for tl in trendlines if tl.line_type == 'resistance']
        
        # Calculate current prices on trend lines
        if support_lines:
            support_prices = []
            for line in support_lines:
                current_line_price = line.slope * current_time + line.intercept
                if current_line_price < current_price:
                    support_prices.append(current_line_price)
            
            if support_prices:
                key_levels['nearest_support'] = max(support_prices)
        
        if resistance_lines:
            resistance_prices = []
            for line in resistance_lines:
                current_line_price = line.slope * current_time + line.intercept
                if current_line_price > current_price:
                    resistance_prices.append(current_line_price)
            
            if resistance_prices:
                key_levels['nearest_resistance'] = min(resistance_prices)
        
        return key_levels

    async def _generate_execution_signals(self, symbol: str, current_price: float,
                                        trendlines: List[TrendLine],
                                        recent_breaks: List[TrendLineBreak],
                                        trend_direction: str) -> List[Dict[str, Union[str, float]]]:
        """Generate execution signals based on trend line analysis"""
        signals = []
        
        # Signals from recent breakouts
        for breakout in recent_breaks:
            if (breakout.break_type in ['bullish_break', 'bearish_break'] and 
                breakout.break_strength >= self.min_breakout_strength and
                breakout.confirmation_candles >= 2):
                
                signal_type = 'buy' if breakout.break_type == 'bullish_break' else 'sell'
                
                signals.append({
                    'type': 'trendline_breakout',
                    'signal': signal_type,
                    'confidence': breakout.break_strength,
                    'entry_price': current_price,
                    'target_price': breakout.target_projection,
                    'stop_loss': breakout.trend_line.intercept + breakout.trend_line.slope * time.time(),
                    'timeframe': 'H4',
                    'breakout_type': breakout.break_type
                })
        
        # Signals from trend line bounces
        current_time = time.time()
        for trendline in trendlines:
            if trendline.strength > 0.7:
                line_price = trendline.slope * current_time + trendline.intercept
                distance_pips = abs(current_price - line_price) * 10000
                
                # Signal if price is near strong trend line
                if distance_pips <= 20:  # Within 20 pips
                    if trendline.line_type == 'support' and current_price >= line_price:
                        signals.append({
                            'type': 'trendline_bounce',
                            'signal': 'buy',
                            'confidence': trendline.strength,
                            'entry_price': current_price,
                            'target_price': line_price + (line_price * 0.01),
                            'stop_loss': line_price - (line_price * 0.005),
                            'timeframe': 'H4',
                            'line_type': 'support'
                        })
                    elif trendline.line_type == 'resistance' and current_price <= line_price:
                        signals.append({
                            'type': 'trendline_bounce',
                            'signal': 'sell',
                            'confidence': trendline.strength,
                            'entry_price': current_price,
                            'target_price': line_price - (line_price * 0.01),
                            'stop_loss': line_price + (line_price * 0.005),
                            'timeframe': 'H4',
                            'line_type': 'resistance'
                        })
        
        return signals

    def _generate_test_data(self) -> List[Dict]:
        """Generate test data for initialization"""
        test_data = []
        base_price = 1.1000
        
        # Create trending data with trend lines
        for i in range(50):
            trend = 0.0002 * i  # Uptrend
            noise = (np.random.random() - 0.5) * 0.001
            price = base_price + trend + noise
            
            test_data.append({
                'timestamp': time.time() - (50 - i) * 3600,
                'open': price,
                'high': price + 0.0005,
                'low': price - 0.0005,
                'close': price,
                'volume': 1000
            })
            
        return test_data

    def _get_price_data_from_timestamps(self, timestamps: List[float]) -> List[Dict]:
        """Helper method to get price data (simplified for testing)"""
        # This would normally retrieve actual price data
        # For testing, return dummy data
        return [{'high': 1.1000, 'low': 1.0990} for _ in timestamps]
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """
        Standard calculate interface for Universal Adapter compatibility
        
        Args:
            high: High prices as numpy array
            low: Low prices as numpy array
            close: Close prices as numpy array
            
        Returns:
            Dict with trendline analysis results
        """
        try:
            # Convert numpy arrays to the format expected by trendline detection
            price_data = []
            for i in range(len(close)):
                price_data.append({
                    'high': float(high[i]),
                    'low': float(low[i]),
                    'close': float(close[i]),
                    'timestamp': i  # Use index as timestamp for simplicity
                })
            
            # Detect trendlines using core logic
            trendlines = self._detect_trendlines_sync(price_data)
            
            # Analyze breakouts and signals
            current_price = float(close[-1])
            breakout_signals = self._analyze_breakouts_sync(trendlines, current_price, price_data)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(trendlines, price_data)
            
            return {
                'trendlines': trendlines,
                'support_lines': [tl for tl in trendlines if tl['type'] == 'support'],
                'resistance_lines': [tl for tl in trendlines if tl['type'] == 'resistance'],
                'breakout_signals': breakout_signals,
                'trend_strength': trend_strength,
                'current_trend': self._determine_trend_direction(trendlines),
                'nearest_support': self._find_nearest_level(trendlines, current_price, 'support'),
                'nearest_resistance': self._find_nearest_level(trendlines, current_price, 'resistance'),
                'active_trendlines': len([tl for tl in trendlines if tl['active']])
            }
            
        except Exception as e:
            self.logger.error(f"RapidTrendlines calculate error: {e}")
            return {
                'trendlines': [],
                'support_lines': [],
                'resistance_lines': [],
                'breakout_signals': [],
                'trend_strength': 0.0,
                'current_trend': 'neutral',
                'nearest_support': None,
                'nearest_resistance': None,
                'active_trendlines': 0,
                'error': str(e)
            }

    def _detect_trendlines_sync(self, price_data: List[Dict]) -> List[Dict]:
        """Synchronous trendline detection for calculate method"""
        trendlines = []
        min_periods = 10
        
        if len(price_data) < min_periods:
            return trendlines
            
        # Detect support trendlines (connecting lows)
        support_lines = self._find_support_trendlines(price_data)
        trendlines.extend(support_lines)
        
        # Detect resistance trendlines (connecting highs)
        resistance_lines = self._find_resistance_trendlines(price_data)
        trendlines.extend(resistance_lines)
        
        return trendlines
    
    def _find_support_trendlines(self, price_data: List[Dict]) -> List[Dict]:
        """Find support trendlines by connecting significant lows"""
        support_lines = []
        lows = [(i, data['low']) for i, data in enumerate(price_data)]
        
        # Find local lows (simplified approach)
        local_lows = []
        for i in range(2, len(lows) - 2):
            if (lows[i][1] < lows[i-1][1] and lows[i][1] < lows[i-2][1] and 
                lows[i][1] < lows[i+1][1] and lows[i][1] < lows[i+2][1]):
                local_lows.append(lows[i])
        
        # Connect pairs of lows to form trendlines
        for i in range(len(local_lows) - 1):
            for j in range(i + 1, len(local_lows)):
                point1 = local_lows[i]
                point2 = local_lows[j]
                
                if point2[0] - point1[0] >= 5:  # Minimum distance between points
                    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
                    
                    # Check if line is approximately horizontal or slightly ascending (support)
                    if -0.001 <= slope <= 0.002:  # Allow slight positive slope for support
                        strength = self._calculate_trendline_strength(price_data, point1, point2, 'support')
                        
                        if strength > 0.3:  # Only keep reasonably strong trendlines
                            support_lines.append({
                                'type': 'support',
                                'point1': {'index': point1[0], 'price': point1[1]},
                                'point2': {'index': point2[0], 'price': point2[1]},
                                'slope': slope,
                                'strength': strength,
                                'active': self._is_trendline_active(point2[0], len(price_data)),
                                'touches': self._count_trendline_touches(price_data, point1, point2, 'support')
                            })
        
        return support_lines
    
    def _find_resistance_trendlines(self, price_data: List[Dict]) -> List[Dict]:
        """Find resistance trendlines by connecting significant highs"""
        resistance_lines = []
        highs = [(i, data['high']) for i, data in enumerate(price_data)]
        
        # Find local highs (simplified approach)
        local_highs = []
        for i in range(2, len(highs) - 2):
            if (highs[i][1] > highs[i-1][1] and highs[i][1] > highs[i-2][1] and 
                highs[i][1] > highs[i+1][1] and highs[i][1] > highs[i+2][1]):
                local_highs.append(highs[i])
        
        # Connect pairs of highs to form trendlines
        for i in range(len(local_highs) - 1):
            for j in range(i + 1, len(local_highs)):
                point1 = local_highs[i]
                point2 = local_highs[j]
                
                if point2[0] - point1[0] >= 5:  # Minimum distance between points
                    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
                    
                    # Check if line is approximately horizontal or slightly descending (resistance)
                    if -0.002 <= slope <= 0.001:  # Allow slight negative slope for resistance
                        strength = self._calculate_trendline_strength(price_data, point1, point2, 'resistance')
                        
                        if strength > 0.3:  # Only keep reasonably strong trendlines
                            resistance_lines.append({
                                'type': 'resistance',
                                'point1': {'index': point1[0], 'price': point1[1]},
                                'point2': {'index': point2[0], 'price': point2[1]},
                                'slope': slope,
                                'strength': strength,
                                'active': self._is_trendline_active(point2[0], len(price_data)),
                                'touches': self._count_trendline_touches(price_data, point1, point2, 'resistance')
                            })
        
        return resistance_lines
    
    def _calculate_trendline_strength(self, price_data: List[Dict], point1: tuple, point2: tuple, line_type: str) -> float:
        """Calculate the strength of a trendline based on how well price respects it"""
        try:
            touches = 0
            near_misses = 0
            violations = 0
            
            for i in range(point1[0], min(point2[0] + 10, len(price_data))):
                # Calculate expected price at this point on the trendline
                expected_price = point1[1] + (point2[1] - point1[1]) * (i - point1[0]) / (point2[0] - point1[0])
                
                if line_type == 'support':
                    low_price = price_data[i]['low']
                    price_diff = abs(low_price - expected_price) / expected_price
                    
                    if price_diff < 0.001:  # Very close to line
                        touches += 1
                    elif price_diff < 0.003:  # Near the line
                        near_misses += 1
                    elif low_price < expected_price * 0.995:  # Significantly below support
                        violations += 1
                        
                else:  # resistance
                    high_price = price_data[i]['high']
                    price_diff = abs(high_price - expected_price) / expected_price
                    
                    if price_diff < 0.001:  # Very close to line
                        touches += 1
                    elif price_diff < 0.003:  # Near the line
                        near_misses += 1
                    elif high_price > expected_price * 1.005:  # Significantly above resistance
                        violations += 1
            
            # Calculate strength score
            total_points = touches + near_misses + violations
            if total_points == 0:
                return 0.5
                
            strength = (touches * 2 + near_misses - violations * 2) / (total_points * 2)
            return max(0.0, min(1.0, strength))
            
        except Exception:
            return 0.5
    
    def _is_trendline_active(self, last_point_index: int, total_length: int) -> bool:
        """Check if trendline is still active (recent enough)"""
        return (total_length - last_point_index) <= 20  # Active if within last 20 periods
    
    def _count_trendline_touches(self, price_data: List[Dict], point1: tuple, point2: tuple, line_type: str) -> int:
        """Count how many times price touched the trendline"""
        touches = 2  # Start with 2 for the defining points
        
        for i in range(point1[0] + 1, min(point2[0] + 5, len(price_data))):
            expected_price = point1[1] + (point2[1] - point1[1]) * (i - point1[0]) / (point2[0] - point1[0])
            
            if line_type == 'support':
                if abs(price_data[i]['low'] - expected_price) / expected_price < 0.002:
                    touches += 1
            else:  # resistance
                if abs(price_data[i]['high'] - expected_price) / expected_price < 0.002:
                    touches += 1
                    
        return touches
    
    def _analyze_breakouts_sync(self, trendlines: List[Dict], current_price: float, price_data: List[Dict]) -> List[Dict]:
        """Analyze potential breakout signals"""
        signals = []
        
        for trendline in trendlines:
            if not trendline['active']:
                continue
                
            # Calculate current trendline level
            point1 = trendline['point1']
            point2 = trendline['point2']
            current_index = len(price_data) - 1
            
            if current_index > point2['index']:
                current_level = point2['price'] + trendline['slope'] * (current_index - point2['index'])
                
                distance_pct = abs(current_price - current_level) / current_level
                
                if distance_pct < 0.01:  # Within 1% of trendline
                    if trendline['type'] == 'support' and current_price > current_level * 1.002:
                        signals.append({
                            'type': 'buy',
                            'reason': 'Support breakout',
                            'level': current_level,
                            'strength': trendline['strength'],
                            'confidence': min(0.9, trendline['strength'] + 0.2)
                        })
                    elif trendline['type'] == 'resistance' and current_price < current_level * 0.998:
                        signals.append({
                            'type': 'sell',
                            'reason': 'Resistance breakout',
                            'level': current_level,
                            'strength': trendline['strength'],
                            'confidence': min(0.9, trendline['strength'] + 0.2)
                        })
        
        return signals
    
    def _calculate_trend_strength(self, trendlines: List[Dict], price_data: List[Dict]) -> float:
        """Calculate overall trend strength based on trendlines"""
        if not trendlines:
            return 0.0
            
        active_lines = [tl for tl in trendlines if tl['active']]
        if not active_lines:
            return 0.0
            
        total_strength = sum(tl['strength'] for tl in active_lines)
        return min(1.0, total_strength / len(active_lines))
    
    def _determine_trend_direction(self, trendlines: List[Dict]) -> str:
        """Determine trend direction based on trendlines"""
        if not trendlines:
            return 'neutral'
            
        active_support = [tl for tl in trendlines if tl['active'] and tl['type'] == 'support']
        active_resistance = [tl for tl in trendlines if tl['active'] and tl['type'] == 'resistance']
        
        support_strength = sum(tl['strength'] for tl in active_support)
        resistance_strength = sum(tl['strength'] for tl in active_resistance)
        
        if support_strength > resistance_strength * 1.5:
            return 'uptrend'
        elif resistance_strength > support_strength * 1.5:
            return 'downtrend'
        else:
            return 'neutral'
    
    def _find_nearest_level(self, trendlines: List[Dict], current_price: float, level_type: str) -> Optional[float]:
        """Find the nearest support or resistance level"""
        relevant_lines = [tl for tl in trendlines if tl['active'] and tl['type'] == level_type]
        
        if not relevant_lines:
            return None
            
        nearest_level = None
        min_distance = float('inf')
        
        for tl in relevant_lines:
            # Use the second point price as the level (simplified)
            level_price = tl['point2']['price']
            distance = abs(current_price - level_price)
            
            if distance < min_distance:
                min_distance = distance
                nearest_level = level_price
                
        return nearest_level
