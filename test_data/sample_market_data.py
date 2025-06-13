#!/usr/bin/env python3
"""
Sample Market Data Generator for Indicator Testing

Provides realistic market data for testing all 157+ indicators during implementation.
Includes various market conditions, timeframes, and scenarios.
"""

import random
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd


class SampleMarketDataGenerator:
    """Generates realistic market data for indicator testing"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    def generate_basic_ohlcv(
        self, length: int = 1000, start_price: float = 100.0, volatility: float = 0.02
    ) -> pd.DataFrame:
        """Generate basic OHLCV data with realistic price movements"""

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=length), periods=length, freq="D"
        )

        # Generate realistic price series using geometric Brownian motion
        returns = np.random.normal(0, volatility, length)
        log_prices = np.cumsum(returns)
        close_prices = start_price * np.exp(log_prices)

        # Generate OHLC from close prices
        data = []
        for i in range(length):
            close = close_prices[i]

            # Generate realistic daily range
            daily_range = abs(np.random.normal(0, volatility * close * 0.5))

            # High and Low
            high = close + random.uniform(0, daily_range)
            low = close - random.uniform(0, daily_range)

            # Open (related to previous close)
            if i == 0:
                open_price = close
            else:
                gap = np.random.normal(0, volatility * close * 0.2)
                open_price = close_prices[i - 1] + gap

            # Volume (realistic pattern)
            base_volume = 1000000
            volume_factor = 1 + abs(returns[i]) * 10  # Higher volume on big moves
            volume = int(base_volume * volume_factor * (0.5 + random.random()))

            data.append(
                {
                    "timestamp": dates[i],
                    "open": max(open_price, 0.01),
                    "high": max(high, open_price, close),
                    "low": min(low, open_price, close),
                    "close": max(close, 0.01),
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)

        # Ensure OHLC constraints
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        return df

    def generate_trending_data(
        self, length: int = 500, trend_strength: float = 0.001
    ) -> pd.DataFrame:
        """Generate data with a clear trend for trend-following indicators"""
        base_data = self.generate_basic_ohlcv(length)

        # Add trend component
        trend = np.linspace(0, trend_strength * length, length)
        for col in ["open", "high", "low", "close"]:
            base_data[col] *= 1 + trend

        return base_data

    def generate_sideways_data(
        self, length: int = 500, range_factor: float = 0.1
    ) -> pd.DataFrame:
        """Generate sideways/ranging market data for oscillator testing"""
        base_data = self.generate_basic_ohlcv(length, volatility=0.01)

        # Constrain to range
        price_center = base_data["close"].iloc[0]
        price_range = price_center * range_factor

        for col in ["open", "high", "low", "close"]:
            base_data[col] = price_center + (base_data[col] - price_center) * 0.2
            base_data[col] = np.clip(
                base_data[col], price_center - price_range, price_center + price_range
            )

        return base_data

    def generate_volatile_data(
        self, length: int = 300, volatility: float = 0.05
    ) -> pd.DataFrame:
        """Generate highly volatile data for volatility indicators"""
        return self.generate_basic_ohlcv(length, volatility=volatility)

    def generate_multiple_timeframes(self) -> Dict[str, pd.DataFrame]:
        """Generate data for multiple timeframes"""
        return {
            "1min": self.generate_basic_ohlcv(
                1440, volatility=0.001
            ),  # 1 day of 1-min data
            "5min": self.generate_basic_ohlcv(
                2016, volatility=0.003
            ),  # 1 week of 5-min data
            "1hour": self.generate_basic_ohlcv(
                720, volatility=0.01
            ),  # 1 month of hourly data
            "1day": self.generate_basic_ohlcv(
                252, volatility=0.02
            ),  # 1 year of daily data
            "1week": self.generate_basic_ohlcv(
                104, volatility=0.05
            ),  # 2 years of weekly data
        }

    def generate_test_scenarios(self) -> Dict[str, pd.DataFrame]:
        """Generate various market scenarios for comprehensive testing"""
        return {
            "trending_up": self.generate_trending_data(500, 0.002),
            "trending_down": self.generate_trending_data(500, -0.002),
            "sideways": self.generate_sideways_data(500),
            "high_volatility": self.generate_volatile_data(300, 0.06),
            "low_volatility": self.generate_volatile_data(300, 0.005),
            "normal_market": self.generate_basic_ohlcv(1000),
            "crash_scenario": self._generate_crash_scenario(),
            "recovery_scenario": self._generate_recovery_scenario(),
        }

    def _generate_crash_scenario(self) -> pd.DataFrame:
        """Generate market crash scenario"""
        # Normal data followed by crash
        normal = self.generate_basic_ohlcv(200)
        crash_returns = np.random.normal(-0.05, 0.03, 50)  # Heavy downside

        crash_data = []
        last_close = normal["close"].iloc[-1]

        for i, ret in enumerate(crash_returns):
            close = last_close * (1 + ret)
            # More realistic crash behavior
            gap_down = ret * 0.5
            open_price = last_close * (1 + gap_down)

            crash_data.append(
                {
                    "timestamp": normal["timestamp"].iloc[-1] + timedelta(days=i + 1),
                    "open": open_price,
                    "high": max(open_price, close) * 1.02,
                    "low": min(open_price, close) * 0.98,
                    "close": close,
                    "volume": normal["volume"].mean()
                    * (2 + abs(ret) * 20),  # High volume on crash
                }
            )
            last_close = close

        return pd.concat([normal, pd.DataFrame(crash_data)], ignore_index=True)

    def _generate_recovery_scenario(self) -> pd.DataFrame:
        """Generate market recovery scenario"""
        # Crash followed by recovery
        crash = self._generate_crash_scenario()
        recovery_returns = np.random.normal(0.02, 0.015, 100)  # Upside bias

        recovery_data = []
        last_close = crash["close"].iloc[-1]

        for i, ret in enumerate(recovery_returns):
            close = last_close * (1 + ret)

            recovery_data.append(
                {
                    "timestamp": crash["timestamp"].iloc[-1] + timedelta(days=i + 1),
                    "open": last_close,
                    "high": max(last_close, close) * 1.01,
                    "low": min(last_close, close) * 0.99,
                    "close": close,
                    "volume": crash["volume"].mean() * (1 + ret * 5),
                }
            )
            last_close = close

        return pd.concat([crash, pd.DataFrame(recovery_data)], ignore_index=True)


# Create test data files
if __name__ == "__main__":
    generator = SampleMarketDataGenerator()

    # Generate and save test scenarios
    scenarios = generator.generate_test_scenarios()
    timeframes = generator.generate_multiple_timeframes()

    # Save as CSV files
    for name, data in scenarios.items():
        data.to_csv(f"test_data_{name}.csv", index=False)
        print(f"Generated {name}: {len(data)} rows")

    for tf, data in timeframes.items():
        data.to_csv(f"test_data_{tf}.csv", index=False)
        print(f"Generated {tf}: {len(data)} rows")

    print("\nAll test data files generated successfully!")
