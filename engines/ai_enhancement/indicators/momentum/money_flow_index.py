"""
Money Flow Index (MFI) Indicator
Trading-grade implementation for Platform3

The Money Flow Index is a volume-weighted momentum indicator that measures
buying and selling pressure. It ranges from 0 to 100 and uses both price
and volume to identify overbought and oversold conditions.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Try absolute import first, fall back to relative
try:
    from engines.ai_enhancement.indicators.base_indicator import (
        IndicatorMetadata,
        IndicatorValidationError,
        StandardIndicatorInterface,
    )
except ImportError:
    from ..base_indicator import (
        IndicatorMetadata,
        IndicatorValidationError,
        StandardIndicatorInterface,
    )

logger = logging.getLogger(__name__)


class MoneyFlowIndexIndicator(StandardIndicatorInterface):
    """
    Money Flow Index (MFI) - Volume-Weighted Momentum Indicator

    Formula:
    1. Typical Price = (High + Low + Close) / 3
    2. Raw Money Flow = Typical Price × Volume
    3. Positive Money Flow = Sum of Raw Money Flow when Typical Price increases
    4. Negative Money Flow = Sum of Raw Money Flow when Typical Price decreases
    5. Money Flow Ratio = Positive Money Flow / Negative Money Flow
    6. MFI = 100 - (100 / (1 + Money Flow Ratio))

    The MFI oscillates between 0 and 100:
    - Values above 80 indicate overbought conditions
    - Values below 20 indicate oversold conditions
    - Values above 50 suggest buying pressure
    - Values below 50 suggest selling pressure
    """

    CATEGORY = "momentum"
    VERSION = "1.0.0"
    AUTHOR = "Platform3"

    def __init__(self, period: int = 14, **kwargs):
        """
        Initialize Money Flow Index

        Args:
            period: Period for MFI calculation (default: 14)
        """
        super().__init__(period=period, **kwargs)

    def validate_parameters(self) -> bool:
        """Validate MFI parameters"""
        period = self.parameters.get("period", 14)

        if not isinstance(period, int) or period < 1:
            raise IndicatorValidationError(
                f"period must be positive integer, got {period}"
            )

        if period > 1000:  # Reasonable upper limit
            raise IndicatorValidationError(
                f"period too large, maximum 1000, got {period}"
            )

        return True

    def _get_required_columns(self) -> List[str]:
        """MFI requires OHLCV data"""
        return ["high", "low", "close", "volume"]

    def _get_minimum_data_points(self) -> int:
        """Minimum data points needed for calculation"""
        return self.parameters.get("period", 14) + 1

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Money Flow Index

        Args:
            data: DataFrame with 'high', 'low', 'close', 'volume' columns

        Returns:
            pd.Series: MFI values with same index as input data
        """
        # Validate input data
        self.validate_input_data(data)

        period = self.parameters.get("period", 14)

        # Extract OHLCV data
        highs = data["high"]
        lows = data["low"]
        closes = data["close"]
        volumes = data["volume"]

        # Calculate typical price
        typical_prices = (highs + lows + closes) / 3

        # Calculate raw money flow
        raw_money_flow = typical_prices * volumes

        # Identify positive and negative money flows
        price_changes = typical_prices.diff()

        # Create masks for positive and negative flows
        positive_mask = price_changes > 0
        negative_mask = price_changes < 0

        # Calculate positive and negative money flows
        positive_flows = raw_money_flow.where(positive_mask, 0)
        negative_flows = raw_money_flow.where(negative_mask, 0)

        # Calculate rolling sums for the period
        positive_mf = positive_flows.rolling(window=period).sum()
        negative_mf = negative_flows.rolling(window=period).sum()

        # Calculate Money Flow Ratio
        # Handle division by zero case
        money_ratio = positive_mf / negative_mf.replace(0, np.nan)

        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))

        # Handle cases where negative_mf is 0 (set MFI to 100)
        mfi = mfi.fillna(100)

        # Store calculation details for analysis
        self._last_calculation = {
            "typical_prices": typical_prices,
            "raw_money_flow": raw_money_flow,
            "positive_mf": positive_mf,
            "negative_mf": negative_mf,
            "money_ratio": money_ratio,
            "period": period,
        }

        return mfi

    def analyze_result(self, result: pd.Series) -> Dict[str, Any]:
        """
        Analyze MFI results and generate trading signals

        Args:
            result: MFI values from calculate()

        Returns:
            Dict containing analysis and signals
        """
        if result is None or len(result) == 0:
            return {"error": "No MFI data available for analysis"}

        # Get the last valid MFI value
        current_mfi = result.dropna().iloc[-1] if not result.dropna().empty else None

        if current_mfi is None:
            return {"error": "No valid MFI values available"}

        # Get recent values for trend analysis
        recent_values = result.dropna().tail(5).tolist()

        # Determine market conditions
        overbought = current_mfi > 80
        oversold = current_mfi < 20

        # Determine signal based on MFI level
        if overbought:
            signal = "sell"
            signal_strength = min(100, (current_mfi - 80) * 5)  # Scale 80-100 to 0-100
        elif oversold:
            signal = "buy"
            signal_strength = min(100, (20 - current_mfi) * 5)  # Scale 0-20 to 100-0
        elif current_mfi > 50:
            signal = "bullish"
            signal_strength = (current_mfi - 50) * 2  # Scale 50-100 to 0-100
        else:
            signal = "bearish"
            signal_strength = (50 - current_mfi) * 2  # Scale 0-50 to 100-0

        # Determine trend from recent values
        if len(recent_values) >= 3:
            if recent_values[-1] > recent_values[-2] > recent_values[-3]:
                trend = "bullish"
            elif recent_values[-1] < recent_values[-2] < recent_values[-3]:
                trend = "bearish"
            else:
                trend = "sideways"
        else:
            trend = "sideways"

        # Calculate confidence based on signal strength and trend consistency
        trend_consistency = 0
        if len(recent_values) >= 2:
            changes = [
                recent_values[i] - recent_values[i - 1]
                for i in range(1, len(recent_values))
            ]
            if changes:
                same_direction = sum(
                    1
                    for i in range(1, len(changes))
                    if (changes[i] > 0) == (changes[i - 1] > 0)
                )
                trend_consistency = (
                    float(same_direction) / (len(changes) - 1)
                    if len(changes) > 1
                    else 0.5
                )

        confidence = int((signal_strength / 100 * 0.7 + trend_consistency * 0.3) * 100)

        return {
            "value": float(current_mfi),
            "signal": signal,
            "trend": trend,
            "strength": float(signal_strength),
            "confidence": confidence,
            "overbought": bool(overbought),
            "oversold": bool(oversold),
            "recent_values": [float(x) for x in recent_values],
            "period": self.parameters.get("period", 14),
        }

    def get_metadata(self) -> IndicatorMetadata:
        """Return indicator metadata"""
        return IndicatorMetadata(
            name="Money Flow Index",
            category=self.CATEGORY,
            description="Volume-weighted momentum indicator measuring buying/selling pressure",
            parameters={
                "period": {
                    "type": "int",
                    "default": 14,
                    "description": "Period for MFI calculation",
                }
            },
            input_requirements=["high", "low", "close", "volume"],
            output_type="pd.Series",
            version=self.VERSION,
            author=self.AUTHOR,
            min_data_points=self.parameters.get("period", 14) + 1,
        )
