import pandas as pd
from datetime import datetime
from engines.momentum.correlation_momentum import (
    DynamicCorrelationIndicator,
    RelativeMomentumIndicator,
)
from engines.pattern.abandoned_baby_pattern import AbandonedBabyPatternEngine
from engines.pattern.three_line_strike_pattern import ThreeLineStrikePatternEngine
from models.market_data import (
    OHLCV,
)  # Ensure this import is correct based on your project structure

# Test data
test_data_df = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        ),
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "volume": [1000, 1010, 1020, 1030, 1040],
    }
)

# Create a more extensive test_data_df for indicators needing more data points
periods = 50
test_data_long_df = pd.DataFrame(
    {
        "timestamp": pd.date_range("2024-01-01", periods=periods, freq="D"),
        "open": [100 + i for i in range(periods)],
        "high": [105 + i for i in range(periods)],
        "low": [95 + i for i in range(periods)],
        "close": [102 + i for i in range(periods)],
        "volume": [1000 + i * 10 for i in range(periods)],
    }
)

print("Testing fixed indicators...")

try:
    dci = DynamicCorrelationIndicator(
        config={"period": 5, "correlation_period": 5}
    )  # Added default config
    result = dci.calculate(test_data_long_df.copy())  # Use the longer dataframe
    print(f"✅ DynamicCorrelationIndicator: PASS, Result: {str(result)[:50]}...")
except Exception as e:
    print(f"❌ DynamicCorrelationIndicator: {type(e).__name__} - {str(e)[:100]}...")

try:
    rmi = RelativeMomentumIndicator(config={"period": 5})  # Added default config
    result = rmi.calculate(test_data_long_df.copy())  # Use the longer dataframe
    print(f"✅ RelativeMomentumIndicator: PASS, Result: {str(result)[:50]}...")
except Exception as e:
    print(f"❌ RelativeMomentumIndicator: {type(e).__name__} - {str(e)[:100]}...")

# Convert DataFrame to list of OHLCV objects for pattern engines
ohlcv_list = []
for _, row in test_data_long_df.iterrows():
    ohlcv_list.append(
        OHLCV(
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
    )

try:
    ab = AbandonedBabyPatternEngine(config={})  # Added default config
    result = ab.detect_abandoned_baby(ohlcv_list)
    print(f"✅ AbandonedBabyPatternEngine: PASS, Result: {str(result)[:50]}...")
except Exception as e:
    print(f"❌ AbandonedBabyPatternEngine: {type(e).__name__} - {str(e)[:100]}...")

try:
    tls = ThreeLineStrikePatternEngine(config={})  # Added default config
    # Ensure enough candles are passed for validation if it checks len(pattern_data['candles'])
    pattern_data_tls = {"candles": ohlcv_list[:4]}
    is_valid = tls.validate_pattern(pattern_data_tls)
    print(f"✅ ThreeLineStrikePatternEngine: PASS, Valid: {is_valid}")
except Exception as e:
    print(f"❌ ThreeLineStrikePatternEngine: {type(e).__name__} - {str(e)[:100]}...")

print("Fix validation completed!")
