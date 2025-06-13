"""
Statistical Analysis Indicators Stubs for Platform3
"""


class AutocorrelationIndicator:
    """Autocorrelation indicator - implements autocorrelation for pattern detection"""

    def __init__(self, period=20, max_lag=10):
        self.period = period
        self.max_lag = max_lag

    def calculate(self, data):
        """Calculate autocorrelation for pattern detection"""
        import numpy as np

        try:
            # Convert data to numpy array
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif hasattr(data, "values"):  # pandas
                prices = data.values.flatten()
            else:
                return None

            if len(prices) < self.period + self.max_lag:
                return None

            # Calculate rolling autocorrelation
            results = []
            for i in range(self.period + self.max_lag - 1, len(prices)):
                window = prices[i - self.period - self.max_lag + 1 : i + 1]

                # Remove trend from the window
                detrended = self._detrend(window)

                # Calculate autocorrelation for different lags
                autocorrs = []
                for lag in range(1, self.max_lag + 1):
                    if len(detrended) > lag:
                        # Calculate autocorrelation for this lag
                        x1 = detrended[:-lag]
                        x2 = detrended[lag:]

                        if len(x1) > 1 and len(x2) > 1:
                            corr = np.corrcoef(x1, x2)[0, 1]
                            autocorrs.append(0.0 if np.isnan(corr) else corr)
                        else:
                            autocorrs.append(0.0)
                    else:
                        autocorrs.append(0.0)

                # Find significant autocorrelations
                significant_lags = []
                for lag, autocorr in enumerate(autocorrs, 1):
                    if abs(autocorr) > 0.2:  # Threshold for significance
                        significant_lags.append({"lag": lag, "correlation": autocorr})

                result = {
                    "autocorrelations": autocorrs,
                    "max_correlation": max(autocorrs, key=abs) if autocorrs else 0.0,
                    "significant_lags": significant_lags,
                    "pattern_strength": (
                        np.mean([abs(ac) for ac in autocorrs]) if autocorrs else 0.0
                    ),
                }
                results.append(result)

            return results

        except Exception:
            return None

    def _detrend(self, data):
        """Remove linear trend from data"""
        import numpy as np

        try:
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, 1)
            trend = coeffs[0] * x + coeffs[1]
            return data - trend
        except Exception:
            return data


class BetaCoefficientIndicator:
    """Beta Coefficient indicator - implements beta calculation for market relationship"""

    def __init__(self, period=20):
        self.period = period

    def calculate(self, asset_data, market_data):
        """Calculate beta coefficient relative to market"""
        import numpy as np

        try:
            # Convert data to numpy arrays
            if isinstance(asset_data, (list, tuple)):
                asset_prices = np.array(asset_data)
            elif isinstance(asset_data, np.ndarray):
                asset_prices = asset_data.flatten()
            elif hasattr(asset_data, "values"):  # pandas
                asset_prices = asset_data.values.flatten()
            else:
                return None

            if isinstance(market_data, (list, tuple)):
                market_prices = np.array(market_data)
            elif isinstance(market_data, np.ndarray):
                market_prices = market_data.flatten()
            elif hasattr(market_data, "values"):  # pandas
                market_prices = market_data.values.flatten()
            else:
                return None

            # Ensure same length
            min_len = min(len(asset_prices), len(market_prices))
            asset_prices = asset_prices[:min_len]
            market_prices = market_prices[:min_len]

            if min_len < self.period + 1:
                return None

            # Calculate returns
            asset_returns = np.diff(asset_prices) / asset_prices[:-1]
            market_returns = np.diff(market_prices) / market_prices[:-1]

            # Calculate rolling beta
            results = []
            for i in range(self.period - 1, len(asset_returns)):
                asset_window = asset_returns[i - self.period + 1 : i + 1]
                market_window = market_returns[i - self.period + 1 : i + 1]

                # Calculate beta using linear regression
                market_variance = np.var(market_window, ddof=1)

                if market_variance != 0:
                    covariance = np.cov(asset_window, market_window, ddof=1)[0, 1]
                    beta = covariance / market_variance
                else:
                    beta = 0.0

                # Calculate correlation for additional insight
                correlation = np.corrcoef(asset_window, market_window)[0, 1]
                correlation = 0.0 if np.isnan(correlation) else correlation

                # Calculate alpha (Jensen's alpha)
                alpha = np.mean(asset_window) - beta * np.mean(market_window)

                # Interpret beta
                interpretation = "neutral"
                if beta > 1.2:
                    interpretation = "high_beta"  # More volatile than market
                elif beta > 0.8:
                    interpretation = "market_beta"  # Similar to market
                elif beta > 0:
                    interpretation = "low_beta"  # Less volatile than market
                elif beta < 0:
                    interpretation = "negative_beta"  # Inverse relationship

                result = {
                    "beta": beta,
                    "alpha": alpha,
                    "correlation": correlation,
                    "interpretation": interpretation,
                    "volatility_ratio": (
                        np.std(asset_window, ddof=1) / np.std(market_window, ddof=1)
                        if np.std(market_window, ddof=1) != 0
                        else 1.0
                    ),
                }
                results.append(result)

            return results

        except Exception:
            return None


class CorrelationCoefficientIndicator:
    """Correlation Coefficient indicator - implements Pearson correlation between price series"""

    def __init__(self, period=20):
        self.period = period

    def calculate(self, data1, data2=None):
        """Calculate Pearson correlation coefficient between two series"""
        import numpy as np

        try:
            # Convert data to numpy arrays
            if isinstance(data1, (list, tuple)):
                series1 = np.array(data1)
            elif isinstance(data1, np.ndarray):
                series1 = data1.flatten()
            elif hasattr(data1, "values"):  # pandas
                series1 = data1.values.flatten()
            else:
                return None

            # If data2 is not provided, correlate with lagged version of data1
            if data2 is None:
                if len(series1) < 2:
                    return None
                series2 = np.roll(series1, 1)[1:]  # Lagged version
                series1 = series1[1:]
            else:
                if isinstance(data2, (list, tuple)):
                    series2 = np.array(data2)
                elif isinstance(data2, np.ndarray):
                    series2 = data2.flatten()
                elif hasattr(data2, "values"):  # pandas
                    series2 = data2.values.flatten()
                else:
                    return None

            # Ensure same length
            min_len = min(len(series1), len(series2))
            series1 = series1[:min_len]
            series2 = series2[:min_len]

            if min_len < self.period:
                return None

            # Calculate rolling correlation
            results = []
            for i in range(self.period - 1, min_len):
                window1 = series1[i - self.period + 1 : i + 1]
                window2 = series2[i - self.period + 1 : i + 1]

                # Calculate Pearson correlation coefficient
                corr_matrix = np.corrcoef(window1, window2)
                if corr_matrix.shape == (2, 2):
                    correlation = corr_matrix[0, 1]
                    # Handle NaN cases
                    correlation = 0.0 if np.isnan(correlation) else correlation
                else:
                    correlation = 0.0

                results.append(correlation)

            return np.array(results)

        except Exception:
            return None


class CointegrationIndicator:
    """Cointegration indicator - implements cointegration test for pairs trading"""

    def __init__(self, period=100):
        self.period = period

    def calculate(self, series1, series2):
        """Calculate cointegration test for pairs trading"""
        import numpy as np

        try:
            # Convert data to numpy arrays
            if isinstance(series1, (list, tuple)):
                prices1 = np.array(series1)
            elif isinstance(series1, np.ndarray):
                prices1 = series1.flatten()
            elif hasattr(series1, "values"):  # pandas
                prices1 = series1.values.flatten()
            else:
                return None

            if isinstance(series2, (list, tuple)):
                prices2 = np.array(series2)
            elif isinstance(series2, np.ndarray):
                prices2 = series2.flatten()
            elif hasattr(series2, "values"):  # pandas
                prices2 = series2.values.flatten()
            else:
                return None

            # Ensure same length
            min_len = min(len(prices1), len(prices2))
            prices1 = prices1[:min_len]
            prices2 = prices2[:min_len]

            if min_len < self.period:
                return None

            # Calculate rolling cointegration
            results = []
            for i in range(self.period - 1, min_len):
                window1 = prices1[i - self.period + 1 : i + 1]
                window2 = prices2[i - self.period + 1 : i + 1]

                # Step 1: Test for unit roots (simplified ADF test)
                stationarity1 = self._test_stationarity(window1)
                stationarity2 = self._test_stationarity(window2)

                # Step 2: If both series are non-stationary, test for cointegration
                if not stationarity1 and not stationarity2:
                    # Perform regression: series1 = alpha + beta * series2 + error
                    beta, alpha = np.polyfit(window2, window1, 1)

                    # Calculate residuals
                    residuals = window1 - (alpha + beta * window2)

                    # Test residuals for stationarity (simplified)
                    residual_stationarity = self._test_stationarity(residuals)

                    # Calculate cointegration metrics
                    correlation = np.corrcoef(window1, window2)[0, 1]
                    correlation = 0.0 if np.isnan(correlation) else correlation

                    # Calculate half-life of mean reversion
                    half_life = self._calculate_half_life(residuals)

                    # Determine cointegration strength
                    if residual_stationarity and abs(correlation) > 0.7:
                        cointegration_strength = "strong"
                    elif residual_stationarity and abs(correlation) > 0.5:
                        cointegration_strength = "moderate"
                    elif abs(correlation) > 0.3:
                        cointegration_strength = "weak"
                    else:
                        cointegration_strength = "none"

                    result = {
                        "cointegrated": residual_stationarity,
                        "correlation": correlation,
                        "beta": beta,
                        "alpha": alpha,
                        "half_life": half_life,
                        "strength": cointegration_strength,
                        "current_spread": residuals[-1],
                        "spread_zscore": (
                            (residuals[-1] - np.mean(residuals)) / np.std(residuals)
                            if np.std(residuals) != 0
                            else 0
                        ),
                    }
                else:
                    # If series are already stationary, no cointegration test needed
                    result = {
                        "cointegrated": False,
                        "correlation": np.corrcoef(window1, window2)[0, 1],
                        "beta": 0,
                        "alpha": 0,
                        "half_life": None,
                        "strength": "stationary_series",
                        "current_spread": 0,
                        "spread_zscore": 0,
                    }

                results.append(result)

            return results

        except Exception:
            return None

    def _test_stationarity(self, data):
        """Simplified stationarity test (Augmented Dickey-Fuller style)"""
        import numpy as np

        try:
            if len(data) < 10:
                return True  # Assume stationary for short series

            # Calculate first differences
            diff_data = np.diff(data)

            # Simple test: if variance of differences is much smaller than variance of levels
            var_levels = np.var(data)
            var_diff = np.var(diff_data)

            # Also check if mean is stable (split series and compare means)
            mid = len(data) // 2
            mean1 = np.mean(data[:mid])
            mean2 = np.mean(data[mid:])
            mean_diff = abs(mean1 - mean2) / np.std(data)

            # Heuristic: series is stationary if differences have lower variance
            # and mean is relatively stable
            is_stationary = (var_diff < var_levels * 0.8) and (mean_diff < 1.0)

            return is_stationary
        except Exception:
            return True  # Default to stationary

    def _calculate_half_life(self, residuals):
        """Calculate half-life of mean reversion"""
        import numpy as np

        try:
            if len(residuals) < 10:
                return None

            # Fit AR(1) model: residual[t] = phi * residual[t-1] + error
            y = residuals[1:]
            x = residuals[:-1]

            if len(x) > 0 and np.var(x) != 0:
                phi = np.cov(x, y)[0, 1] / np.var(x)

                # Half-life = ln(0.5) / ln(phi)
                if 0 < phi < 1:
                    half_life = np.log(0.5) / np.log(phi)
                    return max(0, half_life)  # Ensure positive
                else:
                    return None  # No mean reversion
            else:
                return None
        except Exception:
            return None


class LinearRegressionIndicator:
    """Linear Regression indicator - implements linear regression with R-squared and trend analysis"""

    def __init__(self, period=20):
        self.period = period

    def calculate(self, data):
        """Calculate linear regression with trend analysis"""
        import numpy as np

        try:
            # Convert data to numpy array
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif hasattr(data, "values"):  # pandas
                prices = data.values.flatten()
            else:
                return None

            if len(prices) < self.period:
                return None

            # Calculate rolling linear regression
            results = []
            for i in range(self.period - 1, len(prices)):
                window = prices[i - self.period + 1 : i + 1]
                x = np.arange(len(window))

                # Linear regression: y = mx + b
                coeffs = np.polyfit(x, window, 1)
                slope, intercept = coeffs[0], coeffs[1]

                # Calculate R-squared
                y_pred = slope * x + intercept
                ss_res = np.sum((window - y_pred) ** 2)
                ss_tot = np.sum((window - np.mean(window)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Current regression value (end of window)
                regression_value = slope * (len(window) - 1) + intercept

                # Return comprehensive regression info
                result = {
                    "value": regression_value,
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_squared,
                    "trend": (
                        "bullish"
                        if slope > 0
                        else "bearish" if slope < 0 else "neutral"
                    ),
                }
                results.append(result)

            return results

        except Exception:
            return None


class RSquaredIndicator:
    """R-Squared indicator - implements R-squared for regression quality assessment"""

    def __init__(self, period=20):
        self.period = period

    def calculate(self, data, reference_data=None):
        """Calculate R-squared for regression quality"""
        import numpy as np

        try:
            # Convert data to numpy array
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif hasattr(data, "values"):  # pandas
                prices = data.values.flatten()
            else:
                return None

            if len(prices) < self.period:
                return None

            # If no reference data provided, use time trend as reference
            if reference_data is None:
                # Calculate rolling R-squared against time trend
                results = []
                for i in range(self.period - 1, len(prices)):
                    window = prices[i - self.period + 1 : i + 1]
                    x = np.arange(len(window))

                    # Linear regression against time
                    coeffs = np.polyfit(x, window, 1)
                    y_pred = coeffs[0] * x + coeffs[1]

                    # Calculate R-squared
                    ss_res = np.sum((window - y_pred) ** 2)
                    ss_tot = np.sum((window - np.mean(window)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                    # Additional metrics
                    slope = coeffs[0]
                    correlation = np.corrcoef(x, window)[0, 1]
                    correlation = 0.0 if np.isnan(correlation) else correlation

                    # Interpret trend strength
                    trend_strength = "weak"
                    if r_squared > 0.7:
                        trend_strength = "very_strong"
                    elif r_squared > 0.5:
                        trend_strength = "strong"
                    elif r_squared > 0.3:
                        trend_strength = "moderate"

                    result = {
                        "r_squared": r_squared,
                        "correlation": correlation,
                        "slope": slope,
                        "trend_direction": (
                            "upward"
                            if slope > 0
                            else "downward" if slope < 0 else "flat"
                        ),
                        "trend_strength": trend_strength,
                        "explained_variance": r_squared * 100,  # Percentage
                    }
                    results.append(result)
            else:
                # Calculate R-squared against reference data
                if isinstance(reference_data, (list, tuple)):
                    ref_prices = np.array(reference_data)
                elif isinstance(reference_data, np.ndarray):
                    ref_prices = reference_data.flatten()
                elif hasattr(reference_data, "values"):
                    ref_prices = reference_data.values.flatten()
                else:
                    return None

                # Ensure same length
                min_len = min(len(prices), len(ref_prices))
                prices = prices[:min_len]
                ref_prices = ref_prices[:min_len]

                results = []
                for i in range(self.period - 1, min_len):
                    window = prices[i - self.period + 1 : i + 1]
                    ref_window = ref_prices[i - self.period + 1 : i + 1]

                    # Linear regression
                    if np.var(ref_window) != 0:
                        coeffs = np.polyfit(ref_window, window, 1)
                        y_pred = coeffs[0] * ref_window + coeffs[1]

                        # Calculate R-squared
                        ss_res = np.sum((window - y_pred) ** 2)
                        ss_tot = np.sum((window - np.mean(window)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                        # Additional metrics
                        beta = coeffs[0]
                        alpha = coeffs[1]
                        correlation = np.corrcoef(ref_window, window)[0, 1]
                        correlation = 0.0 if np.isnan(correlation) else correlation
                    else:
                        r_squared = 0
                        beta = 0
                        alpha = np.mean(window)
                        correlation = 0

                    # Interpret relationship strength
                    relationship_strength = "none"
                    if r_squared > 0.8:
                        relationship_strength = "very_strong"
                    elif r_squared > 0.6:
                        relationship_strength = "strong"
                    elif r_squared > 0.4:
                        relationship_strength = "moderate"
                    elif r_squared > 0.2:
                        relationship_strength = "weak"

                    result = {
                        "r_squared": r_squared,
                        "correlation": correlation,
                        "beta": beta,
                        "alpha": alpha,
                        "relationship_strength": relationship_strength,
                        "explained_variance": r_squared * 100,
                    }
                    results.append(result)

            return results

        except Exception:
            return None


class SkewnessIndicator:
    """Skewness indicator - implements distribution skewness analysis"""

    def __init__(self, period=20):
        self.period = period

    def calculate(self, data):
        """Calculate distribution skewness analysis"""
        import numpy as np

        try:
            # Convert data to numpy array
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif hasattr(data, "values"):  # pandas
                prices = data.values.flatten()
            else:
                return None

            if len(prices) < self.period:
                return None

            # Calculate rolling skewness
            results = []
            for i in range(self.period - 1, len(prices)):
                window = prices[i - self.period + 1 : i + 1]

                # Calculate returns for better skewness analysis
                if len(window) > 1:
                    returns = np.diff(window) / window[:-1]
                else:
                    returns = window

                # Calculate skewness using the third moment
                mean = np.mean(returns)
                std = np.std(returns, ddof=1)

                if std != 0 and len(returns) > 2:
                    # Skewness = E[(X - μ)³] / σ³
                    skewness = np.mean(((returns - mean) / std) ** 3)

                    # Sample skewness adjustment for small samples
                    n = len(returns)
                    if n > 2:
                        adjusted_skewness = skewness * np.sqrt(n * (n - 1)) / (n - 2)
                    else:
                        adjusted_skewness = skewness
                else:
                    skewness = 0.0
                    adjusted_skewness = 0.0

                # Interpret skewness
                interpretation = "symmetric"
                if abs(adjusted_skewness) < 0.5:
                    interpretation = "approximately_symmetric"
                elif adjusted_skewness > 0.5:
                    interpretation = "right_skewed"  # Positive skew, tail on right
                elif adjusted_skewness < -0.5:
                    interpretation = "left_skewed"  # Negative skew, tail on left

                # Trading implications
                trading_signal = "neutral"
                if adjusted_skewness > 1.0:
                    trading_signal = "bullish_extremes"  # More extreme positive moves
                elif adjusted_skewness < -1.0:
                    trading_signal = "bearish_extremes"  # More extreme negative moves

                result = {
                    "skewness": skewness,
                    "adjusted_skewness": adjusted_skewness,
                    "interpretation": interpretation,
                    "trading_signal": trading_signal,
                    "distribution_type": self._classify_distribution(adjusted_skewness),
                }
                results.append(result)

            return results

        except Exception:
            return None

    def _classify_distribution(self, skewness):
        """Classify distribution type based on skewness"""
        if abs(skewness) < 0.2:
            return "normal"
        elif 0.2 <= abs(skewness) < 0.5:
            return "slightly_skewed"
        elif 0.5 <= abs(skewness) < 1.0:
            return "moderately_skewed"
        else:
            return "highly_skewed"


class StandardDeviationIndicator:
    """Standard Deviation indicator - calculates rolling standard deviation with proper window handling"""

    def __init__(self, period=20):
        self.period = period

    def calculate(self, data):
        """Calculate rolling standard deviation"""
        import numpy as np

        try:
            # Convert data to numpy array
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif hasattr(data, "values"):  # pandas
                prices = data.values.flatten()
            else:
                return None

            if len(prices) < self.period:
                return None

            # Calculate rolling standard deviation
            results = []
            for i in range(self.period - 1, len(prices)):
                window = prices[i - self.period + 1 : i + 1]
                std_dev = np.std(window, ddof=1)  # Sample standard deviation
                results.append(std_dev)

            return np.array(results)

        except Exception:
            return None


class VarianceRatioIndicator:
    """Variance Ratio indicator - implements variance ratio test for market efficiency"""

    def __init__(self, period=20, k_values=None):
        self.period = period
        self.k_values = k_values or [2, 4, 8, 16]  # Different holding periods

    def calculate(self, data):
        """Calculate variance ratio test for market efficiency"""
        import numpy as np

        try:
            # Convert data to numpy array
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif hasattr(data, "values"):  # pandas
                prices = data.values.flatten()
            else:
                return None

            if len(prices) < self.period * max(self.k_values):
                return None

            # Calculate log returns
            log_returns = np.diff(np.log(prices))

            # Calculate rolling variance ratios
            results = []
            max_k = max(self.k_values)

            for i in range(self.period * max_k - 1, len(log_returns)):
                window = log_returns[i - self.period * max_k + 1 : i + 1]

                variance_ratios = {}
                z_statistics = {}

                for k in self.k_values:
                    if len(window) >= k * self.period:
                        # Calculate variance of k-period returns
                        k_period_returns = []
                        for j in range(0, len(window) - k + 1, k):
                            k_return = np.sum(window[j : j + k])
                            k_period_returns.append(k_return)

                        if len(k_period_returns) > 1:
                            var_k = np.var(k_period_returns, ddof=1)
                            var_1 = np.var(window, ddof=1)

                            # Variance ratio = Var(k-period) / (k * Var(1-period))
                            if var_1 != 0:
                                variance_ratio = var_k / (k * var_1)
                                variance_ratios[k] = variance_ratio

                                # Calculate z-statistic for significance testing
                                n = len(k_period_returns)
                                if n > 1:
                                    # Asymptotic variance of variance ratio
                                    theta = 2 * (2 * k - 1) * (k - 1) / (3 * k)
                                    z_stat = (
                                        np.sqrt(n)
                                        * (variance_ratio - 1)
                                        / np.sqrt(theta)
                                    )
                                    z_statistics[k] = z_stat
                                else:
                                    z_statistics[k] = 0.0
                            else:
                                variance_ratios[k] = 1.0
                                z_statistics[k] = 0.0
                        else:
                            variance_ratios[k] = 1.0
                            z_statistics[k] = 0.0
                    else:
                        variance_ratios[k] = 1.0
                        z_statistics[k] = 0.0

                # Interpret results
                interpretation = self._interpret_variance_ratios(
                    variance_ratios, z_statistics
                )

                result = {
                    "variance_ratios": variance_ratios,
                    "z_statistics": z_statistics,
                    "interpretation": interpretation,
                    "efficiency_score": self._calculate_efficiency_score(
                        variance_ratios
                    ),
                }
                results.append(result)

            return results

        except Exception:
            return None

    def _interpret_variance_ratios(self, variance_ratios, z_statistics):
        """Interpret variance ratio results"""
        # Count significant deviations from 1.0
        significant_deviations = 0
        total_ratios = len(variance_ratios)

        for k, ratio in variance_ratios.items():
            z_stat = z_statistics.get(k, 0)
            if abs(z_stat) > 1.96:  # 95% confidence
                significant_deviations += 1

        if significant_deviations == 0:
            return "efficient_market"
        elif significant_deviations / total_ratios < 0.5:
            return "mostly_efficient"
        else:
            return "inefficient_market"

    def _calculate_efficiency_score(self, variance_ratios):
        """Calculate market efficiency score (0-1, higher = more efficient)"""
        import numpy as np

        try:
            # Calculate how close ratios are to 1.0 (perfect efficiency)
            deviations = [abs(ratio - 1.0) for ratio in variance_ratios.values()]
            avg_deviation = np.mean(deviations) if deviations else 0

            # Convert to efficiency score (higher = more efficient)
            efficiency_score = 1.0 / (1.0 + avg_deviation)
            return efficiency_score
        except Exception:
            return 0.5  # Neutral score


class ZScoreIndicator:
    """Z-Score indicator - implements z-score normalization for statistical analysis"""

    def __init__(self, period=20):
        self.period = period

    def calculate(self, data):
        """Calculate rolling z-score normalization"""
        import numpy as np

        try:
            # Convert data to numpy array
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif hasattr(data, "values"):  # pandas
                prices = data.values.flatten()
            else:
                return None

            if len(prices) < self.period:
                return None

            # Calculate rolling z-score
            results = []
            for i in range(self.period - 1, len(prices)):
                window = prices[i - self.period + 1 : i + 1]
                current_price = prices[i]

                # Calculate mean and standard deviation of window
                mean = np.mean(window)
                std = np.std(window, ddof=1)

                # Calculate z-score
                if std != 0:
                    z_score = (current_price - mean) / std
                else:
                    z_score = 0.0
                # Interpretation
                interpretation = "normal"
                if abs(z_score) > 2.0:
                    interpretation = "extreme_outlier"
                elif abs(z_score) > 1.5:
                    interpretation = "moderate_outlier"
                elif abs(z_score) > 1.0:
                    interpretation = "mild_outlier"

                result = {
                    "z_score": z_score,
                    "mean": mean,
                    "std": std,
                    "interpretation": interpretation,
                    "percentile": self._z_score_to_percentile(z_score),
                }
                results.append(result)

            return results

        except Exception:
            return None

    def _z_score_to_percentile(self, z_score):
        """Convert z-score to approximate percentile"""
        import math

        try:
            # Approximation using cumulative normal distribution
            # Using error function approximation
            percentile = 50 + 35 * math.erf(z_score / math.sqrt(2))
            return max(0, min(100, percentile))
        except Exception:
            return 50  # Default to median


class ChaosFractalDimension:
    """Chaos Fractal Dimension indicator - Real Implementation"""

    def __init__(self, period=20):
        self.period = period

    def calculate(self, data):
        """Calculate Chaos Fractal Dimension using box-counting method"""
        import numpy as np

        try:
            # Convert data to numpy array
            if isinstance(data, (list, tuple)):
                prices = np.array(data)
            elif isinstance(data, np.ndarray):
                prices = data.flatten()
            elif hasattr(data, "values"):  # pandas
                prices = data.values.flatten()
            else:
                return None

            if len(prices) < self.period:
                return None

            # Use recent window
            recent_prices = prices[-self.period :]

            # Calculate fractal dimension using box-counting
            # Normalize prices to range [0, 1]
            min_price = np.min(recent_prices)
            max_price = np.max(recent_prices)
            if max_price == min_price:
                return 1.0  # No variation = dimension 1

            normalized = (recent_prices - min_price) / (max_price - min_price)

            # Create different box sizes
            box_sizes = np.logspace(-3, 0, 20)  # From 0.001 to 1.0
            box_counts = []

            for box_size in box_sizes:
                # Count boxes needed to cover the price path
                if box_size == 0:
                    continue

                # Discretize the path using box_size
                discretized = np.floor(normalized / box_size).astype(int)

                # Count unique boxes occupied
                unique_boxes = len(np.unique(discretized))
                box_counts.append(unique_boxes)

            # Calculate fractal dimension as slope of log-log plot
            valid_indices = np.array(box_counts) > 0
            if np.sum(valid_indices) < 2:
                return 1.5  # Default reasonable value

            log_sizes = np.log(box_sizes[valid_indices])
            log_counts = np.log(np.array(box_counts)[valid_indices])

            # Linear regression to find slope
            if len(log_sizes) >= 2:
                slope = np.polyfit(log_sizes, log_counts, 1)[0]
                fractal_dimension = (
                    -slope
                )  # Negative because we want positive dimension

                # Clamp to reasonable range [1, 2] for price data
                fractal_dimension = max(1.0, min(2.0, fractal_dimension))
                return fractal_dimension
            else:
                return 1.5

        except Exception:
            return None
