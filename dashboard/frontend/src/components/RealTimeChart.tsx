import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  FormControlLabel,
  Switch,
  Grid,
  Chip,
  IconButton,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  Timeline,
  TrendingUp,
  TrendingDown,
  ShowChart,
  Settings,
  Fullscreen,
  Refresh,
} from '@mui/icons-material';
import { createChart, IChartApi, ISeriesApi, LineStyle, CrosshairMode } from 'lightweight-charts';
import { useWebSocket } from '../contexts/WebSocketContext';

interface PriceData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface IndicatorData {
  time: number;
  value: number;
}

interface TechnicalIndicators {
  rsi: IndicatorData[];
  macd: IndicatorData[];
  sma20: IndicatorData[];
  sma50: IndicatorData[];
  ema12: IndicatorData[];
  ema26: IndicatorData[];
  bollinger_upper: IndicatorData[];
  bollinger_lower: IndicatorData[];
  bollinger_middle: IndicatorData[];
}

interface RealTimeChartProps {
  symbol?: string;
  height?: number;
  showControls?: boolean;
}

const RealTimeChart: React.FC<RealTimeChartProps> = ({
  symbol = 'EUR/USD',
  height = 600,
  showControls = true,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const indicatorSeriesRef = useRef<{ [key: string]: ISeriesApi<'Line'> }>({});

  const [timeframe, setTimeframe] = useState('M15');
  const [activeIndicators, setActiveIndicators] = useState<string[]>(['sma20', 'sma50']);
  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [indicators, setIndicators] = useState<TechnicalIndicators>({
    rsi: [],
    macd: [],
    sma20: [],
    sma50: [],
    ema12: [],
    ema26: [],
    bollinger_upper: [],
    bollinger_lower: [],
    bollinger_middle: [],
  });
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [isLoading, setIsLoading] = useState(true);

  const { sendMessage, lastMessage } = useWebSocket();

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height - 100, // Account for controls
      layout: {
        background: { color: '#1a1b23' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
        horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: 'rgba(197, 203, 206, 0.4)',
      },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.4)',
        timeVisible: true,
        secondsVisible: timeframe === 'M1',
      },
    });

    chartRef.current = chart;

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#4caf50',
      downColor: '#f44336',
      borderDownColor: '#f44336',
      borderUpColor: '#4caf50',
      wickDownColor: '#f44336',
      wickUpColor: '#4caf50',
    });

    candlestickSeriesRef.current = candlestickSeries;

    // Create volume series
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    });

    volumeSeriesRef.current = volumeSeries;

    // Set volume scale
    chart.priceScale('volume').applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, [height, timeframe]);

  // Generate mock data for demonstration
  const generateMockData = useCallback(() => {
    const data: PriceData[] = [];
    const indicatorData: TechnicalIndicators = {
      rsi: [],
      macd: [],
      sma20: [],
      sma50: [],
      ema12: [],
      ema26: [],
      bollinger_upper: [],
      bollinger_lower: [],
      bollinger_middle: [],
    };

    let basePrice = 1.0780; // EUR/USD base price
    const now = Date.now();
    const timeframeMs = getTimeframeMs(timeframe);
    const dataPoints = 100;

    for (let i = dataPoints; i >= 0; i--) {
      const time = Math.floor((now - i * timeframeMs) / 1000);

      // Generate realistic price movement
      const volatility = 0.0005;
      const trend = Math.sin(i * 0.1) * 0.0002;
      const noise = (Math.random() - 0.5) * volatility;

      basePrice += trend + noise;

      const open = basePrice;
      const high = open + Math.random() * 0.0003;
      const low = open - Math.random() * 0.0003;
      const close = low + Math.random() * (high - low);
      const volume = Math.random() * 1000 + 500;

      data.push({ time, open, high, low, close, volume });

      // Generate indicator data
      if (data.length >= 20) {
        const sma20 = data.slice(-20).reduce((sum, d) => sum + d.close, 0) / 20;
        indicatorData.sma20.push({ time, value: sma20 });
      }

      if (data.length >= 50) {
        const sma50 = data.slice(-50).reduce((sum, d) => sum + d.close, 0) / 50;
        indicatorData.sma50.push({ time, value: sma50 });
      }

      // Simple RSI calculation
      if (data.length >= 14) {
        const changes = data.slice(-14).map((d, idx) =>
          idx > 0 ? d.close - data[data.length - 14 + idx - 1].close : 0
        ).slice(1);
        const gains = changes.filter(c => c > 0).reduce((sum, c) => sum + c, 0) / 14;
        const losses = Math.abs(changes.filter(c => c < 0).reduce((sum, c) => sum + c, 0)) / 14;
        const rs = gains / (losses || 0.0001);
        const rsi = 100 - (100 / (1 + rs));
        indicatorData.rsi.push({ time, value: rsi });
      }

      // EMA calculations
      if (data.length >= 12) {
        const ema12 = calculateEMA(data.slice(-12).map(d => d.close), 12);
        indicatorData.ema12.push({ time, value: ema12 });
      }

      if (data.length >= 26) {
        const ema26 = calculateEMA(data.slice(-26).map(d => d.close), 26);
        indicatorData.ema26.push({ time, value: ema26 });
      }
    }

    setPriceData(data);
    setIndicators(indicatorData);
    setCurrentPrice(data[data.length - 1]?.close || 0);
    setPriceChange(data.length > 1 ? data[data.length - 1].close - data[data.length - 2].close : 0);
    setIsLoading(false);
  }, [timeframe]);

  const calculateEMA = (prices: number[], period: number): number => {
    const multiplier = 2 / (period + 1);
    let ema = prices[0];
    for (let i = 1; i < prices.length; i++) {
      ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
    }
    return ema;
  };

  const getTimeframeMs = (tf: string): number => {
    switch (tf) {
      case 'M1': return 60 * 1000;
      case 'M5': return 5 * 60 * 1000;
      case 'M15': return 15 * 60 * 1000;
      case 'H1': return 60 * 60 * 1000;
      case 'H4': return 4 * 60 * 60 * 1000;
      default: return 15 * 60 * 1000;
    }
  };

  // Load initial data
  useEffect(() => {
    generateMockData();
  }, [generateMockData]);

  // Update chart data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !volumeSeriesRef.current) return;

    candlestickSeriesRef.current.setData(priceData);

    const volumeData = priceData.map(d => ({
      time: d.time,
      value: d.volume || 0,
      color: d.close >= d.open ? '#4caf50' : '#f44336',
    }));

    volumeSeriesRef.current.setData(volumeData);
  }, [priceData]);

  // Handle indicator toggles
  const handleIndicatorToggle = (indicator: string) => {
    setActiveIndicators(prev => {
      const newIndicators = prev.includes(indicator)
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator];

      // Remove indicator series if deactivated
      if (prev.includes(indicator) && indicatorSeriesRef.current[indicator]) {
        chartRef.current?.removeSeries(indicatorSeriesRef.current[indicator]);
        delete indicatorSeriesRef.current[indicator];
      }

      return newIndicators;
    });
  };

  // Add indicator series
  useEffect(() => {
    if (!chartRef.current) return;

    activeIndicators.forEach(indicator => {
      if (!indicatorSeriesRef.current[indicator] && indicators[indicator as keyof TechnicalIndicators].length > 0) {
        const series = chartRef.current!.addLineSeries({
          color: getIndicatorColor(indicator),
          lineWidth: 2,
          lineStyle: getIndicatorLineStyle(indicator),
        });

        series.setData(indicators[indicator as keyof TechnicalIndicators]);
        indicatorSeriesRef.current[indicator] = series;
      }
    });
  }, [activeIndicators, indicators]);

  const getIndicatorColor = (indicator: string): string => {
    const colors: { [key: string]: string } = {
      sma20: '#2196f3',
      sma50: '#ff9800',
      ema12: '#9c27b0',
      ema26: '#e91e63',
      rsi: '#00bcd4',
      macd: '#4caf50',
      bollinger_upper: '#f44336',
      bollinger_lower: '#f44336',
      bollinger_middle: '#ffeb3b',
    };
    return colors[indicator] || '#ffffff';
  };

  const getIndicatorLineStyle = (indicator: string): LineStyle => {
    if (indicator.includes('bollinger')) return LineStyle.Dashed;
    return LineStyle.Solid;
  };

  const handleTimeframeChange = (newTimeframe: string) => {
    setTimeframe(newTimeframe);
    setIsLoading(true);
  };

  const handleRefresh = () => {
    setIsLoading(true);
    generateMockData();
  };

  return (
    <Paper elevation={3} sx={{ p: 2, height }}>
      {/* Chart Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ShowChart />
            {symbol}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="h6" color={priceChange >= 0 ? 'success.main' : 'error.main'}>
              {currentPrice.toFixed(5)}
            </Typography>
            <Chip
              icon={priceChange >= 0 ? <TrendingUp /> : <TrendingDown />}
              label={`${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(5)}`}
              color={priceChange >= 0 ? 'success' : 'error'}
              size="small"
            />
          </Box>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} size="small">
              <Refresh />
            </IconButton>
          </Tooltip>
          <Tooltip title="Chart Settings">
            <IconButton size="small">
              <Settings />
            </IconButton>
          </Tooltip>
          <Tooltip title="Fullscreen">
            <IconButton size="small">
              <Fullscreen />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {showControls && (
        <Grid container spacing={2} sx={{ mb: 2 }}>
          {/* Timeframe Selection */}
          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Timeframe</InputLabel>
              <Select
                value={timeframe}
                label="Timeframe"
                onChange={(e) => handleTimeframeChange(e.target.value)}
              >
                <MenuItem value="M1">1 Minute</MenuItem>
                <MenuItem value="M5">5 Minutes</MenuItem>
                <MenuItem value="M15">15 Minutes</MenuItem>
                <MenuItem value="H1">1 Hour</MenuItem>
                <MenuItem value="H4">4 Hours</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {/* Indicator Toggles */}
          <Grid item xs={12} md={9}>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {Object.keys(indicators).map((indicator) => (
                <FormControlLabel
                  key={indicator}
                  control={
                    <Switch
                      checked={activeIndicators.includes(indicator)}
                      onChange={() => handleIndicatorToggle(indicator)}
                      size="small"
                    />
                  }
                  label={indicator.toUpperCase()}
                  sx={{ mr: 2 }}
                />
              ))}
            </Box>
          </Grid>
        </Grid>
      )}

      {/* Chart Container */}
      <Box
        ref={chartContainerRef}
        sx={{
          width: '100%',
          height: height - 150,
          position: 'relative',
          '& canvas': {
            borderRadius: 1,
          },
        }}
      />

      {isLoading && (
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 1000,
          }}
        >
          <Typography>Loading chart data...</Typography>
        </Box>
      )}
    </Paper>
  );
};

export default RealTimeChart;
