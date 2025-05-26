const axios = require('axios');
const cron = require('node-cron');

class MarketDataProvider {
  constructor(pool) {
    this.pool = pool;
    this.isConnected = false;
    this.lastUpdate = null;
    this.subscribers = new Set();
    
    // Major forex pairs
    this.pairs = [
      'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD',
      'EURGBP', 'EURJPY', 'GBPJPY', 'AUDJPY', 'EURAUD', 'GBPAUD', 'EURCHF'
    ];
  }

  async initialize() {
    try {
      console.log('📊 Market Data Provider: Initializing...');
      
      // Initialize pairs in database
      await this.initializePairs();
      
      // Start periodic data fetching
      this.startDataFetching();
      
      this.isConnected = true;
      console.log('✅ Market Data Provider: Initialized successfully');
    } catch (error) {
      console.error('❌ Market Data Provider: Failed to initialize:', error.message);
      this.isConnected = false;
    }
  }

  async initializePairs() {
    const client = await this.pool.connect();
    
    try {
      for (const symbol of this.pairs) {
        const baseAsset = symbol.substring(0, 3);
        const quoteAsset = symbol.substring(3, 6);
        
        const query = `
          INSERT INTO pairs (symbol, name, base_asset, quote_asset, active)
          VALUES ($1, $2, $3, $4, true)
          ON CONFLICT (symbol) DO UPDATE SET
            name = EXCLUDED.name,
            active = EXCLUDED.active
        `;
        
        await client.query(query, [
          symbol,
          `${baseAsset}/${quoteAsset}`,
          baseAsset,
          quoteAsset
        ]);
      }
      
      console.log(`📊 Initialized ${this.pairs.length} currency pairs`);
    } catch (error) {
      console.error('Error initializing pairs:', error.message);
    } finally {
      client.release();
    }
  }

  async fetchAndStorePrices() {
    if (!this.isConnected) return;

    try {
      const prices = await this.generateMockPrices();
      await this.storePrices(prices);
      
      // Notify WebSocket subscribers
      this.notifySubscribers(prices);
      
      this.lastUpdate = new Date();
    } catch (error) {
      console.error('Error fetching and storing prices:', error.message);
    }
  }
  async generateMockPrices() {
    // Enhanced realistic forex prices with real market patterns
    const baseRates = {
      'EURUSD': 1.0950,
      'GBPUSD': 1.2750,
      'USDJPY': 149.50,
      'USDCHF': 0.8850,
      'AUDUSD': 0.6650,
      'USDCAD': 1.3550,
      'NZDUSD': 0.6100,
      'EURGBP': 0.8600,
      'EURJPY': 163.80,
      'GBPJPY': 190.50,
      'AUDJPY': 99.40,
      'EURAUD': 1.6475,
      'GBPAUD': 1.9180,
      'EURCHF': 0.9695
    };

    const prices = [];
    const now = new Date();
    
    // Check if markets are open (simplified - would need proper market hours)
    const isMarketOpen = this.isMarketOpen(now);
    const volatilityMultiplier = isMarketOpen ? 1.0 : 0.3; // Lower volatility when markets closed
    
    for (const [symbol, basePrice] of Object.entries(baseRates)) {
      // Enhanced price simulation with trends and momentum
      const previousPrice = this.lastPrices?.[symbol] || basePrice;
      
      // Add trend momentum (70% chance to continue direction)
      const momentum = this.priceMomentum?.[symbol] || 0;
      const shouldContinueTrend = Math.random() > 0.3;
      
      // Base variation with volatility adjustment
      let variation = (Math.random() - 0.5) * 0.002 * volatilityMultiplier;
      
      // Apply momentum if continuing trend
      if (shouldContinueTrend && Math.abs(momentum) > 0.0001) {
        variation += momentum * 0.5;
      }
      
      // Calculate new price with proper bounds checking
      let currentPrice = previousPrice * (1 + variation);
      
      // Ensure price doesn't deviate too far from base
      const maxDeviation = basePrice * 0.02; // 2% max deviation
      currentPrice = Math.max(basePrice - maxDeviation, 
                             Math.min(basePrice + maxDeviation, currentPrice));
      
      // Calculate realistic bid/ask spread based on volatility
      const baseSpread = this.getSpreadForPair(symbol, currentPrice);
      const volatilitySpread = Math.abs(variation) * currentPrice * 2;
      const totalSpread = baseSpread + volatilitySpread;
      
      const bid = currentPrice - totalSpread / 2;
      const ask = currentPrice + totalSpread / 2;

      const priceData = {
        symbol,
        bid: bid.toFixed(symbol.includes('JPY') ? 3 : 5),
        ask: ask.toFixed(symbol.includes('JPY') ? 3 : 5),
        change: ((currentPrice - basePrice) / basePrice * 100).toFixed(4),
        changePercent: ((currentPrice - basePrice) / basePrice * 100).toFixed(2),
        volume: Math.floor(Math.random() * 10000000) + 1000000, // Simulated volume
        timestamp: now
      };

      prices.push(priceData);

      // Store for momentum calculation
      this.lastPrices = this.lastPrices || {};
      this.priceMomentum = this.priceMomentum || {};
      
      this.lastPrices[symbol] = currentPrice;
      this.priceMomentum[symbol] = variation;
    }

    return prices;
  }

  getSpreadForPair(symbol, price) {
    // Realistic spreads for major pairs (in price units)
    const spreads = {
      'EURUSD': 0.00015,  // 1.5 pips
      'GBPUSD': 0.00020,  // 2.0 pips
      'USDJPY': 0.015,    // 1.5 pips (JPY pairs)
      'USDCHF': 0.00020,  // 2.0 pips
      'AUDUSD': 0.00025,  // 2.5 pips
      'USDCAD': 0.00025,  // 2.5 pips
      'NZDUSD': 0.00030,  // 3.0 pips
      'EURGBP': 0.00025,  // 2.5 pips
      'EURJPY': 0.020,    // 2.0 pips
      'GBPJPY': 0.030,    // 3.0 pips
      'AUDJPY': 0.025,    // 2.5 pips
      'EURAUD': 0.00035,  // 3.5 pips
      'GBPAUD': 0.00040,  // 4.0 pips
      'EURCHF': 0.00030   // 3.0 pips
    };
    
    return spreads[symbol] || 0.00030; // Default 3 pips
  }

  isMarketOpen(date) {
    // Simplified market hours check (would need proper timezone handling)
    const hour = date.getUTCHours();
    const day = date.getUTCDay();
    
    // Weekend check
    if (day === 0 || (day === 6 && hour > 21) || (day === 1 && hour < 21)) {
      return false;
    }
    
    // Major session hours (simplified)
    return true; // For demo, always consider market open
  }

  async storePrices(prices) {
    const client = await this.pool.connect();
    
    try {
      await client.query('BEGIN');

      for (const price of prices) {
        // Store current price
        const insertQuery = `
          INSERT INTO market_prices (symbol, bid, ask, timestamp)
          VALUES ($1, $2, $3, $4)
        `;
        
        await client.query(insertQuery, [
          price.symbol,
          price.bid,
          price.ask,
          price.timestamp
        ]);

        // Update latest price cache
        const updateQuery = `
          INSERT INTO current_prices (symbol, current_price, bid, ask, last_updated)
          VALUES ($1, $2, $3, $4, $5)
          ON CONFLICT (symbol) DO UPDATE SET
            current_price = EXCLUDED.current_price,
            bid = EXCLUDED.bid,
            ask = EXCLUDED.ask,
            last_updated = EXCLUDED.last_updated
        `;
        
        const currentPrice = ((parseFloat(price.bid) + parseFloat(price.ask)) / 2).toFixed(5);
        
        await client.query(updateQuery, [
          price.symbol,
          currentPrice,
          price.bid,
          price.ask,
          price.timestamp
        ]);
      }

      await client.query('COMMIT');
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async getCurrentPrices() {
    try {
      const query = `
        SELECT p.symbol, p.name, cp.current_price, cp.bid, cp.ask, cp.last_updated
        FROM pairs p
        LEFT JOIN current_prices cp ON p.symbol = cp.symbol
        WHERE p.active = true
        ORDER BY p.symbol
      `;
      
      const result = await this.pool.query(query);
      return result.rows;
    } catch (error) {
      throw new Error(`Failed to get current prices: ${error.message}`);
    }
  }

  async getHistoricalData(symbol, timeframe = '1h', limit = 100) {
    try {
      let interval;
      switch (timeframe) {
        case '1m': interval = '1 minute'; break;
        case '5m': interval = '5 minutes'; break;
        case '15m': interval = '15 minutes'; break;
        case '1h': interval = '1 hour'; break;
        case '4h': interval = '4 hours'; break;
        case '1d': interval = '1 day'; break;
        default: interval = '1 hour';
      }

      const query = `
        SELECT 
          date_trunc($2, timestamp) as time_bucket,
          first(bid ORDER BY timestamp) as open,
          max((bid + ask) / 2) as high,
          min((bid + ask) / 2) as low,
          last(ask ORDER BY timestamp) as close,
          avg((bid + ask) / 2) as average
        FROM market_prices
        WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '30 days'
        GROUP BY time_bucket
        ORDER BY time_bucket DESC
        LIMIT $3
      `;

      const result = await this.pool.query(query, [symbol, interval, limit]);
      return result.rows;
    } catch (error) {
      throw new Error(`Failed to get historical data: ${error.message}`);
    }
  }

  startDataFetching() {
    // Fetch prices every 10 seconds
    cron.schedule('*/10 * * * * *', async () => {
      await this.fetchAndStorePrices();
    });

    console.log('📊 Market data fetching scheduled every 10 seconds');
  }

  addSubscriber(callback) {
    this.subscribers.add(callback);
  }

  removeSubscriber(callback) {
    this.subscribers.delete(callback);
  }

  notifySubscribers(data) {
    this.subscribers.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error notifying subscriber:', error.message);
      }
    });
  }

  async checkConnection() {
    return this.isConnected;
  }

  getLastUpdate() {
    return this.lastUpdate;
  }

  async getTechnicalIndicators(symbol, timeframe = '1h', period = 14) {
    try {
      const historicalData = await this.getHistoricalData(symbol, timeframe, 100);
      
      if (historicalData.length < period) {
        throw new Error(`Insufficient data for calculation. Need at least ${period} periods.`);
      }

      const prices = historicalData.map(d => parseFloat(d.close || d.current_price));
      const highs = historicalData.map(d => parseFloat(d.high || d.current_price));
      const lows = historicalData.map(d => parseFloat(d.low || d.current_price));
      const volumes = historicalData.map(d => parseFloat(d.volume || 1000000));

      const indicators = {
        symbol,
        timeframe,
        timestamp: new Date().toISOString(),
        sma: this.calculateSMA(prices, period),
        ema: this.calculateEMA(prices, period),
        rsi: this.calculateRSI(prices, period),
        macd: this.calculateMACD(prices),
        bollinger: this.calculateBollingerBands(prices, period),
        stochastic: this.calculateStochastic(highs, lows, prices, period),
        atr: this.calculateATR(highs, lows, prices, period),
        volume_sma: this.calculateSMA(volumes, period)
      };

      return indicators;
    } catch (error) {
      throw new Error(`Failed to calculate technical indicators: ${error.message}`);
    }
  }

  calculateSMA(prices, period) {
    if (prices.length < period) return null;
    const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
    return (sum / period).toFixed(5);
  }

  calculateEMA(prices, period) {
    if (prices.length < period) return null;
    
    const multiplier = 2 / (period + 1);
    let ema = this.calculateSMA(prices.slice(0, period), period);
    
    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
    }
    
    return parseFloat(ema).toFixed(5);
  }

  calculateRSI(prices, period) {
    if (prices.length < period + 1) return null;
    
    const gains = [];
    const losses = [];
    
    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    const avgGain = gains.slice(-period).reduce((a, b) => a + b, 0) / period;
    const avgLoss = losses.slice(-period).reduce((a, b) => a + b, 0) / period;
    
    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));
    
    return rsi.toFixed(2);
  }

  calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    if (prices.length < slowPeriod) return null;
    
    const fastEMA = parseFloat(this.calculateEMA(prices, fastPeriod));
    const slowEMA = parseFloat(this.calculateEMA(prices, slowPeriod));
    const macdLine = fastEMA - slowEMA;
    
    // Calculate signal line (EMA of MACD line)
    const macdHistory = [macdLine]; // In real implementation, use historical MACD values
    const signalLine = parseFloat(this.calculateEMA(macdHistory, signalPeriod)) || macdLine;
    const histogram = macdLine - signalLine;
    
    return {
      macd: macdLine.toFixed(5),
      signal: signalLine.toFixed(5),
      histogram: histogram.toFixed(5)
    };
  }

  calculateBollingerBands(prices, period, stdDev = 2) {
    if (prices.length < period) return null;
    
    const sma = parseFloat(this.calculateSMA(prices, period));
    const recentPrices = prices.slice(-period);
    
    // Calculate standard deviation
    const variance = recentPrices.reduce((sum, price) => {
      return sum + Math.pow(price - sma, 2);
    }, 0) / period;
    
    const standardDeviation = Math.sqrt(variance);
    
    return {
      upper: (sma + (standardDeviation * stdDev)).toFixed(5),
      middle: sma.toFixed(5),
      lower: (sma - (standardDeviation * stdDev)).toFixed(5)
    };
  }

  calculateStochastic(highs, lows, closes, period) {
    if (highs.length < period) return null;
    
    const recentHighs = highs.slice(-period);
    const recentLows = lows.slice(-period);
    const currentClose = closes[closes.length - 1];
    
    const highest = Math.max(...recentHighs);
    const lowest = Math.min(...recentLows);
    
    const k = ((currentClose - lowest) / (highest - lowest)) * 100;
    
    return {
      k: k.toFixed(2),
      d: k.toFixed(2) // Simplified - would need smoothing in real implementation
    };
  }

  calculateATR(highs, lows, closes, period) {
    if (highs.length < period + 1) return null;
    
    const trueRanges = [];
    
    for (let i = 1; i < highs.length; i++) {
      const high = highs[i];
      const low = lows[i];
      const prevClose = closes[i - 1];
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      
      trueRanges.push(tr);
    }
    
    const atr = trueRanges.slice(-period).reduce((a, b) => a + b, 0) / period;
    return atr.toFixed(5);
  }

  getSessionStatus(session, date) {
    const hour = date.getUTCHours();
    const sessions = {
      sydney: { start: 21, end: 6 }, // 21:00 - 06:00 UTC
      tokyo: { start: 23, end: 8 },  // 23:00 - 08:00 UTC
      london: { start: 7, end: 16 }, // 07:00 - 16:00 UTC
      newYork: { start: 12, end: 21 } // 12:00 - 21:00 UTC
    };

    const sessionTimes = sessions[session];
    if (!sessionTimes) return { active: false, nextOpen: null };

    let isActive;
    if (sessionTimes.start > sessionTimes.end) {
      // Session crosses midnight
      isActive = hour >= sessionTimes.start || hour < sessionTimes.end;
    } else {
      isActive = hour >= sessionTimes.start && hour < sessionTimes.end;
    }

    return {
      active: isActive,
      localTime: this.convertToSessionTime(date, session),
      nextOpen: isActive ? null : this.getNextSessionOpen(session, date),
      nextClose: isActive ? this.getNextSessionClose(session, date) : null
    };
  }

  convertToSessionTime(utcDate, session) {
    // Simplified timezone conversion
    const timezoneOffsets = {
      sydney: 11,   // UTC+11 (simplified)
      tokyo: 9,     // UTC+9
      london: 0,    // UTC+0 (GMT)
      newYork: -5   // UTC-5 (EST, simplified)
    };

    const offset = timezoneOffsets[session] || 0;
    const localTime = new Date(utcDate.getTime() + offset * 60 * 60 * 1000);
    return localTime.toTimeString().slice(0, 8);
  }

  getNextSessionOpen(session, date) {
    // Simplified calculation - would need proper timezone handling
    const sessions = {
      sydney: 21,
      tokyo: 23,
      london: 7,
      newYork: 12
    };

    const sessionStart = sessions[session];
    const nextOpen = new Date(date);
    nextOpen.setUTCHours(sessionStart, 0, 0, 0);
    
    if (nextOpen <= date) {
      nextOpen.setUTCDate(nextOpen.getUTCDate() + 1);
    }
    
    return nextOpen.toISOString();
  }

  getNextSessionClose(session, date) {
    const sessions = {
      sydney: 6,
      tokyo: 8,
      london: 16,
      newYork: 21
    };

    const sessionEnd = sessions[session];
    const nextClose = new Date(date);
    nextClose.setUTCHours(sessionEnd, 0, 0, 0);
    
    if (nextClose <= date) {
      nextClose.setUTCDate(nextClose.getUTCDate() + 1);
    }
    
    return nextClose.toISOString();
  }
}

module.exports = MarketDataProvider;
