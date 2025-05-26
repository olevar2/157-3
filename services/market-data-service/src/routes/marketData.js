const express = require('express');
const router = express.Router();
const MarketDataProvider = require('../services/MarketDataProvider');

/**
 * @route GET /api/market-data/prices
 * @desc Get current prices for all currency pairs
 * @access Public
 */
router.get('/prices', async (req, res) => {
  try {
    const prices = await MarketDataProvider.getCurrentPrices();
    res.json({
      success: true,
      data: prices,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error fetching current prices:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch current prices'
    });
  }
});

/**
 * @route GET /api/market-data/prices/:symbol
 * @desc Get current price for a specific currency pair
 * @access Public
 */
router.get('/prices/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const price = await MarketDataProvider.getCurrentPrice(symbol.toUpperCase());
    
    if (!price) {
      return res.status(404).json({
        success: false,
        error: 'Currency pair not found'
      });
    }

    res.json({
      success: true,
      data: price,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error fetching price for symbol:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch price data'
    });
  }
});

/**
 * @route GET /api/market-data/history/:symbol
 * @desc Get historical price data for a currency pair
 * @access Public
 */
router.get('/history/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { limit = 100, timeframe = '1m' } = req.query;
    
    const history = await MarketDataProvider.getHistoricalData(
      symbol.toUpperCase(),
      parseInt(limit),
      timeframe
    );

    res.json({
      success: true,
      data: history,
      symbol: symbol.toUpperCase(),
      timeframe,
      count: history.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error fetching historical data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch historical data'
    });
  }
});

/**
 * @route GET /api/market-data/stats/:symbol
 * @desc Get trading statistics for a currency pair
 * @access Public
 */
router.get('/stats/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const stats = await MarketDataProvider.getTradingStats(symbol.toUpperCase());
    
    if (!stats) {
      return res.status(404).json({
        success: false,
        error: 'Currency pair not found'
      });
    }

    res.json({
      success: true,
      data: stats,
      symbol: symbol.toUpperCase(),
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error fetching trading stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch trading statistics'
    });
  }
});

/**
 * @route GET /api/market-data/instruments
 * @desc Get list of available trading instruments
 * @access Public
 */
router.get('/instruments', (req, res) => {
  try {
    const instruments = MarketDataProvider.getAvailableInstruments();
    
    res.json({
      success: true,
      data: instruments,
      count: instruments.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error fetching instruments:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch instruments'
    });
  }
});

/**
 * @route GET /api/market-data/indicators/:symbol
 * @desc Get technical indicators for a currency pair
 * @access Public
 */
router.get('/indicators/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { timeframe = '1h', period = 14 } = req.query;
    
    const indicators = await MarketDataProvider.getTechnicalIndicators(
      symbol.toUpperCase(),
      timeframe,
      parseInt(period)
    );

    res.json({
      success: true,
      data: indicators,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error calculating technical indicators:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to calculate technical indicators',
      details: error.message
    });
  }
});

/**
 * @route GET /api/market-data/market-status
 * @desc Get overall market status and session information
 * @access Public
 */
router.get('/market-status', async (req, res) => {
  try {
    const now = new Date();
    const marketStatus = {
      timestamp: now.toISOString(),
      utcTime: now.toUTCString(),
      sessions: {
        sydney: MarketDataProvider.getSessionStatus('sydney', now),
        tokyo: MarketDataProvider.getSessionStatus('tokyo', now),
        london: MarketDataProvider.getSessionStatus('london', now),
        newYork: MarketDataProvider.getSessionStatus('newYork', now)
      },
      isWeekend: now.getUTCDay() === 0 || now.getUTCDay() === 6,
      serverTime: now.toISOString(),
      marketVolatility: 'normal' // Would calculate from recent price movements
    };

    res.json({
      success: true,
      data: marketStatus,
      timestamp: now.toISOString()
    });
  } catch (error) {
    console.error('Error getting market status:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get market status'
    });
  }
});

/**
 * @route GET /api/market-data/watchlist
 * @desc Get watchlist with enhanced price information
 * @access Public
 */
router.get('/watchlist', async (req, res) => {
  try {
    const { symbols } = req.query;
    let targetSymbols = symbols ? symbols.split(',').map(s => s.toUpperCase()) : null;
    
    const prices = await MarketDataProvider.getCurrentPrices();
    let watchlist = prices;
    
    if (targetSymbols) {
      watchlist = prices.filter(price => targetSymbols.includes(price.symbol));
    }
    
    // Enhance with additional data
    const enhancedWatchlist = await Promise.all(
      watchlist.map(async (price) => {
        try {
          const indicators = await MarketDataProvider.getTechnicalIndicators(price.symbol, '1h', 14);
          return {
            ...price,
            rsi: indicators?.rsi || null,
            trend: indicators?.sma && parseFloat(price.current_price) > parseFloat(indicators.sma) ? 'bullish' : 'bearish',
            volatility: indicators?.atr || null
          };
        } catch (error) {
          return {
            ...price,
            rsi: null,
            trend: 'neutral',
            volatility: null
          };
        }
      })
    );

    res.json({
      success: true,
      data: enhancedWatchlist,
      count: enhancedWatchlist.length,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting watchlist:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get watchlist data'
    });
  }
});

module.exports = router;
