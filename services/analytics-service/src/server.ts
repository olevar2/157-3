// Analytics Service - AI-powered market analysis and trading recommendations
// Provides ML model integration, technical analysis, and pattern recognition

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import winston from 'winston';
import dotenv from 'dotenv';
import cron from 'node-cron';

import { TechnicalAnalysisEngine } from './engines/TechnicalAnalysisEngine';
import { MLModelEngine } from './engines/MLModelEngine';
import { PatternRecognitionEngine } from './engines/PatternRecognitionEngine';
import { RiskAnalysisEngine } from './engines/RiskAnalysisEngine';
import { MarketDataCollector } from './services/MarketDataCollector';
import { RecommendationEngine } from './services/RecommendationEngine';
import { AnalyticsCache } from './services/AnalyticsCache';
import { AuthenticationMiddleware } from './middleware/AuthenticationMiddleware';

// Load environment variables
dotenv.config();
dotenv.config({ path: '.env.local' });

// Initialize logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({ 
      filename: 'logs/analytics-service.log',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    })
  ]
});

const app = express();
const PORT = process.env.PORT || 3007;

// Rate limiting
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // 100 requests per window
  standardHeaders: true,
  legacyHeaders: false,
});

// Middleware
app.use(helmet());
app.use(cors());
app.use('/api/v1/', apiLimiter);
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Initialize analytics engines
const technicalAnalysis = new TechnicalAnalysisEngine(logger);
const mlModel = new MLModelEngine(logger);
const patternRecognition = new PatternRecognitionEngine(logger);
const riskAnalysis = new RiskAnalysisEngine(logger);
const marketDataCollector = new MarketDataCollector(logger);
const recommendationEngine = new RecommendationEngine(logger);
const analyticsCache = new AnalyticsCache(logger);
const authMiddleware = new AuthenticationMiddleware(logger);

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'analytics-service',
    timestamp: new Date().toISOString(),
    engines: {
      technicalAnalysis: technicalAnalysis.isReady(),
      mlModel: mlModel.isReady(),
      patternRecognition: patternRecognition.isReady(),
      riskAnalysis: riskAnalysis.isReady()
    },
    uptime: process.uptime()
  });
});

// Service info
app.get('/api/info', (req, res) => {
  res.json({
    service: 'analytics-service',
    version: '1.0.0',
    capabilities: [
      'technical-analysis',
      'ml-predictions',
      'pattern-recognition',
      'risk-assessment',
      'trading-recommendations'
    ],
    supportedSymbols: ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD', 'EURGBP'],
    timestamp: new Date().toISOString()
  });
});

// Technical Analysis Endpoints
app.get('/api/v1/analysis/technical/:symbol', authMiddleware.authenticate, async (req, res) => {
  try {
    const { symbol } = req.params;
    const { timeframe = '1h', period = 100 } = req.query;

    // Check cache first
    const cacheKey = `technical:${symbol}:${timeframe}:${period}`;
    const cached = await analyticsCache.get(cacheKey);
    if (cached) {
      return res.json(cached);
    }

    // Get market data
    const marketData = await marketDataCollector.getHistoricalData(symbol, timeframe as string, parseInt(period as string));
    
    // Perform technical analysis
    const analysis = await technicalAnalysis.analyze(symbol, marketData);
    
    // Cache result for 5 minutes
    await analyticsCache.set(cacheKey, analysis, 300);
    
    res.json(analysis);
  } catch (error) {
    logger.error('Technical analysis error:', error);
    res.status(500).json({ error: 'Technical analysis failed' });
  }
});

// ML Predictions Endpoint
app.get('/api/v1/predictions/:symbol', authMiddleware.authenticate, async (req, res) => {
  try {
    const { symbol } = req.params;
    const { horizon = '1h' } = req.query;

    const cacheKey = `prediction:${symbol}:${horizon}`;
    const cached = await analyticsCache.get(cacheKey);
    if (cached) {
      return res.json(cached);
    }

    // Get market data for ML model
    const marketData = await marketDataCollector.getHistoricalData(symbol, '1h', 500);
    
    // Generate ML predictions
    const predictions = await mlModel.predict(symbol, marketData, horizon as string);
    
    // Cache for 10 minutes
    await analyticsCache.set(cacheKey, predictions, 600);
    
    res.json(predictions);
  } catch (error) {
    logger.error('ML prediction error:', error);
    res.status(500).json({ error: 'Prediction generation failed' });
  }
});

// Pattern Recognition Endpoint
app.get('/api/v1/patterns/:symbol', authMiddleware.authenticate, async (req, res) => {
  try {
    const { symbol } = req.params;
    const { timeframe = '1h' } = req.query;

    const cacheKey = `patterns:${symbol}:${timeframe}`;
    const cached = await analyticsCache.get(cacheKey);
    if (cached) {
      return res.json(cached);
    }

    // Get market data
    const marketData = await marketDataCollector.getHistoricalData(symbol, timeframe as string, 200);
    
    // Detect patterns
    const patterns = await patternRecognition.detectPatterns(symbol, marketData);
    
    // Cache for 15 minutes
    await analyticsCache.set(cacheKey, patterns, 900);
    
    res.json(patterns);
  } catch (error) {
    logger.error('Pattern recognition error:', error);
    res.status(500).json({ error: 'Pattern detection failed' });
  }
});

// Risk Analysis Endpoint
app.post('/api/v1/risk/portfolio', authMiddleware.authenticate, async (req, res) => {
  try {
    const { positions, accountBalance } = req.body;

    if (!positions || !accountBalance) {
      return res.status(400).json({ error: 'Positions and account balance required' });
    }

    // Perform risk analysis
    const riskAnalysisResult = await riskAnalysis.analyzePortfolio(positions, accountBalance);
    
    res.json(riskAnalysisResult);
  } catch (error) {
    logger.error('Risk analysis error:', error);
    res.status(500).json({ error: 'Risk analysis failed' });
  }
});

// Trading Recommendations Endpoint
app.get('/api/v1/recommendations/:symbol', authMiddleware.authenticate, async (req, res) => {
  try {
    const { symbol } = req.params;
    const { riskLevel = 'medium' } = req.query;

    const cacheKey = `recommendations:${symbol}:${riskLevel}`;
    const cached = await analyticsCache.get(cacheKey);
    if (cached) {
      return res.json(cached);
    }

    // Generate comprehensive recommendations
    const recommendations = await recommendationEngine.generateRecommendations(
      symbol, 
      riskLevel as string
    );
    
    // Cache for 5 minutes
    await analyticsCache.set(cacheKey, recommendations, 300);
    
    res.json(recommendations);
  } catch (error) {
    logger.error('Recommendation generation error:', error);
    res.status(500).json({ error: 'Recommendation generation failed' });
  }
});

// Market Sentiment Endpoint
app.get('/api/v1/sentiment/:symbol', authMiddleware.authenticate, async (req, res) => {
  try {
    const { symbol } = req.params;

    const cacheKey = `sentiment:${symbol}`;
    const cached = await analyticsCache.get(cacheKey);
    if (cached) {
      return res.json(cached);
    }

    // Get market data for sentiment analysis
    const marketData = await marketDataCollector.getHistoricalData(symbol, '1h', 100);
    
    // Analyze market sentiment
    const sentiment = await technicalAnalysis.analyzeSentiment(symbol, marketData);
    
    // Cache for 30 minutes
    await analyticsCache.set(cacheKey, sentiment, 1800);
    
    res.json(sentiment);
  } catch (error) {
    logger.error('Sentiment analysis error:', error);
    res.status(500).json({ error: 'Sentiment analysis failed' });
  }
});

// Batch Analysis Endpoint
app.post('/api/v1/analysis/batch', authMiddleware.authenticate, async (req, res) => {
  try {
    const { symbols, analysisTypes } = req.body;

    if (!symbols || !Array.isArray(symbols)) {
      return res.status(400).json({ error: 'Symbols array required' });
    }

    const results = {};
    
    for (const symbol of symbols) {
      results[symbol] = {};
      
      if (analysisTypes.includes('technical')) {
        const marketData = await marketDataCollector.getHistoricalData(symbol, '1h', 100);
        results[symbol].technical = await technicalAnalysis.analyze(symbol, marketData);
      }
      
      if (analysisTypes.includes('patterns')) {
        const marketData = await marketDataCollector.getHistoricalData(symbol, '1h', 200);
        results[symbol].patterns = await patternRecognition.detectPatterns(symbol, marketData);
      }
      
      if (analysisTypes.includes('predictions')) {
        const marketData = await marketDataCollector.getHistoricalData(symbol, '1h', 500);
        results[symbol].predictions = await mlModel.predict(symbol, marketData, '1h');
      }
    }
    
    res.json({
      timestamp: new Date().toISOString(),
      results
    });
  } catch (error) {
    logger.error('Batch analysis error:', error);
    res.status(500).json({ error: 'Batch analysis failed' });
  }
});

// Error handling middleware
app.use((error: any, req: any, res: any, next: any) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Initialize analytics engines
async function initializeEngines() {
  logger.info('Initializing analytics engines...');
  
  try {
    await technicalAnalysis.initialize();
    await mlModel.initialize();
    await patternRecognition.initialize();
    await riskAnalysis.initialize();
    await marketDataCollector.initialize();
    await recommendationEngine.initialize();
    await analyticsCache.initialize();
    
    logger.info('✅ All analytics engines initialized successfully');
  } catch (error) {
    logger.error('❌ Failed to initialize analytics engines:', error);
    process.exit(1);
  }
}

// Schedule periodic tasks
function schedulePeriodicTasks() {
  // Update ML models every hour
  cron.schedule('0 * * * *', async () => {
    logger.info('Running hourly ML model update...');
    try {
      await mlModel.updateModels();
    } catch (error) {
      logger.error('ML model update failed:', error);
    }
  });

  // Clear old cache entries every 30 minutes
  cron.schedule('*/30 * * * *', async () => {
    logger.info('Running cache cleanup...');
    try {
      await analyticsCache.cleanup();
    } catch (error) {
      logger.error('Cache cleanup failed:', error);
    }
  });

  logger.info('✅ Periodic tasks scheduled');
}

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  process.exit(0);
});

// Start server
async function startServer() {
  await initializeEngines();
  schedulePeriodicTasks();
  
  app.listen(PORT, () => {
    logger.info(`🚀 Analytics Service running on port ${PORT}`);
    logger.info(`🧠 AI-powered market analysis ready`);
    logger.info(`📊 Technical analysis engine active`);
    logger.info(`🔮 ML prediction models loaded`);
    logger.info(`🔍 Pattern recognition enabled`);
    logger.info(`⚖️ Risk analysis engine ready`);
  });
}

startServer().catch(error => {
  logger.error('Failed to start Analytics Service:', error);
  process.exit(1);
});

export default app;
