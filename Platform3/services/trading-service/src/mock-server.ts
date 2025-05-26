import express, { Request, Response } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import winston from 'winston';
import { Decimal } from 'decimal.js';
import dotenv from 'dotenv';

dotenv.config();

// Logger setup
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'logs/trading-service-mock.log' })
  ]
});

const app = express();
const PORT = process.env.PORT || 3003;

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// In-memory storage for demo
const positions = new Map();
const orders = new Map();
const accounts = new Map();

// Mock trading account
const demoAccount = {
  id: 'demo-account-1',
  userId: 'demo-user-1',
  balance: new Decimal(10000),
  equity: new Decimal(10000),
  margin: new Decimal(0),
  freeMargin: new Decimal(10000),
  marginLevel: 0,
  currency: 'USD',
  leverage: 100,
  createdAt: new Date(),
  updatedAt: new Date()
};

accounts.set('demo-user-1', demoAccount);

// Mock market data
const mockMarketData = {
  'EURUSD': { bid: 1.0850, ask: 1.0852, spread: 0.0002 },
  'GBPUSD': { bid: 1.2650, ask: 1.2653, spread: 0.0003 },
  'USDJPY': { bid: 149.80, ask: 149.85, spread: 0.05 },
  'AUDUSD': { bid: 0.6580, ask: 0.6583, spread: 0.0003 },
  'USDCAD': { bid: 1.3750, ask: 1.3753, spread: 0.0003 }
};

// Routes
app.post('/api/v1/orders', async (req: Request, res: Response): Promise<void> => {
  try {
    const { symbol, type, side, volume, price, stopLoss, takeProfit } = req.body;
    
    const orderId = 'order-' + Date.now();
    const order = {
      id: orderId,
      userId: 'demo-user-1',
      symbol,
      type,
      side,
      volume: new Decimal(volume),
      price: price ? new Decimal(price) : null,
      stopLoss: stopLoss ? new Decimal(stopLoss) : null,
      takeProfit: takeProfit ? new Decimal(takeProfit) : null,
      status: type === 'market' ? 'filled' : 'pending',
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    orders.set(orderId, order);
    
    // If market order, create position immediately
    if (type === 'market') {
      const positionId = 'position-' + Date.now();
      const marketPrice = side === 'buy' ? mockMarketData[symbol as keyof typeof mockMarketData]?.ask : mockMarketData[symbol as keyof typeof mockMarketData]?.bid;
      
      const position = {
        id: positionId,
        userId: 'demo-user-1',
        symbol,
        side,
        volume: new Decimal(volume),
        openPrice: new Decimal(marketPrice || 1.0000),
        currentPrice: new Decimal(marketPrice || 1.0000),
        stopLoss: order.stopLoss,
        takeProfit: order.takeProfit,
        pnl: new Decimal(0),
        commission: new Decimal(2),
        swap: new Decimal(0),
        status: 'open',
        openTime: new Date(),
        updatedAt: new Date()
      };
      
      positions.set(positionId, position);
      
      res.json({
        success: true,
        order: {
          id: order.id,
          status: order.status,
          fillPrice: marketPrice
        },
        position: {
          id: position.id,
          symbol: position.symbol,
          side: position.side,
          volume: position.volume.toString(),
          openPrice: position.openPrice.toString(),
          pnl: position.pnl.toString()
        }
      });
    } else {
      res.json({
        success: true,
        order: {
          id: order.id,
          status: order.status
        }
      });
    }
  } catch (error) {
    logger.error('Order creation error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/positions', async (req: Request, res: Response): Promise<void> => {
  try {
    const userPositions = Array.from(positions.values())
      .filter(p => p.userId === 'demo-user-1' && p.status === 'open')
      .map(p => ({
        id: p.id,
        symbol: p.symbol,
        side: p.side,
        volume: p.volume.toString(),
        openPrice: p.openPrice.toString(),
        currentPrice: p.currentPrice.toString(),
        pnl: p.pnl.toString(),
        commission: p.commission.toString(),
        swap: p.swap.toString(),
        openTime: p.openTime,
        stopLoss: p.stopLoss?.toString(),
        takeProfit: p.takeProfit?.toString()
      }));
    
    res.json({
      success: true,
      positions: userPositions
    });
  } catch (error) {
    logger.error('Positions fetch error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/orders', async (req: Request, res: Response): Promise<void> => {
  try {
    const userOrders = Array.from(orders.values())
      .filter(o => o.userId === 'demo-user-1')
      .map(o => ({
        id: o.id,
        symbol: o.symbol,
        type: o.type,
        side: o.side,
        volume: o.volume.toString(),
        price: o.price?.toString(),
        status: o.status,
        createdAt: o.createdAt,
        stopLoss: o.stopLoss?.toString(),
        takeProfit: o.takeProfit?.toString()
      }));
    
    res.json({
      success: true,
      orders: userOrders
    });
  } catch (error) {
    logger.error('Orders fetch error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/account', async (req: Request, res: Response) => {
  try {
    const account = accounts.get('demo-user-1');
    
    res.json({
      success: true,
      account: {
        id: account.id,
        userId: account.userId,
        balance: account.balance.toString(),
        equity: account.equity.toString(),
        margin: account.margin.toString(),
        freeMargin: account.freeMargin.toString(),
        marginLevel: account.marginLevel,
        currency: account.currency,
        leverage: account.leverage
      }
    });
  } catch (error) {
    logger.error('Account fetch error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/api/v1/market/:symbol', async (req: Request, res: Response): Promise<void> => {
  try {
    const { symbol } = req.params;
    const marketData = mockMarketData[symbol as keyof typeof mockMarketData];
      if (!marketData) {
      res.status(404).json({ error: 'Market data not available' });
      return;
    }
    
    // Add some random variation
    const variation = (Math.random() - 0.5) * 0.0001;
    const bid = marketData.bid + variation;
    const ask = bid + marketData.spread;
    
    res.json({
      success: true,
      symbol,
      bid: bid.toFixed(5),
      ask: ask.toFixed(5),
      spread: marketData.spread,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Market data error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.delete('/api/v1/positions/:id', async (req: Request, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    const position = positions.get(id);
      if (!position || position.userId !== 'demo-user-1') {
      res.status(404).json({ error: 'Position not found' });
      return;
    }    
    // Calculate final P&L
    const closePrice = position.side === 'buy' 
      ? mockMarketData[position.symbol as keyof typeof mockMarketData]?.bid 
      : mockMarketData[position.symbol as keyof typeof mockMarketData]?.ask;
    
    const priceDiff = position.side === 'buy' 
      ? new Decimal(closePrice || 1.0000).minus(position.openPrice)
      : position.openPrice.minus(new Decimal(closePrice || 1.0000));
    
    const finalPnl = priceDiff.times(position.volume).times(100000); // Assuming standard lot size
    
    position.status = 'closed';
    position.closePrice = new Decimal(closePrice || 1.0000);
    position.pnl = finalPnl;
    position.closeTime = new Date();
    
    res.json({
      success: true,
      message: 'Position closed successfully',
      pnl: finalPnl.toString()
    });
  } catch (error) {
    logger.error('Close position error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.get('/health', (req: Request, res: Response) => {
  res.json({ 
    status: 'ok', 
    service: 'trading-service-mock',
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  logger.info(`Mock Trading Service running on port ${PORT}`);
  logger.info('Ready to handle demo trading operations');
});

export default app;
