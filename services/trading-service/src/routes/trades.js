const express = require('express');
const Trade = require('../models/Trade');
const { authenticateToken, validateTrade } = require('../middleware/auth');

const router = express.Router();

// Initialize Trade model with database pool
let tradeModel;

const initializeTradeRoutes = (pool) => {
  tradeModel = new Trade(pool);
};

// Create a new trade
router.post('/', authenticateToken, validateTrade, async (req, res) => {
  try {
    const tradeData = {
      user_id: req.user.id,
      pair_id: req.body.pair_id,
      type: req.body.type,
      amount: parseFloat(req.body.amount),
      price: parseFloat(req.body.price),
      strategy_id: req.body.strategy_id || null,
      notes: req.body.notes || null
    };

    const trade = await tradeModel.create(tradeData);
    
    res.status(201).json({
      success: true,
      data: trade,
      message: 'Trade created successfully'
    });
  } catch (error) {
    console.error('Create trade error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get user's trades with pagination
router.get('/', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const limit = parseInt(req.query.limit) || 50;
    const offset = parseInt(req.query.offset) || 0;

    const trades = await tradeModel.getByUserId(userId, limit, offset);
    
    res.json({
      success: true,
      data: trades,
      pagination: {
        limit,
        offset,
        total: trades.length
      }
    });
  } catch (error) {
    console.error('Get trades error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get specific trade by ID
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const tradeId = req.params.id;
    const userId = req.user.id;

    const trade = await tradeModel.getById(tradeId, userId);
    
    if (!trade) {
      return res.status(404).json({
        success: false,
        message: 'Trade not found'
      });
    }
    
    res.json({
      success: true,
      data: trade
    });
  } catch (error) {
    console.error('Get trade error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Update trade status (execute, cancel)
router.patch('/:id/status', authenticateToken, async (req, res) => {
  try {
    const tradeId = req.params.id;
    const { status, executed_price } = req.body;

    // Validate status
    if (!['executed', 'cancelled'].includes(status)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid status. Must be "executed" or "cancelled"'
      });
    }

    // Verify trade ownership
    const existingTrade = await tradeModel.getById(tradeId, req.user.id);
    if (!existingTrade) {
      return res.status(404).json({
        success: false,
        message: 'Trade not found'
      });
    }

    if (existingTrade.status !== 'pending') {
      return res.status(400).json({
        success: false,
        message: 'Only pending trades can be updated'
      });
    }

    const updatedTrade = await tradeModel.updateStatus(
      tradeId, 
      status, 
      executed_price || null
    );
    
    res.json({
      success: true,
      data: updatedTrade,
      message: `Trade ${status} successfully`
    });
  } catch (error) {
    console.error('Update trade status error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get pending trades
router.get('/status/pending', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const pendingTrades = await tradeModel.getPendingTrades(userId);
    
    res.json({
      success: true,
      data: pendingTrades
    });
  } catch (error) {
    console.error('Get pending trades error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get trading statistics
router.get('/stats/summary', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const days = parseInt(req.query.days) || 30;
    
    const stats = await tradeModel.getTradingStats(userId, days);
    
    res.json({
      success: true,
      data: {
        ...stats,
        period_days: days
      }
    });
  } catch (error) {
    console.error('Get trading stats error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

module.exports = { router, initializeTradeRoutes };
