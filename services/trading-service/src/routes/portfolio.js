const express = require('express');
const Portfolio = require('../models/Portfolio');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();

// Initialize Portfolio model with database pool
let portfolioModel;

const initializePortfolioRoutes = (pool) => {
  portfolioModel = new Portfolio(pool);
};

// Get portfolio balance
router.get('/balance', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const balance = await portfolioModel.getBalance(userId);
    
    res.json({
      success: true,
      data: balance
    });
  } catch (error) {
    console.error('Get portfolio balance error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get portfolio value and positions
router.get('/value', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const portfolioValue = await portfolioModel.getPortfolioValue(userId);
    
    res.json({
      success: true,
      data: portfolioValue
    });
  } catch (error) {
    console.error('Get portfolio value error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Update portfolio balance (internal use - typically called by trade execution)
router.post('/balance/update', authenticateToken, async (req, res) => {
  try {
    const { asset_symbol, amount, operation } = req.body;
    const userId = req.user.id;

    if (!asset_symbol || amount === undefined || !operation) {
      return res.status(400).json({
        success: false,
        message: 'asset_symbol, amount, and operation are required'
      });
    }

    if (!['add', 'subtract'].includes(operation)) {
      return res.status(400).json({
        success: false,
        message: 'operation must be "add" or "subtract"'
      });
    }

    const updatedBalance = await portfolioModel.updateBalance(
      userId,
      asset_symbol,
      parseFloat(amount),
      operation
    );
    
    res.json({
      success: true,
      data: updatedBalance,
      message: 'Portfolio balance updated successfully'
    });
  } catch (error) {
    console.error('Update portfolio balance error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get portfolio performance metrics
router.get('/performance', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    const days = parseInt(req.query.days) || 30;
    
    const performance = await portfolioModel.getPerformanceMetrics(userId, days);
    
    res.json({
      success: true,
      data: {
        metrics: performance,
        period_days: days
      }
    });
  } catch (error) {
    console.error('Get portfolio performance error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// Get portfolio summary (combines balance and value)
router.get('/summary', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.id;
    
    const [balance, portfolioValue] = await Promise.all([
      portfolioModel.getBalance(userId),
      portfolioModel.getPortfolioValue(userId)
    ]);
    
    res.json({
      success: true,
      data: {
        balance,
        portfolio_value: portfolioValue.total_value,
        positions: portfolioValue.positions,
        last_updated: portfolioValue.calculated_at
      }
    });
  } catch (error) {
    console.error('Get portfolio summary error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

module.exports = { router, initializePortfolioRoutes };
