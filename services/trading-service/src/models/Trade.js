from shared.logging.platform3_logger import Platform3Logger
from shared.error_handling.platform3_error_system import Platform3ErrorSystem, ServiceError
from shared.database.platform3_database_manager import Platform3DatabaseManager
from shared.communication.platform3_communication_framework import Platform3CommunicationFramework
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
const { Pool } = require('pg');

class Trade {
  constructor(pool) {
    this.pool = pool;
  }

  async create(tradeData) {
    const {
      user_id,
      pair_id,
      type, // 'buy' or 'sell'
      amount,
      price,
      strategy_id = null,
      notes = null
    } = tradeData;

    const query = `
      INSERT INTO trades (user_id, pair_id, type, amount, price, strategy_id, notes, status, created_at)
      VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending', NOW())
      RETURNING *
    `;

    try {
      const result = await this.pool.query(query, [
        user_id, pair_id, type, amount, price, strategy_id, notes
      ]);
      return result.rows[0];
    } catch (error) {
      throw new Error(`Failed to create trade: ${error.message}`);
    }
  }

  async getByUserId(userId, limit = 50, offset = 0) {
    const query = `
      SELECT t.*, p.symbol as pair_symbol, p.name as pair_name
      FROM trades t
      JOIN pairs p ON t.pair_id = p.id
      WHERE t.user_id = $1
      ORDER BY t.created_at DESC
      LIMIT $2 OFFSET $3
    `;

    try {
      const result = await this.pool.query(query, [userId, limit, offset]);
      return result.rows;
    } catch (error) {
      throw new Error(`Failed to fetch user trades: ${error.message}`);
    }
  }

  async getById(tradeId, userId) {
    const query = `
      SELECT t.*, p.symbol as pair_symbol, p.name as pair_name
      FROM trades t
      JOIN pairs p ON t.pair_id = p.id
      WHERE t.id = $1 AND t.user_id = $2
    `;

    try {
      const result = await this.pool.query(query, [tradeId, userId]);
      return result.rows[0] || null;
    } catch (error) {
      throw new Error(`Failed to fetch trade: ${error.message}`);
    }
  }

  async updateStatus(tradeId, status, executedPrice = null) {
    const query = `
      UPDATE trades 
      SET status = $2, executed_price = $3, executed_at = CASE WHEN $2 = 'executed' THEN NOW() ELSE executed_at END
      WHERE id = $1
      RETURNING *
    `;

    try {
      const result = await this.pool.query(query, [tradeId, status, executedPrice]);
      return result.rows[0];
    } catch (error) {
      throw new Error(`Failed to update trade status: ${error.message}`);
    }
  }

  async getPendingTrades(userId) {
    const query = `
      SELECT t.*, p.symbol as pair_symbol
      FROM trades t
      JOIN pairs p ON t.pair_id = p.id
      WHERE t.user_id = $1 AND t.status = 'pending'
      ORDER BY t.created_at ASC
    `;

    try {
      const result = await this.pool.query(query, [userId]);
      return result.rows;
    } catch (error) {
      throw new Error(`Failed to fetch pending trades: ${error.message}`);
    }
  }

  async getTradingStats(userId, days = 30) {
    const query = `
      SELECT 
        COUNT(*) as total_trades,
        COUNT(CASE WHEN status = 'executed' THEN 1 END) as executed_trades,
        COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_trades,
        SUM(CASE WHEN status = 'executed' THEN amount ELSE 0 END) as total_volume,
        AVG(CASE WHEN status = 'executed' THEN executed_price ELSE NULL END) as avg_execution_price
      FROM trades
      WHERE user_id = $1 AND created_at >= NOW() - INTERVAL '$2 days'
    `;

    try {
      const result = await this.pool.query(query, [userId, days]);
      return result.rows[0];
    } catch (error) {
      throw new Error(`Failed to fetch trading stats: ${error.message}`);
    }
  }
}

module.exports = Trade;
