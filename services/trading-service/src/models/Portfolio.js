const { Pool } = require('pg');

class Portfolio {
  constructor(pool) {
    this.pool = pool;
  }

  async getBalance(userId) {
    const query = `
      SELECT * FROM portfolio_balances 
      WHERE user_id = $1
      ORDER BY asset_symbol ASC
    `;

    try {
      const result = await this.pool.query(query, [userId]);
      return result.rows;
    } catch (error) {
      throw new Error(`Failed to fetch portfolio balance: ${error.message}`);
    }
  }

  async updateBalance(userId, assetSymbol, amount, operation = 'add') {
    const client = await this.pool.connect();
    
    try {
      await client.query('BEGIN');

      // Check if balance exists
      const checkQuery = `
        SELECT balance FROM portfolio_balances 
        WHERE user_id = $1 AND asset_symbol = $2
      `;
      const checkResult = await client.query(checkQuery, [userId, assetSymbol]);

      let newBalance;
      if (checkResult.rows.length === 0) {
        // Create new balance record
        newBalance = operation === 'add' ? amount : 0;
        const insertQuery = `
          INSERT INTO portfolio_balances (user_id, asset_symbol, balance, updated_at)
          VALUES ($1, $2, $3, NOW())
          RETURNING *
        `;
        const result = await client.query(insertQuery, [userId, assetSymbol, newBalance]);
        await client.query('COMMIT');
        return result.rows[0];
      } else {
        // Update existing balance
        const currentBalance = parseFloat(checkResult.rows[0].balance);
        newBalance = operation === 'add' ? currentBalance + amount : currentBalance - amount;
        
        if (newBalance < 0) {
          throw new Error('Insufficient balance');
        }

        const updateQuery = `
          UPDATE portfolio_balances 
          SET balance = $3, updated_at = NOW()
          WHERE user_id = $1 AND asset_symbol = $2
          RETURNING *
        `;
        const result = await client.query(updateQuery, [userId, assetSymbol, newBalance]);
        await client.query('COMMIT');
        return result.rows[0];
      }
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  async getPortfolioValue(userId) {
    // This would typically fetch current market prices and calculate total value
    // For now, returning basic structure
    const query = `
      SELECT 
        pb.*,
        COALESCE(mp.current_price, 1) as current_price,
        (pb.balance * COALESCE(mp.current_price, 1)) as value
      FROM portfolio_balances pb
      LEFT JOIN market_prices mp ON pb.asset_symbol = mp.symbol
      WHERE pb.user_id = $1 AND pb.balance > 0
    `;

    try {
      const result = await this.pool.query(query, [userId]);
      const positions = result.rows;
      
      const totalValue = positions.reduce((sum, position) => sum + parseFloat(position.value), 0);
      
      return {
        positions,
        total_value: totalValue,
        calculated_at: new Date()
      };
    } catch (error) {
      throw new Error(`Failed to calculate portfolio value: ${error.message}`);
    }
  }

  async getPerformanceMetrics(userId, days = 30) {
    const query = `
      SELECT 
        DATE(created_at) as date,
        SUM(CASE WHEN type = 'buy' AND status = 'executed' THEN amount * executed_price ELSE 0 END) as bought,
        SUM(CASE WHEN type = 'sell' AND status = 'executed' THEN amount * executed_price ELSE 0 END) as sold
      FROM trades
      WHERE user_id = $1 AND created_at >= NOW() - INTERVAL '$2 days'
      GROUP BY DATE(created_at)
      ORDER BY date ASC
    `;

    try {
      const result = await this.pool.query(query, [userId, days]);
      return result.rows;
    } catch (error) {
      throw new Error(`Failed to fetch performance metrics: ${error.message}`);
    }
  }
}

module.exports = Portfolio;
