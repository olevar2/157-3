# Trading Service

A comprehensive trading service that handles trade execution, portfolio management, and trading statistics for the Platform3 trading platform.

## Features

- **Trade Management**: Create, execute, and cancel trades
- **Portfolio Tracking**: Real-time balance and position tracking
- **Trading Statistics**: Performance metrics and analytics
- **Security**: JWT authentication and rate limiting
- **Database Integration**: PostgreSQL with connection pooling

## API Endpoints

### Trades
- `POST /api/trades` - Create a new trade
- `GET /api/trades` - Get user's trades (with pagination)
- `GET /api/trades/:id` - Get specific trade details
- `PATCH /api/trades/:id/status` - Update trade status (execute/cancel)
- `GET /api/trades/status/pending` - Get pending trades
- `GET /api/trades/stats/summary` - Get trading statistics

### Portfolio
- `GET /api/portfolio/balance` - Get portfolio balances
- `GET /api/portfolio/value` - Get portfolio value and positions
- `POST /api/portfolio/balance/update` - Update portfolio balance
- `GET /api/portfolio/performance` - Get performance metrics
- `GET /api/portfolio/summary` - Get complete portfolio summary

### System
- `GET /health` - Health check endpoint
- `GET /api/info` - Service information

## Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the service:**
   ```bash
   # Development
   npm run dev
   
   # Production
   npm start
   ```

## Database Schema Requirements

The service requires the following database tables:

### trades
```sql
CREATE TABLE trades (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  pair_id INTEGER NOT NULL,
  type VARCHAR(10) NOT NULL CHECK (type IN ('buy', 'sell')),
  amount DECIMAL(20,8) NOT NULL,
  price DECIMAL(20,8) NOT NULL,
  executed_price DECIMAL(20,8),
  status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'executed', 'cancelled')),
  strategy_id INTEGER,
  notes TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  executed_at TIMESTAMP
);
```

### portfolio_balances
```sql
CREATE TABLE portfolio_balances (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  asset_symbol VARCHAR(10) NOT NULL,
  balance DECIMAL(20,8) NOT NULL DEFAULT 0,
  updated_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(user_id, asset_symbol)
);
```

### pairs
```sql
CREATE TABLE pairs (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(20) UNIQUE NOT NULL,
  name VARCHAR(100) NOT NULL,
  base_asset VARCHAR(10) NOT NULL,
  quote_asset VARCHAR(10) NOT NULL,
  active BOOLEAN DEFAULT true
);
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `3003` |
| `NODE_ENV` | Environment | `development` |
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `platform3` |
| `DB_USER` | Database user | `postgres` |
| `DB_PASSWORD` | Database password | `password` |
| `JWT_SECRET` | JWT secret key | Required |
| `ALLOWED_ORIGINS` | CORS origins | `http://localhost:3000` |

## Usage Examples

### Create a Trade
```javascript
POST /api/trades
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "pair_id": 1,
  "type": "buy",
  "amount": 0.5,
  "price": 50000,
  "notes": "Market entry position"
}
```

### Get Portfolio Summary
```javascript
GET /api/portfolio/summary
Authorization: Bearer <jwt_token>

Response:
{
  "success": true,
  "data": {
    "balance": [
      {
        "asset_symbol": "BTC",
        "balance": "0.12345678",
        "updated_at": "2024-01-15T10:30:00.000Z"
      }
    ],
    "portfolio_value": 61728.90,
    "positions": [...],
    "last_updated": "2024-01-15T10:30:00.000Z"
  }
}
```

## Development

The service is built with:
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **PostgreSQL** - Database
- **JWT** - Authentication
- **Helmet** - Security middleware

## Security Features

- **JWT Authentication**: All endpoints require valid JWT tokens
- **Rate Limiting**: 100 requests per 15 minutes per IP
- **Input Validation**: Comprehensive validation for all inputs
- **SQL Injection Protection**: Parameterized queries
- **CORS Protection**: Configurable origin restrictions
- **Helmet.js**: Security headers

## Error Handling

The service provides comprehensive error handling with:
- Detailed error messages in development
- Generic error messages in production
- Request ID tracking for debugging
- Proper HTTP status codes

## Monitoring

- Health check endpoint for monitoring
- Request logging with timestamps
- Database connection monitoring
- Graceful shutdown handling
