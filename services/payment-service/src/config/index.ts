import dotenv from 'dotenv';
import { PaymentConfig } from '@/types';

dotenv.config();

export const config: PaymentConfig = {
  stripe: {
    secretKey: process.env.STRIPE_SECRET_KEY || 'sk_test_...',
    webhookSecret: process.env.STRIPE_WEBHOOK_SECRET || 'whsec_...',
    publicKey: process.env.STRIPE_PUBLIC_KEY || 'pk_test_...'
  },
  paypal: {
    clientId: process.env.PAYPAL_CLIENT_ID || '',
    clientSecret: process.env.PAYPAL_CLIENT_SECRET || '',
    mode: (process.env.PAYPAL_MODE as 'sandbox' | 'live') || 'sandbox'
  },
  limits: {
    dailyDepositLimit: Number(process.env.DAILY_DEPOSIT_LIMIT) || 10000,
    dailyWithdrawalLimit: Number(process.env.DAILY_WITHDRAWAL_LIMIT) || 5000,
    monthlyDepositLimit: Number(process.env.MONTHLY_DEPOSIT_LIMIT) || 100000,
    monthlyWithdrawalLimit: Number(process.env.MONTHLY_WITHDRAWAL_LIMIT) || 50000,
    minDepositAmount: Number(process.env.MIN_DEPOSIT_AMOUNT) || 10,
    minWithdrawalAmount: Number(process.env.MIN_WITHDRAWAL_AMOUNT) || 10
  },
  fraud: {
    maxDailyTransactions: Number(process.env.MAX_DAILY_TRANSACTIONS) || 20,
    maxTransactionAmount: Number(process.env.MAX_TRANSACTION_AMOUNT) || 50000,
    suspiciousCountries: (process.env.SUSPICIOUS_COUNTRIES || '').split(',').filter(Boolean)
  }
};

export const serverConfig = {
  port: Number(process.env.PORT) || 3006,
  nodeEnv: process.env.NODE_ENV || 'development',
  jwtSecret: process.env.JWT_SECRET || 'your-secret-key',
  
  // Database configuration
  database: {
    host: process.env.DB_HOST || 'localhost',
    port: Number(process.env.DB_PORT) || 5432,
    name: process.env.DB_NAME || 'forex_platform',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'password',
    ssl: process.env.DB_SSL === 'true'
  },
  
  // Redis configuration
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: Number(process.env.REDIS_PORT) || 6379,
    password: process.env.REDIS_PASSWORD || '',
    db: Number(process.env.REDIS_DB) || 0
  },
  
  // External service URLs
  services: {
    userService: process.env.USER_SERVICE_URL || 'http://localhost:3001',
    tradingService: process.env.TRADING_SERVICE_URL || 'http://localhost:3002',
    eventSystem: process.env.EVENT_SYSTEM_URL || 'http://localhost:3005'
  }
};

export default { config, serverConfig };
