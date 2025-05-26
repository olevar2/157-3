// Forex Account Management Service Types
export interface ForexAccount {
  id: string;
  userId: string;
  brokerName: string;
  accountNumber: string;
  accountType: 'demo' | 'live' | 'micro' | 'standard' | 'ecn';
  baseCurrency: string;
  leverage: number;
  balance: number;
  equity: number;
  freeMargin: number;
  marginLevel: number;
  isActive: boolean;
  isDefault: boolean;
  lastSync: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface ForexCredentials {
  id: string;
  accountId: string;
  loginId: string;
  password: string; // encrypted
  serverAddress: string;
  serverPort?: number;
  platformType: 'mt4' | 'mt5' | 'ctrader' | 'ninjatrader' | 'api';
  encryptedAt: Date;
  lastUsed?: Date;
}

export interface BrokerConnection {
  id: string;
  accountId: string;
  status: 'connected' | 'disconnected' | 'error' | 'testing';
  lastConnected?: Date;
  lastError?: string;
  connectionAttempts: number;
  responseTime?: number;
}

export interface AccountSyncResult {
  success: boolean;
  accountId: string;
  syncedAt: Date;
  balance?: number;
  equity?: number;
  freeMargin?: number;
  marginLevel?: number;
  openPositions?: number;
  error?: string;
}

export interface ForexBroker {
  id: string;
  name: string;
  displayName: string;
  website: string;
  supportedPlatforms: string[];
  minDeposit: number;
  maxLeverage: number;
  spreads: Record<string, number>;
  isSupported: boolean;
  apiEndpoint?: string;
  documentation?: string;
}

export interface TradingPlatform {
  type: 'mt4' | 'mt5' | 'ctrader' | 'ninjatrader' | 'api';
  name: string;
  version?: string;
  features: string[];
  connectionMethods: ('login' | 'api' | 'bridge')[];
}

export interface AccountValidation {
  accountId: string;
  isValid: boolean;
  canConnect: boolean;
  canTrade: boolean;
  issues: string[];
  recommendations: string[];
  checkedAt: Date;
}

export interface AccountPerformance {
  accountId: string;
  period: '1d' | '1w' | '1m' | '3m' | '6m' | '1y' | 'all';
  totalProfit: number;
  totalLoss: number;
  netProfit: number;
  profitFactor: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  maxDrawdown: number;
  sharpeRatio: number;
  tradesCount: number;
  calculatedAt: Date;
}

export interface AccountSwitch {
  id: string;
  userId: string;
  fromAccountId: string;
  toAccountId: string;
  reason: string;
  switchedAt: Date;
  status: 'success' | 'failed' | 'pending';
  error?: string;
}

export interface EncryptionConfig {
  algorithm: 'aes-256-gcm';
  keyDerivation: 'pbkdf2';
  iterations: number;
  saltLength: number;
  ivLength: number;
  tagLength: number;
}

export interface AccountConfig {
  maxAccountsPerUser: number;
  syncInterval: number; // minutes
  connectionTimeout: number; // seconds
  maxRetryAttempts: number;
  supportedBrokers: string[];
  encryptionSettings: EncryptionConfig;
}

export interface AccountHealthCheck {
  accountId: string;
  isHealthy: boolean;
  connectionStatus: 'good' | 'slow' | 'error';
  balanceStatus: 'normal' | 'low' | 'critical';
  marginStatus: 'safe' | 'warning' | 'danger';
  lastCheck: Date;
  nextCheck: Date;
  alerts: string[];
}

export interface BrokerApiConfig {
  brokerId: string;
  apiType: 'rest' | 'websocket' | 'fix' | 'custom';
  baseUrl: string;
  authMethod: 'basic' | 'oauth' | 'api_key' | 'certificate';
  rateLimits: {
    requestsPerMinute: number;
    requestsPerHour: number;
  };
  endpoints: {
    balance: string;
    positions: string;
    orders: string;
    history: string;
  };
}

export interface AccountActivity {
  id: string;
  accountId: string;
  activityType: 'login' | 'sync' | 'trade' | 'balance_check' | 'connection_test';
  status: 'success' | 'failed' | 'warning';
  details?: Record<string, any>;
  timestamp: Date;
  duration?: number; // milliseconds
}

export interface ForexAccountRequest {
  brokerName: string;
  accountNumber: string;
  loginId: string;
  password: string;
  serverAddress: string;
  serverPort?: number;
  platformType: 'mt4' | 'mt5' | 'ctrader' | 'ninjatrader' | 'api';
  accountType: 'demo' | 'live' | 'micro' | 'standard' | 'ecn';
  leverage?: number;
  isDefault?: boolean;
}

export interface AccountUpdateRequest {
  brokerName?: string;
  loginId?: string;
  password?: string;
  serverAddress?: string;
  serverPort?: number;
  leverage?: number;
  isActive?: boolean;
  isDefault?: boolean;
}

export interface ConnectionTestRequest {
  accountId: string;
  testType: 'basic' | 'full' | 'trading';
  timeout?: number;
}

export interface SyncAccountsRequest {
  accountIds?: string[];
  force?: boolean;
  includeHistory?: boolean;
}

export interface BrokerIntegration {
  brokerId: string;
  name: string;
  status: 'active' | 'inactive' | 'maintenance';
  supportLevel: 'full' | 'limited' | 'basic';
  features: {
    realTimeData: boolean;
    orderExecution: boolean;
    historyAccess: boolean;
    balanceSync: boolean;
  };
  connectionTypes: ('mt4' | 'mt5' | 'api' | 'bridge')[];
}
