export interface MarketData {
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  timestamp: string;
  volume?: number;
  high?: number;
  low?: number;
  open?: number;
  close?: number;
  change?: number;
  changePercent?: number;
}

export interface TradingSignal {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  confidence: number;
  entryPrice: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning: string;
  timestamp: string;
  source: 'ai' | 'technical' | 'fundamental';
}

export interface OrderUpdate {
  orderId: string;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  symbol: string;
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  side: 'buy' | 'sell';
  quantity: number;
  price?: number;
  filledQuantity?: number;
  averagePrice?: number;
  timestamp: string;
  reason?: string;
}

export interface PriceAlert {
  id: string;
  symbol: string;
  condition: 'above' | 'below' | 'crosses_up' | 'crosses_down';
  targetPrice: number;
  currentPrice: number;
  message: string;
  timestamp: string;
  triggered: boolean;
}

export interface WebSocketMessage {
  type: 'market_data' | 'trading_signal' | 'order_update' | 'price_alert' | 'error';
  data: MarketData | TradingSignal | OrderUpdate | PriceAlert | any;
  timestamp: string;
}
