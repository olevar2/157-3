import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import toast from 'react-hot-toast';

import { useAuth } from './AuthContext';
import { MarketData, TradingSignal, OrderUpdate, PriceAlert } from '../types/websocket';

interface WebSocketContextType {
  socket: Socket | null;
  connected: boolean;
  marketData: Record<string, MarketData>;
  subscribeToSymbol: (symbol: string) => void;
  unsubscribeFromSymbol: (symbol: string) => void;
  subscribeToPriceAlerts: () => void;
  subscribeToOrderUpdates: () => void;
  subscribeToTradingSignals: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const { user, isAuthenticated } = useAuth();
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);
  const [marketData, setMarketData] = useState<Record<string, MarketData>>({});

  useEffect(() => {
    if (isAuthenticated && user) {      // Initialize WebSocket connection to Market Data Service
      const newSocket = io(import.meta.env.VITE_WS_URL || 'http://localhost:3004', {
        auth: {
          token: localStorage.getItem('token'),
        },
        transports: ['websocket', 'polling'],
        timeout: 10000,
        retries: 3,
      });

      // Connection event handlers
      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setConnected(true);
        toast.success('Real-time connection established');
      });

      newSocket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setConnected(false);
        if (reason === 'io server disconnect') {
          // Server disconnected, try to reconnect
          newSocket.connect();
        } else {
          toast.error('Real-time connection lost');
        }
      });

      newSocket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setConnected(false);
        toast.error('Failed to connect to real-time services');
      });

      // Market data updates
      newSocket.on('market_data', (data: MarketData) => {
        setMarketData(prev => ({
          ...prev,
          [data.symbol]: data,
        }));
      });

      // Trading signals
      newSocket.on('trading_signal', (signal: TradingSignal) => {
        toast.success(
          `Trading Signal: ${signal.type.toUpperCase()} ${signal.symbol} at ${signal.price}`,
          {
            duration: 8000,
            icon: '📈',
          }
        );
      });

      // Order updates
      newSocket.on('order_update', (update: OrderUpdate) => {
        const statusColors = {
          filled: '✅',
          cancelled: '❌',
          rejected: '⚠️',
          partially_filled: '🔄',
        };

        const icon = statusColors[update.status] || '📊';
        const message = `Order ${update.status}: ${update.symbol} ${update.side} ${update.quantity}`;
        
        if (update.status === 'filled') {
          toast.success(message, { icon, duration: 6000 });
        } else if (update.status === 'cancelled' || update.status === 'rejected') {
          toast.error(message, { icon, duration: 6000 });
        } else {
          toast(message, { icon, duration: 4000 });
        }
      });

      // Price alerts
      newSocket.on('price_alert', (alert: PriceAlert) => {
        toast(
          `Price Alert: ${alert.symbol} ${alert.condition} ${alert.price}`,
          {
            icon: '🔔',
            duration: 8000,
            style: {
              background: '#ff9800',
              color: '#000',
            },
          }
        );
      });

      // Account updates
      newSocket.on('account_update', (data: any) => {
        console.log('Account update:', data);
        // Handle account balance/margin updates
      });

      // Risk alerts
      newSocket.on('risk_alert', (alert: any) => {
        toast.error(
          `Risk Alert: ${alert.message}`,
          {
            icon: '⚠️',
            duration: 10000,
          }
        );
      });

      // System notifications
      newSocket.on('system_notification', (notification: any) => {
        toast(notification.message, {
          icon: '🔔',
          duration: 6000,
        });
      });

      setSocket(newSocket);

      return () => {
        newSocket.disconnect();
      };
    } else {
      // Clean up if not authenticated
      if (socket) {
        socket.disconnect();
        setSocket(null);
        setConnected(false);
        setMarketData({});
      }
    }
  }, [isAuthenticated, user]);

  const subscribeToSymbol = (symbol: string) => {
    if (socket && connected) {
      socket.emit('subscribe_market_data', { symbol });
      console.log(`Subscribed to market data for ${symbol}`);
    }
  };

  const unsubscribeFromSymbol = (symbol: string) => {
    if (socket && connected) {
      socket.emit('unsubscribe_market_data', { symbol });
      setMarketData(prev => {
        const updated = { ...prev };
        delete updated[symbol];
        return updated;
      });
      console.log(`Unsubscribed from market data for ${symbol}`);
    }
  };

  const subscribeToPriceAlerts = () => {
    if (socket && connected) {
      socket.emit('subscribe_price_alerts');
      console.log('Subscribed to price alerts');
    }
  };

  const subscribeToOrderUpdates = () => {
    if (socket && connected) {
      socket.emit('subscribe_order_updates');
      console.log('Subscribed to order updates');
    }
  };

  const subscribeToTradingSignals = () => {
    if (socket && connected) {
      socket.emit('subscribe_trading_signals');
      console.log('Subscribed to trading signals');
    }
  };

  const value: WebSocketContextType = {
    socket,
    connected,
    marketData,
    subscribeToSymbol,
    unsubscribeFromSymbol,
    subscribeToPriceAlerts,
    subscribeToOrderUpdates,
    subscribeToTradingSignals,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};
