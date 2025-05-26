import axios, { AxiosInstance, AxiosResponse } from 'axios';
import toast from 'react-hot-toast';
import { User, LoginCredentials, LoginResponse } from '../types/auth';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001';
const USER_SERVICE_URL = import.meta.env.VITE_USER_SERVICE_URL || 'http://localhost:3002';

// Create axios instance with default configuration
const createApiInstance = (baseURL: string): AxiosInstance => {
  const instance = axios.create({
    baseURL,
    timeout: 10000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor to add auth token
  instance.interceptors.request.use(
    (config) => {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor for error handling
  instance.interceptors.response.use(
    (response: AxiosResponse) => {
      return response;
    },
    (error) => {
      if (error.response?.status === 401) {
        // Unauthorized - clear token and redirect to login
        localStorage.removeItem('token');
        window.location.href = '/login';
      } else if (error.response?.status >= 500) {
        toast.error('Server error. Please try again later.');
      }
      return Promise.reject(error);
    }
  );

  return instance;
};

// API instances
const apiGateway = createApiInstance(API_BASE_URL);
const userServiceApi = createApiInstance(USER_SERVICE_URL);

// Authentication API
export const authAPI = {
  // Login user
  login: async (credentials: LoginCredentials): Promise<LoginResponse> => {
    try {
      const response = await userServiceApi.post('/api/v1/auth/login', credentials);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Login failed');
    }
  },

  // Register owner (one-time setup)
  registerOwner: async (userData: any): Promise<LoginResponse> => {
    try {
      const response = await userServiceApi.post('/api/v1/auth/register-owner', userData);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Registration failed');
    }
  },

  // Get current user
  getCurrentUser: async (): Promise<User> => {
    try {
      const response = await userServiceApi.get('/api/v1/users/profile');
      return response.data.user || response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get user data');
    }
  },

  // Update user profile
  updateProfile: async (userData: Partial<User>): Promise<User> => {
    try {
      const response = await userServiceApi.put('/api/v1/users/profile', userData);
      return response.data.user || response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to update profile');
    }
  },

  // Logout user
  logout: async (): Promise<void> => {
    try {
      await userServiceApi.post('/api/v1/auth/logout');
    } catch (error: any) {
      // Even if logout fails on server, we'll clear local data
      console.warn('Logout request failed:', error.message);
    }
  },

  // Refresh token
  refreshToken: async (): Promise<LoginResponse> => {
    try {
      const response = await userServiceApi.post('/api/v1/auth/refresh');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Token refresh failed');
    }
  },
};

// Trading API
export const tradingAPI = {
  // Get user trades
  getTrades: async (params?: any) => {
    try {
      const response = await apiGateway.get('/api/v1/trades', { params });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get trades');
    }
  },

  // Create new trade
  createTrade: async (tradeData: any) => {
    try {
      const response = await apiGateway.post('/api/v1/trades', tradeData);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to create trade');
    }
  },

  // Update trade status
  updateTradeStatus: async (tradeId: string, status: string) => {
    try {
      const response = await apiGateway.patch(`/api/v1/trades/${tradeId}/status`, { status });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to update trade');
    }
  },

  // Get portfolio summary
  getPortfolioSummary: async () => {
    try {
      const response = await apiGateway.get('/api/v1/portfolio/summary');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get portfolio data');
    }
  },
};

// Market Data API
export const marketDataAPI = {
  // Get current prices
  getCurrentPrices: async (symbols?: string[]) => {
    try {
      const params = symbols ? { symbols: symbols.join(',') } : {};
      const response = await apiGateway.get('/api/v1/market-data/prices', { params });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get market data');
    }
  },

  // Get historical data
  getHistoricalData: async (symbol: string, timeframe: string, limit?: number) => {
    try {
      const response = await apiGateway.get(`/api/v1/market-data/history`, {
        params: { symbol, timeframe, limit },
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get historical data');
    }
  },

  // Get market statistics
  getMarketStats: async () => {
    try {
      const response = await apiGateway.get('/api/v1/market-data/stats');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get market stats');
    }
  },

  // Get trading instruments
  getInstruments: async () => {
    try {
      const response = await apiGateway.get('/api/v1/market-data/instruments');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get instruments');
    }
  },
};

export default {
  authAPI,
  tradingAPI,
  marketDataAPI,
};
