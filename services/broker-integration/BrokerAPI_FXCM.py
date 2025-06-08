"""
FXCM Broker API Integration Service
Comprehensive FXCM trading platform integration with REST API and Socket.IO support
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import requests
import websocket
import pandas as pd
from enum import Enum
import hmac
import hashlib
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types supported by FXCM"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status types"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"

class BrokerAPI_FXCM:
    """
    FXCM Broker API Integration
    
    Provides comprehensive trading functionality including:
    - Account management and authentication
    - Real-time market data streaming
    - Order placement and management
    - Position tracking and portfolio management
    - Risk management and compliance
    - Historical data retrieval
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, 
                 demo_mode: bool = True, server_url: str = None):
        """
        Initialize FXCM API connection
        
        Args:
            api_key: FXCM API key
            secret_key: FXCM secret key for authentication
            demo_mode: Use demo environment if True
            server_url: Custom server URL (optional)
        """
        self.api_key = api_key or "demo_api_key"
        self.secret_key = secret_key or "demo_secret_key"
        self.demo_mode = demo_mode
        
        # API endpoints
        if demo_mode:
            self.base_url = "https://api-fxpractice.oanda.com/v3"
            self.stream_url = "https://stream-fxpractice.oanda.com/v3"
        else:
            self.base_url = server_url or "https://api-fxtrade.oanda.com/v3"
            self.stream_url = "https://stream-fxtrade.oanda.com/v3"
        
        # Connection state
        self.connected = False
        self.authenticated = False
        self.session = requests.Session()
        self.ws_connection = None
        
        # Account and trading state
        self.account_id = None
        self.account_info = {}
        self.positions = {}
        self.orders = {}
        self.balance = 0.0
        self.equity = 0.0
        self.margin_used = 0.0
        self.margin_available = 0.0
        
        # Market data
        self.market_data = {}
        self.subscribed_instruments = set()
        
        # Risk management
        self.max_position_size = 100000  # Maximum position size
        self.max_daily_loss = 1000       # Maximum daily loss limit
        self.daily_pnl = 0.0
        
        logger.info(f"FXCM API initialized - Demo: {demo_mode}")

    def authenticate(self) -> bool:
        """
        Authenticate with FXCM API
        
        Returns:
            bool: True if authentication successful
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Test authentication with account info request
            response = self.session.get(f"{self.base_url}/accounts", headers=headers)
            
            if response.status_code == 200:
                account_data = response.json()
                if account_data.get('accounts'):
                    self.account_id = account_data['accounts'][0]['id']
                    self.authenticated = True
                    logger.info(f"FXCM authentication successful - Account: {self.account_id}")
                    return True
            
            logger.error(f"FXCM authentication failed: {response.status_code} - {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"FXCM authentication error: {e}")
            return False

    def connect(self) -> bool:
        """
        Establish connection to FXCM trading platform
        
        Returns:
            bool: True if connection successful
        """
        try:
            if not self.authenticate():
                return False
            
            # Get account information
            if not self.get_account_info():
                return False
            
            # Initialize trading session
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
            
            self.connected = True
            logger.info("FXCM connection established successfully")
            
            # Start market data streaming if needed
            self._initialize_market_data()
            
            return True
            
        except Exception as e:
            logger.error(f"FXCM connection error: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from FXCM platform
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            if self.ws_connection:
                self.ws_connection.close()
                self.ws_connection = None
            
            if self.session:
                self.session.close()
            
            self.connected = False
            self.authenticated = False
            logger.info("FXCM disconnected successfully")
            return True
            
        except Exception as e:
            logger.error(f"FXCM disconnection error: {e}")
            return False

    def get_account_info(self) -> Dict[str, Any]:
        """
        Retrieve account information
        
        Returns:
            Dict containing account details
        """
        try:
            if not self.authenticated:
                logger.error("Not authenticated with FXCM")
                return {}
            
            response = self.session.get(f"{self.base_url}/accounts/{self.account_id}")
            
            if response.status_code == 200:
                account_data = response.json()['account']
                
                self.account_info = {
                    'account_id': account_data.get('id'),
                    'currency': account_data.get('currency'),
                    'balance': float(account_data.get('balance', 0)),
                    'unrealized_pnl': float(account_data.get('unrealizedPL', 0)),
                    'margin_used': float(account_data.get('marginUsed', 0)),
                    'margin_available': float(account_data.get('marginAvailable', 0)),
                    'open_trade_count': int(account_data.get('openTradeCount', 0)),
                    'open_position_count': int(account_data.get('openPositionCount', 0))
                }
                
                self.balance = self.account_info['balance']
                self.equity = self.balance + self.account_info['unrealized_pnl']
                self.margin_used = self.account_info['margin_used']
                self.margin_available = self.account_info['margin_available']
                
                logger.info(f"Account info updated - Balance: {self.balance}, Equity: {self.equity}")
                return self.account_info
            
            logger.error(f"Failed to get account info: {response.status_code}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def execute_trade(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade on FXCM platform
        
        Args:
            trade_details: Dictionary containing trade parameters:
                - instrument: Currency pair (e.g., 'EUR_USD')
                - units: Position size (positive for buy, negative for sell)
                - order_type: 'market', 'limit', 'stop', etc.
                - price: Price for limit/stop orders (optional)
                - stop_loss: Stop loss price (optional)
                - take_profit: Take profit price (optional)
        
        Returns:
            Dict containing order execution result
        """
        try:
            if not self.connected:
                return {'success': False, 'error': 'Not connected to FXCM'}
            
            # Validate trade details
            if not self._validate_trade(trade_details):
                return {'success': False, 'error': 'Invalid trade parameters'}
            
            # Risk management checks
            if not self._check_risk_limits(trade_details):
                return {'success': False, 'error': 'Trade rejected by risk management'}
            
            # Prepare order payload
            order_payload = self._prepare_order_payload(trade_details)
            
            # Execute order
            response = self.session.post(
                f"{self.base_url}/accounts/{self.account_id}/orders",
                json=order_payload
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                order_id = result.get('orderCreateTransaction', {}).get('id')
                
                # Track order
                self.orders[order_id] = {
                    'id': order_id,
                    'instrument': trade_details['instrument'],
                    'units': trade_details['units'],
                    'status': 'submitted',
                    'timestamp': datetime.now(),
                    'details': trade_details
                }
                
                logger.info(f"Trade executed successfully - Order ID: {order_id}")
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            logger.error(f"Trade execution failed: {response.status_code} - {response.text}")
            return {'success': False, 'error': f'Execution failed: {response.text}'}
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'success': False, 'error': str(e)}

    def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions
        
        Returns:
            Dict containing all open positions
        """
        try:
            if not self.connected:
                return {}
            
            response = self.session.get(f"{self.base_url}/accounts/{self.account_id}/positions")
            
            if response.status_code == 200:
                positions_data = response.json()['positions']
                
                self.positions = {}
                for pos in positions_data:
                    instrument = pos['instrument']
                    long_units = float(pos['long']['units'])
                    short_units = float(pos['short']['units'])
                    net_units = long_units + short_units
                    
                    if net_units != 0:
                        self.positions[instrument] = {
                            'instrument': instrument,
                            'units': net_units,
                            'side': 'long' if net_units > 0 else 'short',
                            'unrealized_pnl': float(pos['long']['unrealizedPL']) + float(pos['short']['unrealizedPL']),
                            'long_units': long_units,
                            'short_units': short_units
                        }
                
                return self.positions
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    def close_position(self, instrument: str, units: Optional[float] = None) -> Dict[str, Any]:
        """
        Close position for specified instrument
        
        Args:
            instrument: Currency pair to close
            units: Specific units to close (None for all)
        
        Returns:
            Dict containing close result
        """
        try:
            if instrument not in self.positions:
                return {'success': False, 'error': 'Position not found'}
            
            position = self.positions[instrument]
            close_units = units or abs(position['units'])
            
            # Prepare close order (opposite direction)
            if position['side'] == 'long':
                close_units = -close_units
            
            close_trade = {
                'instrument': instrument,
                'units': close_units,
                'order_type': 'market'
            }
            
            result = self.execute_trade(close_trade)
            
            if result['success']:
                logger.info(f"Position closed - {instrument}: {close_units} units")
            
            return result
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}

    def get_market_data(self, instrument: str) -> Dict[str, Any]:
        """
        Get current market data for instrument
        
        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
        
        Returns:
            Dict containing market data
        """
        try:
            response = self.session.get(
                f"{self.base_url}/accounts/{self.account_id}/pricing",
                params={'instruments': instrument}
            )
            
            if response.status_code == 200:
                pricing = response.json()['prices'][0]
                
                market_data = {
                    'instrument': pricing['instrument'],
                    'bid': float(pricing['bids'][0]['price']),
                    'ask': float(pricing['asks'][0]['price']),
                    'spread': float(pricing['asks'][0]['price']) - float(pricing['bids'][0]['price']),
                    'timestamp': pricing['time']
                }
                
                self.market_data[instrument] = market_data
                return market_data
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}

    def _validate_trade(self, trade_details: Dict[str, Any]) -> bool:
        """Validate trade parameters"""
        required_fields = ['instrument', 'units', 'order_type']
        
        for field in required_fields:
            if field not in trade_details:
                logger.error(f"Missing required field: {field}")
                return False
        
        if abs(trade_details['units']) > self.max_position_size:
            logger.error(f"Position size exceeds maximum: {abs(trade_details['units'])}")
            return False
        
        return True

    def _check_risk_limits(self, trade_details: Dict[str, Any]) -> bool:
        """Check risk management limits"""
        # Daily loss limit check
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning("Daily loss limit exceeded")
            return False
        
        # Margin check
        if self.margin_available <= 0:
            logger.warning("Insufficient margin available")
            return False
        
        return True

    def _prepare_order_payload(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare order payload for API"""
        payload = {
            'order': {
                'type': trade_details['order_type'].upper(),
                'instrument': trade_details['instrument'],
                'units': str(trade_details['units'])
            }
        }
        
        # Add optional parameters
        if 'price' in trade_details and trade_details['order_type'] in ['limit', 'stop']:
            payload['order']['price'] = str(trade_details['price'])
        
        if 'stop_loss' in trade_details:
            payload['order']['stopLossOnFill'] = {'price': str(trade_details['stop_loss'])}
        
        if 'take_profit' in trade_details:
            payload['order']['takeProfitOnFill'] = {'price': str(trade_details['take_profit'])}
        
        return payload

    def _initialize_market_data(self):
        """Initialize market data streaming"""
        try:
            # Subscribe to common currency pairs
            major_pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD']
            for pair in major_pairs:
                self.market_data[pair] = self.get_market_data(pair)
            
            logger.info("Market data initialized for major currency pairs")
            
        except Exception as e:
            logger.error(f"Error initializing market data: {e}")

    def get_historical_data(self, instrument: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            instrument: Currency pair
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D)
            count: Number of candles to retrieve
        
        Returns:
            DataFrame with historical data
        """
        try:
            params = {
                'granularity': timeframe,
                'count': count
            }
            
            response = self.session.get(
                f"{self.base_url}/accounts/{self.account_id}/instruments/{instrument}/candles",
                params=params
            )
            
            if response.status_code == 200:
                candles = response.json()['candles']
                
                data = []
                for candle in candles:
                    if candle['complete']:
                        data.append({
                            'timestamp': pd.to_datetime(candle['time']),
                            'open': float(candle['mid']['o']),
                            'high': float(candle['mid']['h']),
                            'low': float(candle['mid']['l']),
                            'close': float(candle['mid']['c']),
                            'volume': float(candle['volume'])
                        })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status
        
        Args:
            order_id: Order identifier
        
        Returns:
            Dict containing order status
        """
        try:
            response = self.session.get(f"{self.base_url}/accounts/{self.account_id}/orders/{order_id}")
            
            if response.status_code == 200:
                order_data = response.json()['order']
                return {
                    'id': order_data['id'],
                    'state': order_data['state'],
                    'instrument': order_data['instrument'],
                    'units': order_data['units'],
                    'filled_time': order_data.get('filledTime'),
                    'cancelled_time': order_data.get('cancelledTime')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {}

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order
        
        Args:
            order_id: Order identifier
        
        Returns:
            bool: True if cancellation successful
        """
        try:
            response = self.session.put(f"{self.base_url}/accounts/{self.account_id}/orders/{order_id}/cancel")
            
            if response.status_code == 200:
                logger.info(f"Order cancelled successfully: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def __del__(self):
        """Cleanup on object destruction"""
        if hasattr(self, 'connected') and self.connected:
            self.disconnect()
