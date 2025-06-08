"""
Oanda Broker API Integration Service
Comprehensive Oanda v20 REST API integration with streaming and advanced features
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
import requests
import pandas as pd
from enum import Enum
import threading
from queue import Queue
import ssl
import websocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstrumentType(Enum):
    """Instrument types supported by Oanda"""
    CURRENCY = "CURRENCY"
    CFD = "CFD"
    METAL = "METAL"

class OrderType(Enum):
    """Order types for Oanda"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"
    LIMIT_IF_TOUCHED = "LIMIT_IF_TOUCHED"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderState(Enum):
    """Order states"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"

class TimeInForce(Enum):
    """Time in force options"""
    FOK = "FOK"  # Fill or Kill
    IOC = "IOC"  # Immediate or Cancel
    GTC = "GTC"  # Good Till Cancelled
    GTD = "GTD"  # Good Till Date

class BrokerAPI_Oanda:
    """
    Oanda v20 REST API Integration
    
    Comprehensive trading platform integration providing:
    - Account management and authentication
    - Real-time streaming market data
    - Advanced order management system
    - Position and portfolio tracking
    - Risk management and compliance
    - Historical data and analytics
    - Automated trading capabilities
    """
    
    def __init__(self, api_key: str = None, account_id: str = None, 
                 demo_mode: bool = True, stream_timeout: int = 30):
        """
        Initialize Oanda API connection
        
        Args:
            api_key: Oanda API access token
            account_id: Trading account identifier
            demo_mode: Use practice environment if True
            stream_timeout: Streaming connection timeout in seconds
        """
        self.api_key = api_key or "demo_api_token"
        self.account_id = account_id or "demo_account_id"
        self.demo_mode = demo_mode
        self.stream_timeout = stream_timeout
        
        # API endpoints
        if demo_mode:
            self.api_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"
        else:
            self.api_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"
        
        # Connection state
        self.connected = False
        self.authenticated = False
        self.streaming = False
        self.session = requests.Session()
        
        # Account and trading state
        self.account_info = {}
        self.positions = {}
        self.orders = {}
        self.trades = {}
        self.transactions = []
        
        # Market data and streaming
        self.market_data = {}
        self.price_stream = None
        self.stream_thread = None
        self.price_callbacks = []
        self.subscribed_instruments = set()
        
        # Performance tracking
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.balance = 0.0
        self.equity = 0.0
        self.margin_used = 0.0
        self.margin_available = 0.0
        
        # Risk management
        self.max_position_size = 1000000  # Maximum position size in units
        self.max_daily_trades = 100       # Maximum trades per day
        self.max_drawdown_percent = 5.0   # Maximum drawdown percentage
        self.daily_trade_count = 0
        self.session_start_balance = 0.0
        
        # Headers for API requests
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        }
        
        logger.info(f"Oanda API initialized - Demo: {demo_mode}, Account: {self.account_id}")

    def connect(self) -> bool:
        """
        Establish connection to Oanda platform
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Test API connectivity
            response = self.session.get(
                f"{self.api_url}/v3/accounts",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                accounts = response.json().get('accounts', [])
                
                # Verify account access
                account_found = False
                for account in accounts:
                    if account['id'] == self.account_id:
                        account_found = True
                        break
                
                if not account_found and accounts:
                    # Use first available account if specified account not found
                    self.account_id = accounts[0]['id']
                    logger.warning(f"Using first available account: {self.account_id}")
                
                # Get account details
                if self.get_account_info():
                    self.connected = True
                    self.authenticated = True
                    self.session_start_balance = self.balance
                    
                    logger.info(f"Oanda connection established - Account: {self.account_id}")
                    return True
            
            logger.error(f"Oanda connection failed: {response.status_code} - {response.text}")
            return False
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Oanda connection error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected connection error: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from Oanda platform
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            # Stop streaming
            if self.streaming:
                self.stop_price_stream()
            
            # Close session
            if self.session:
                self.session.close()
            
            self.connected = False
            self.authenticated = False
            
            logger.info("Oanda disconnected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Oanda disconnection error: {e}")
            return False

    def get_account_info(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive account information
        
        Returns:
            Dict containing account details and metrics
        """
        try:
            if not self.api_key:
                logger.error("API key not provided")
                return {}
            
            response = self.session.get(
                f"{self.api_url}/v3/accounts/{self.account_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                account_data = response.json()['account']
                
                self.account_info = {
                    'id': account_data['id'],
                    'alias': account_data.get('alias', ''),
                    'currency': account_data['currency'],
                    'balance': float(account_data['balance']),
                    'unrealized_pnl': float(account_data['unrealizedPL']),
                    'realized_pnl': float(account_data['pl']),
                    'margin_used': float(account_data['marginUsed']),
                    'margin_available': float(account_data['marginAvailable']),
                    'margin_closeout_percent': float(account_data.get('marginCloseoutPercent', 0)),
                    'open_trade_count': int(account_data['openTradeCount']),
                    'open_position_count': int(account_data['openPositionCount']),
                    'pending_order_count': int(account_data['pendingOrderCount']),
                    'hedging_enabled': account_data.get('hedgingEnabled', False),
                    'last_transaction_id': account_data['lastTransactionID']
                }
                
                # Update internal state
                self.balance = self.account_info['balance']
                self.unrealized_pnl = self.account_info['unrealized_pnl']
                self.realized_pnl = self.account_info['realized_pnl']
                self.equity = self.balance + self.unrealized_pnl
                self.margin_used = self.account_info['margin_used']
                self.margin_available = self.account_info['margin_available']
                
                logger.info(f"Account info updated - Balance: {self.balance}, Equity: {self.equity}")
                return self.account_info
            
            logger.error(f"Failed to get account info: {response.status_code} - {response.text}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def execute_trade(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade order on Oanda platform
        
        Args:
            trade_details: Dictionary containing trade parameters:
                - instrument: Currency pair (e.g., 'EUR_USD')
                - units: Position size (positive for buy, negative for sell)
                - order_type: Order type (MARKET, LIMIT, STOP, etc.)
                - price: Price for limit/stop orders (optional)
                - stop_loss: Stop loss price (optional)
                - take_profit: Take profit price (optional)
                - time_in_force: Time in force (GTC, FOK, IOC, GTD)
                - expiry_time: Expiry time for GTD orders (optional)
        
        Returns:
            Dict containing order execution result
        """
        try:
            if not self.connected:
                return {'success': False, 'error': 'Not connected to Oanda'}
            
            # Validate trade parameters
            validation_result = self._validate_trade(trade_details)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
            
            # Risk management checks
            risk_check = self._check_risk_limits(trade_details)
            if not risk_check['allowed']:
                return {'success': False, 'error': risk_check['reason']}
            
            # Prepare order payload
            order_payload = self._build_order_payload(trade_details)
            
            # Execute order
            response = self.session.post(
                f"{self.api_url}/v3/accounts/{self.account_id}/orders",
                headers=self.headers,
                json=order_payload
            )
            
            if response.status_code == 201:
                result = response.json()
                
                # Extract order information
                order_transaction = result.get('orderCreateTransaction', {})
                order_fill = result.get('orderFillTransaction', {})
                
                order_id = order_transaction.get('id')
                trade_id = order_fill.get('tradeOpened', {}).get('tradeID') if order_fill else None
                
                # Track order and trade
                if order_id:
                    self.orders[order_id] = {
                        'id': order_id,
                        'instrument': trade_details['instrument'],
                        'units': trade_details['units'],
                        'type': trade_details.get('order_type', 'MARKET'),
                        'state': order_transaction.get('reason', 'PENDING'),
                        'time': order_transaction.get('time'),
                        'price': order_fill.get('price') if order_fill else trade_details.get('price'),
                        'trade_id': trade_id
                    }
                
                # Update trade count
                self.daily_trade_count += 1
                
                logger.info(f"Trade executed successfully - Order ID: {order_id}, Trade ID: {trade_id}")
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'trade_id': trade_id,
                    'transaction': result,
                    'fill_price': order_fill.get('price') if order_fill else None,
                    'timestamp': datetime.now()
                }
            
            logger.error(f"Trade execution failed: {response.status_code} - {response.text}")
            return {'success': False, 'error': f'Execution failed: {response.text}'}
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'success': False, 'error': str(e)}

    def get_positions(self) -> Dict[str, Any]:
        """
        Get all current positions
        
        Returns:
            Dict containing position details by instrument
        """
        try:
            if not self.connected:
                return {}
            
            response = self.session.get(
                f"{self.api_url}/v3/accounts/{self.account_id}/positions",
                headers=self.headers
            )
            
            if response.status_code == 200:
                positions_data = response.json().get('positions', [])
                
                self.positions = {}
                for pos in positions_data:
                    instrument = pos['instrument']
                    long_units = float(pos['long']['units'])
                    short_units = float(pos['short']['units'])
                    net_units = long_units + short_units
                    
                    if net_units != 0:
                        unrealized_pnl = float(pos['long']['unrealizedPL']) + float(pos['short']['unrealizedPL'])
                        
                        self.positions[instrument] = {
                            'instrument': instrument,
                            'units': net_units,
                            'side': 'long' if net_units > 0 else 'short',
                            'unrealized_pnl': unrealized_pnl,
                            'long_units': long_units,
                            'short_units': short_units,
                            'long_average_price': float(pos['long']['averagePrice']) if long_units != 0 else 0,
                            'short_average_price': float(pos['short']['averagePrice']) if short_units != 0 else 0,
                            'margin_used': float(pos['marginUsed'])
                        }
                
                return self.positions
            
            logger.error(f"Failed to get positions: {response.status_code}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    def close_position(self, instrument: str, units: Optional[str] = None, 
                      long_units: str = "ALL", short_units: str = "ALL") -> Dict[str, Any]:
        """
        Close position for specified instrument
        
        Args:
            instrument: Currency pair to close
            units: Specific units to close (for net position)
            long_units: Long units to close ("ALL" or specific amount)
            short_units: Short units to close ("ALL" or specific amount)
        
        Returns:
            Dict containing close result
        """
        try:
            if instrument not in self.positions:
                return {'success': False, 'error': 'Position not found'}
            
            # Prepare close position payload
            payload = {}
            
            if units:
                # Close net position
                payload['units'] = str(units)
            else:
                # Close long and short separately
                if long_units:
                    payload['longUnits'] = str(long_units)
                if short_units:
                    payload['shortUnits'] = str(short_units)
            
            response = self.session.put(
                f"{self.api_url}/v3/accounts/{self.account_id}/positions/{instrument}/close",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract close transactions
                long_close = result.get('longOrderFillTransaction')
                short_close = result.get('shortOrderFillTransaction')
                
                close_info = {
                    'success': True,
                    'instrument': instrument,
                    'timestamp': datetime.now(),
                    'transactions': result
                }
                
                if long_close:
                    close_info['long_close'] = {
                        'units': long_close.get('units'),
                        'price': long_close.get('price'),
                        'pnl': long_close.get('pl')
                    }
                
                if short_close:
                    close_info['short_close'] = {
                        'units': short_close.get('units'),
                        'price': short_close.get('price'),
                        'pnl': short_close.get('pl')
                    }
                
                logger.info(f"Position closed - {instrument}")
                return close_info
            
            logger.error(f"Failed to close position: {response.status_code} - {response.text}")
            return {'success': False, 'error': response.text}
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}

    def get_market_data(self, instruments: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get current market pricing data
        
        Args:
            instruments: Single instrument or list of instruments
        
        Returns:
            Dict containing pricing data by instrument
        """
        try:
            if isinstance(instruments, str):
                instruments = [instruments]
            
            instruments_param = ','.join(instruments)
            
            response = self.session.get(
                f"{self.api_url}/v3/accounts/{self.account_id}/pricing",
                headers=self.headers,
                params={'instruments': instruments_param}
            )
            
            if response.status_code == 200:
                prices = response.json().get('prices', [])
                
                market_data = {}
                for price in prices:
                    instrument = price['instrument']
                    
                    # Get best bid/ask
                    bids = price.get('bids', [])
                    asks = price.get('asks', [])
                    
                    if bids and asks:
                        bid = float(bids[0]['price'])
                        ask = float(asks[0]['price'])
                        
                        market_data[instrument] = {
                            'instrument': instrument,
                            'bid': bid,
                            'ask': ask,
                            'spread': ask - bid,
                            'timestamp': price.get('time'),
                            'tradeable': price.get('tradeable', True),
                            'closeout_bid': float(price.get('closeoutBid', bid)),
                            'closeout_ask': float(price.get('closeoutAsk', ask))
                        }
                        
                        # Update internal cache
                        self.market_data[instrument] = market_data[instrument]
                
                return market_data
            
            logger.error(f"Failed to get market data: {response.status_code}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}

    def start_price_stream(self, instruments: List[str], 
                          callback: Optional[Callable] = None) -> bool:
        """
        Start real-time price streaming
        
        Args:
            instruments: List of instruments to stream
            callback: Optional callback function for price updates
        
        Returns:
            bool: True if streaming started successfully
        """
        try:
            if self.streaming:
                logger.warning("Price stream already running")
                return True
            
            if callback:
                self.price_callbacks.append(callback)
            
            self.subscribed_instruments = set(instruments)
            instruments_param = ','.join(instruments)
            
            # Start streaming in separate thread
            def stream_prices():
                try:
                    response = self.session.get(
                        f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream",
                        headers=self.headers,
                        params={'instruments': instruments_param},
                        stream=True,
                        timeout=self.stream_timeout
                    )
                    
                    for line in response.iter_lines():
                        if not self.streaming:
                            break
                        
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                self._process_price_update(data)
                            except json.JSONDecodeError:
                                continue
                    
                except Exception as e:
                    logger.error(f"Price streaming error: {e}")
                finally:
                    self.streaming = False
            
            self.stream_thread = threading.Thread(target=stream_prices)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            self.streaming = True
            logger.info(f"Price streaming started for: {instruments}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start price stream: {e}")
            return False

    def stop_price_stream(self) -> bool:
        """
        Stop real-time price streaming
        
        Returns:
            bool: True if streaming stopped successfully
        """
        try:
            self.streaming = False
            
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=5)
            
            logger.info("Price streaming stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping price stream: {e}")
            return False

    def get_historical_data(self, instrument: str, granularity: str = "M1", 
                           count: int = 500, from_time: str = None, 
                           to_time: str = None) -> pd.DataFrame:
        """
        Get historical candle data
        
        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            granularity: Candle granularity (S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30, H1, H2, H3, H4, H6, H8, H12, D, W, M)
            count: Number of candles (max 5000)
            from_time: Start time in RFC3339 format
            to_time: End time in RFC3339 format
        
        Returns:
            DataFrame with historical data
        """
        try:
            params = {
                'granularity': granularity,
                'price': 'MBA',  # Mid, Bid, Ask prices
            }
            
            if count:
                params['count'] = min(count, 5000)
            if from_time:
                params['from'] = from_time
            if to_time:
                params['to'] = to_time
            
            response = self.session.get(
                f"{self.api_url}/v3/accounts/{self.account_id}/instruments/{instrument}/candles",
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                candles = response.json().get('candles', [])
                
                data = []
                for candle in candles:
                    if candle.get('complete', False):
                        mid = candle.get('mid', {})
                        bid = candle.get('bid', {})
                        ask = candle.get('ask', {})
                        
                        data.append({
                            'timestamp': pd.to_datetime(candle['time']),
                            'open': float(mid.get('o', 0)),
                            'high': float(mid.get('h', 0)),
                            'low': float(mid.get('l', 0)),
                            'close': float(mid.get('c', 0)),
                            'volume': int(candle.get('volume', 0)),
                            'bid_open': float(bid.get('o', 0)) if bid else None,
                            'bid_high': float(bid.get('h', 0)) if bid else None,
                            'bid_low': float(bid.get('l', 0)) if bid else None,
                            'bid_close': float(bid.get('c', 0)) if bid else None,
                            'ask_open': float(ask.get('o', 0)) if ask else None,
                            'ask_high': float(ask.get('h', 0)) if ask else None,
                            'ask_low': float(ask.get('l', 0)) if ask else None,
                            'ask_close': float(ask.get('c', 0)) if ask else None,
                        })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                
                return df
            
            logger.error(f"Failed to get historical data: {response.status_code}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def _validate_trade(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade parameters"""
        required_fields = ['instrument', 'units']
        
        for field in required_fields:
            if field not in trade_details:
                return {'valid': False, 'error': f'Missing required field: {field}'}
        
        # Validate units
        try:
            units = float(trade_details['units'])
            if abs(units) > self.max_position_size:
                return {'valid': False, 'error': f'Position size exceeds maximum: {abs(units)}'}
        except (ValueError, TypeError):
            return {'valid': False, 'error': 'Invalid units value'}
        
        # Validate instrument format
        instrument = trade_details['instrument']
        if '_' not in instrument or len(instrument.split('_')) != 2:
            return {'valid': False, 'error': 'Invalid instrument format (should be CCY1_CCY2)'}
        
        return {'valid': True}

    def _check_risk_limits(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """Check risk management limits"""
        # Daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            return {'allowed': False, 'reason': 'Daily trade limit exceeded'}
        
        # Drawdown check
        if self.session_start_balance > 0:
            current_drawdown = ((self.session_start_balance - self.equity) / self.session_start_balance) * 100
            if current_drawdown > self.max_drawdown_percent:
                return {'allowed': False, 'reason': f'Maximum drawdown exceeded: {current_drawdown:.2f}%'}
        
        # Margin check
        if self.margin_available <= 0:
            return {'allowed': False, 'reason': 'Insufficient margin available'}
        
        return {'allowed': True}

    def _build_order_payload(self, trade_details: Dict[str, Any]) -> Dict[str, Any]:
        """Build order payload for API request"""
        order_type = trade_details.get('order_type', 'MARKET').upper()
        
        order = {
            'type': order_type,
            'instrument': trade_details['instrument'],
            'units': str(trade_details['units']),
            'timeInForce': trade_details.get('time_in_force', 'FOK')
        }
        
        # Add type-specific parameters
        if order_type in ['LIMIT', 'STOP', 'MARKET_IF_TOUCHED', 'LIMIT_IF_TOUCHED']:
            if 'price' in trade_details:
                order['price'] = str(trade_details['price'])
        
        # Add stop loss
        if 'stop_loss' in trade_details:
            order['stopLossOnFill'] = {
                'price': str(trade_details['stop_loss'])
            }
        
        # Add take profit
        if 'take_profit' in trade_details:
            order['takeProfitOnFill'] = {
                'price': str(trade_details['take_profit'])
            }
        
        # Add expiry time for GTD orders
        if trade_details.get('time_in_force') == 'GTD' and 'expiry_time' in trade_details:
            order['gtdTime'] = trade_details['expiry_time']
        
        return {'order': order}

    def _process_price_update(self, data: Dict[str, Any]):
        """Process incoming price stream data"""
        try:
            if data.get('type') == 'PRICE':
                instrument = data.get('instrument')
                if instrument:
                    # Update market data
                    bids = data.get('bids', [])
                    asks = data.get('asks', [])
                    
                    if bids and asks:
                        bid = float(bids[0]['price'])
                        ask = float(asks[0]['price'])
                        
                        price_data = {
                            'instrument': instrument,
                            'bid': bid,
                            'ask': ask,
                            'spread': ask - bid,
                            'timestamp': data.get('time'),
                            'tradeable': data.get('tradeable', True)
                        }
                        
                        self.market_data[instrument] = price_data
                        
                        # Call registered callbacks
                        for callback in self.price_callbacks:
                            try:
                                callback(price_data)
                            except Exception as e:
                                logger.error(f"Price callback error: {e}")
            
        except Exception as e:
            logger.error(f"Error processing price update: {e}")

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status by ID"""
        try:
            response = self.session.get(
                f"{self.api_url}/v3/accounts/{self.account_id}/orders/{order_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                order = response.json().get('order', {})
                return {
                    'id': order.get('id'),
                    'state': order.get('state'),
                    'instrument': order.get('instrument'),
                    'units': order.get('units'),
                    'type': order.get('type'),
                    'create_time': order.get('createTime'),
                    'fill_time': order.get('filledTime'),
                    'cancel_time': order.get('cancelledTime')
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            response = self.session.put(
                f"{self.api_url}/v3/accounts/{self.account_id}/orders/{order_id}/cancel",
                headers=self.headers
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    def __del__(self):
        """Cleanup on object destruction"""
        if hasattr(self, 'connected') and self.connected:
            self.disconnect()
