"""
Platform3 WebSocket Server for Real-Time Python-TypeScript Communication
High-performance WebSocket server for streaming data between Python AI engines and TypeScript services
"""

import asyncio
import websockets
import json
import logging
import time
from typing import Dict, Set, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketDataUpdate:
    """Real-time market data update"""
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    volume: int = 0
    change: float = 0.0
    change_percent: float = 0.0

@dataclass
class TradingSignalUpdate:
    """Real-time trading signal update"""
    symbol: str
    signal: str  # buy, sell, hold
    confidence: float
    price_target: Optional[float]
    stop_loss: Optional[float]
    risk_score: float
    reasoning: str
    timestamp: datetime

@dataclass
class SystemAlert:
    """System alert/notification"""
    level: str  # info, warning, error, critical
    message: str
    source: str
    timestamp: datetime
    data: Dict[str, Any] = None

class WebSocketConnectionManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self):
        self.connections: Set[WebSocketServerProtocol] = set()
        self.subscriptions: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.connection_metadata: Dict[WebSocketServerProtocol, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        self.connections.add(websocket)
        self.connection_metadata[websocket] = {
            'connected_at': datetime.now(),
            'path': path,
            'subscriptions': set(),
            'message_count': 0
        }
        
        logger.info(f"New WebSocket connection: {websocket.remote_address}")
        
        # Send welcome message
        await self.send_to_connection(websocket, {
            'type': 'connection_established',
            'timestamp': datetime.now().isoformat(),
            'server': 'platform3-websocket-bridge'
        })
        
    async def disconnect(self, websocket: WebSocketServerProtocol):
        """Handle WebSocket disconnection"""
        if websocket in self.connections:
            self.connections.remove(websocket)
            
            # Remove from all subscriptions
            for topic_connections in self.subscriptions.values():
                topic_connections.discard(websocket)
                
            # Clean up metadata
            if websocket in self.connection_metadata:
                metadata = self.connection_metadata.pop(websocket)
                logger.info(f"WebSocket disconnected: {websocket.remote_address}, "
                          f"messages: {metadata['message_count']}, "
                          f"duration: {datetime.now() - metadata['connected_at']}")
    
    async def subscribe(self, websocket: WebSocketServerProtocol, topic: str):
        """Subscribe connection to a topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
            
        self.subscriptions[topic].add(websocket)
        
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]['subscriptions'].add(topic)
            
        logger.info(f"Connection {websocket.remote_address} subscribed to {topic}")
        
    async def unsubscribe(self, websocket: WebSocketServerProtocol, topic: str):
        """Unsubscribe connection from a topic"""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(websocket)
            
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]['subscriptions'].discard(topic)
            
        logger.info(f"Connection {websocket.remote_address} unsubscribed from {topic}")
    
    async def send_to_connection(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """Send message to specific connection"""
        try:
            await websocket.send(json.dumps(message, default=str))
            
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['message_count'] += 1
                
        except ConnectionClosed:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error sending message to {websocket.remote_address}: {e}")
            
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """Broadcast message to all connections subscribed to a topic"""
        if topic not in self.subscriptions:
            return
            
        disconnected_connections = set()
        
        for websocket in self.subscriptions[topic]:
            try:
                await websocket.send(json.dumps(message, default=str))
                
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]['message_count'] += 1
                    
            except ConnectionClosed:
                disconnected_connections.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to {websocket.remote_address}: {e}")
                disconnected_connections.add(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            await self.disconnect(websocket)
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = len(self.connections)
        topic_stats = {topic: len(connections) for topic, connections in self.subscriptions.items()}
        
        return {
            'total_connections': total_connections,
            'topics': topic_stats,
            'timestamp': datetime.now().isoformat()
        }

# Global connection manager
connection_manager = WebSocketConnectionManager()

async def handle_websocket_message(websocket: WebSocketServerProtocol, message: str):
    """Handle incoming WebSocket messages"""
    try:
        data = json.loads(message)
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            topic = data.get('topic')
            if topic:
                await connection_manager.subscribe(websocket, topic)
                await connection_manager.send_to_connection(websocket, {
                    'type': 'subscription_confirmed',
                    'topic': topic,
                    'timestamp': datetime.now().isoformat()
                })
                
        elif message_type == 'unsubscribe':
            topic = data.get('topic')
            if topic:
                await connection_manager.unsubscribe(websocket, topic)
                await connection_manager.send_to_connection(websocket, {
                    'type': 'unsubscription_confirmed',
                    'topic': topic,
                    'timestamp': datetime.now().isoformat()
                })
                
        elif message_type == 'ping':
            await connection_manager.send_to_connection(websocket, {
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            })
            
        elif message_type == 'stats':
            stats = await connection_manager.get_stats()
            await connection_manager.send_to_connection(websocket, {
                'type': 'stats_response',
                'data': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        else:
            await connection_manager.send_to_connection(websocket, {
                'type': 'error',
                'message': f'Unknown message type: {message_type}',
                'timestamp': datetime.now().isoformat()
            })
            
    except json.JSONDecodeError:
        await connection_manager.send_to_connection(websocket, {
            'type': 'error',
            'message': 'Invalid JSON message',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await connection_manager.send_to_connection(websocket, {
            'type': 'error',
            'message': 'Message processing failed',
            'timestamp': datetime.now().isoformat()
        })

async def websocket_handler(websocket: WebSocketServerProtocol, path: str):
    """Main WebSocket connection handler"""
    await connection_manager.connect(websocket, path)
    
    try:
        async for message in websocket:
            await handle_websocket_message(websocket, message)
    except ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        await connection_manager.disconnect(websocket)

# Data streaming functions
async def stream_market_data():
    """Stream mock market data for testing"""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    
    while True:
        for symbol in symbols:
            # Generate mock market data
            base_price = {'EURUSD': 1.0945, 'GBPUSD': 1.2534, 'USDJPY': 149.83, 'AUDUSD': 0.6543}[symbol]
            
            # Add some random variation
            import random
            variation = random.uniform(-0.002, 0.002)
            bid = base_price + variation
            ask = bid + random.uniform(0.0001, 0.0005)
            
            market_update = MarketDataUpdate(
                symbol=symbol,
                bid=round(bid, 5),
                ask=round(ask, 5),
                spread=round(ask - bid, 5),
                timestamp=datetime.now(),
                volume=random.randint(100, 1000),
                change=variation,
                change_percent=round((variation / base_price) * 100, 3)
            )
            
            # Broadcast to subscribers
            await connection_manager.broadcast_to_topic(f'market_data.{symbol}', {
                'type': 'market_data_update',
                'data': asdict(market_update),
                'timestamp': datetime.now().isoformat()
            })
            
        await asyncio.sleep(1)  # Update every second

async def stream_trading_signals():
    """Stream mock trading signals for testing"""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    signals = ['buy', 'sell', 'hold']
    
    while True:
        await asyncio.sleep(30)  # Update every 30 seconds
        
        for symbol in symbols:
            import random
            
            signal_update = TradingSignalUpdate(
                symbol=symbol,
                signal=random.choice(signals),
                confidence=round(random.uniform(0.6, 0.95), 2),
                price_target=round(random.uniform(1.08, 1.12), 5) if symbol == 'EURUSD' else None,
                stop_loss=round(random.uniform(1.09, 1.095), 5) if symbol == 'EURUSD' else None,
                risk_score=round(random.uniform(0.2, 0.8), 2),
                reasoning=f"AI analysis indicates {random.choice(['bullish', 'bearish', 'neutral'])} momentum",
                timestamp=datetime.now()
            )
            
            # Broadcast to subscribers
            await connection_manager.broadcast_to_topic(f'trading_signals.{symbol}', {
                'type': 'trading_signal_update',
                'data': asdict(signal_update),
                'timestamp': datetime.now().isoformat()
            })

async def stream_system_alerts():
    """Stream system alerts and notifications"""
    alert_types = ['info', 'warning', 'error']
    sources = ['ai_engine', 'risk_manager', 'market_data', 'trading_engine']
    
    while True:
        await asyncio.sleep(60)  # Update every minute
        
        import random
        
        alert = SystemAlert(
            level=random.choice(alert_types),
            message=f"System status update from {random.choice(sources)}",
            source=random.choice(sources),
            timestamp=datetime.now(),
            data={'cpu_usage': random.uniform(20, 80), 'memory_usage': random.uniform(30, 70)}
        )
        
        # Broadcast to subscribers
        await connection_manager.broadcast_to_topic('system_alerts', {
            'type': 'system_alert',
            'data': asdict(alert),
            'timestamp': datetime.now().isoformat()
        })

async def start_websocket_server():
    """Start the WebSocket server"""
    logger.info("🚀 Starting Platform3 WebSocket Bridge Server...")
    
    # Start background tasks
    asyncio.create_task(stream_market_data())
    asyncio.create_task(stream_trading_signals())
    asyncio.create_task(stream_system_alerts())
    
    # Start WebSocket server
    server = await websockets.serve(
        websocket_handler,
        "localhost",
        8001,
        ping_interval=20,
        ping_timeout=10,
        max_size=2**20,  # 1MB max message size
        max_queue=32
    )
    
    logger.info("✅ WebSocket Bridge Server running on ws://localhost:8001")
    logger.info("📡 Real-time streaming active for market data, signals, and alerts")
    
    return server

if __name__ == "__main__":
    # Start the server
    start_server = start_websocket_server()
    
    try:
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logger.info("🛑 WebSocket Bridge Server stopped")
    except Exception as e:
        logger.error(f"Server error: {e}")