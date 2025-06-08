"""
Real-time Analytics WebSocket Server
Provides WebSocket endpoints for streaming analytics data to the dashboard
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol
import aioredis
from AdvancedAnalyticsFramework import AdvancedAnalyticsFramework, AnalyticsEvent

logger = logging.getLogger(__name__)

class AnalyticsWebSocketServer:
    """WebSocket server for real-time analytics data streaming"""
    
    def __init__(self, host: str = "localhost", port: int = 8001, redis_url: str = "redis://localhost:6379"):
        self.host = host
        self.port = port
        self.redis_url = redis_url
        
        # Connected clients
        self.clients: Set[WebSocketServerProtocol] = set()
        
        # Analytics framework
        self.analytics_framework: Optional[AdvancedAnalyticsFramework] = None
        
        # Redis client for pub/sub
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Last known state
        self.last_metrics = {}
        self.last_engines_status = []
        
        logger.info(f"Analytics WebSocket Server initialized on {host}:{port}")

    async def initialize(self):
        """Initialize the WebSocket server and dependencies"""
        try:
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(self.redis_url)
            
            # Initialize analytics framework
            self.analytics_framework = AdvancedAnalyticsFramework(self.redis_url)
            await self.analytics_framework.initialize()
            
            # Subscribe to analytics events
            self._setup_event_listeners()
            
            # Start Redis subscription task
            asyncio.create_task(self._redis_subscription_task())
            
            logger.info("Analytics WebSocket Server fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket server: {e}")
            raise

    def _setup_event_listeners(self):
        """Setup event listeners for analytics framework events"""
        if self.analytics_framework:
            # Listen for analytics data events
            self.analytics_framework.subscribe_to_events(
                "analytics_data", 
                self._handle_analytics_data_event
            )
            
            # Listen for report generation events
            self.analytics_framework.subscribe_to_events(
                "report_generated", 
                self._handle_report_generated_event
            )

    async def _handle_analytics_data_event(self, event: AnalyticsEvent):
        """Handle analytics data events from the framework"""
        try:
            # Update last known state
            if 'metrics' in event.data:
                self.last_metrics = event.data['metrics']
            
            # Broadcast to all connected clients
            message = {
                "type": "analytics_data",
                "timestamp": event.timestamp.isoformat(),
                "source": event.source,
                "data": event.data
            }
            
            await self._broadcast_message(message)
            
        except Exception as e:
            logger.error(f"Error handling analytics data event: {e}")

    async def _handle_report_generated_event(self, report):
        """Handle report generation events"""
        try:
            message = {
                "type": "report_generated",
                "timestamp": datetime.utcnow().isoformat(),
                "report": {
                    "report_id": report.report_id,
                    "report_type": report.report_type,
                    "generated_at": report.generated_at.isoformat(),
                    "summary": report.summary,
                    "recommendations": report.recommendations,
                    "confidence_score": report.confidence_score
                }
            }
            
            await self._broadcast_message(message)
            
        except Exception as e:
            logger.error(f"Error handling report generated event: {e}")

    async def _redis_subscription_task(self):
        """Background task to listen for Redis pub/sub messages"""
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("analytics_stream", "metrics_stream")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        channel = message["channel"].decode()
                        
                        if channel == "analytics_stream":
                            await self._handle_analytics_stream_message(data)
                        elif channel == "metrics_stream":
                            await self._handle_metrics_stream_message(data)
                            
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
                        
        except Exception as e:
            logger.error(f"Redis subscription task error: {e}")

    async def _handle_analytics_stream_message(self, data: Dict[str, Any]):
        """Handle analytics stream messages from Redis"""
        message = {
            "type": "analytics_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        await self._broadcast_message(message)

    async def _handle_metrics_stream_message(self, data: Dict[str, Any]):
        """Handle metrics stream messages from Redis"""
        self.last_metrics = data
        
        # Update engines status based on metrics
        await self._update_engines_status()
        
        message = {
            "type": "metrics_update",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": data
        }
        await self._broadcast_message(message)

    async def _update_engines_status(self):
        """Update engines status based on current metrics"""
        try:
            engines_status = []
            
            # Extract engine information from metrics
            engine_names = set()
            for metric_name in self.last_metrics.keys():
                if '_' in metric_name:
                    engine_name = metric_name.split('_')[0]
                    engine_names.add(engine_name)
            
            for engine_name in engine_names:
                # Find performance metric for this engine
                performance_key = f"{engine_name}_performance"
                performance_score = 0.0
                processed_items = 0
                status = "inactive"
                
                if performance_key in self.last_metrics:
                    metric = self.last_metrics[performance_key]
                    performance_score = metric.get('value', 0.0)
                    status = "active" if performance_score > 0 else "inactive"
                
                # Count processed items (if available)
                processed_key = f"{engine_name}_processed"
                if processed_key in self.last_metrics:
                    processed_items = self.last_metrics[processed_key].get('value', 0)
                
                engines_status.append({
                    "name": engine_name.replace('_', ' ').title(),
                    "status": status,
                    "performance_score": performance_score,
                    "processed_items": processed_items,
                    "last_update": datetime.utcnow().isoformat()
                })
            
            self.last_engines_status = engines_status
            
            # Broadcast engines status update
            message = {
                "type": "engines_status",
                "timestamp": datetime.utcnow().isoformat(),
                "engines": engines_status
            }
            await self._broadcast_message(message)
            
        except Exception as e:
            logger.error(f"Error updating engines status: {e}")

    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected_clients

    async def handle_client_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new client connections"""
        try:
            # Add client to connected clients
            self.clients.add(websocket)
            logger.info(f"New client connected from {websocket.remote_address}. Total clients: {len(self.clients)}")
            
            # Send initial state to new client
            await self._send_initial_state(websocket)
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {websocket.remote_address} disconnected")
        except Exception as e:
            logger.error(f"Error handling client connection: {e}")
        finally:
            # Remove client from connected clients
            self.clients.discard(websocket)
            logger.info(f"Client removed. Total clients: {len(self.clients)}")

    async def _send_initial_state(self, websocket: WebSocketServerProtocol):
        """Send initial state to newly connected client"""
        try:
            # Send current metrics
            if self.last_metrics:
                await websocket.send(json.dumps({
                    "type": "metrics_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": self.last_metrics
                }))
            
            # Send engines status
            if self.last_engines_status:
                await websocket.send(json.dumps({
                    "type": "engines_status",
                    "timestamp": datetime.utcnow().isoformat(),
                    "engines": self.last_engines_status
                }))
            
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connection_status",
                "message": "Connected to Advanced Analytics Framework",
                "timestamp": datetime.utcnow().isoformat()
            }))
            
        except Exception as e:
            logger.error(f"Error sending initial state: {e}")

    async def _handle_client_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle messages from clients"""
        action = data.get("action")
        
        if action == "refresh_metrics":
            await self._handle_refresh_metrics(websocket)
        elif action == "refresh_all":
            await self._handle_refresh_all(websocket)
        elif action == "generate_report":
            await self._handle_generate_report(websocket, data)
        elif action == "download_report":
            await self._handle_download_report(websocket, data)
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown action: {action}"
            }))

    async def _handle_refresh_metrics(self, websocket: WebSocketServerProtocol):
        """Handle refresh metrics request"""
        try:
            if self.analytics_framework:
                metrics = self.analytics_framework.get_realtime_metrics()
                metrics_dict = {k: {
                    "metric_name": v.metric_name,
                    "value": v.value,
                    "timestamp": v.timestamp.isoformat(),
                    "context": v.context
                } for k, v in metrics.items()}
                
                await websocket.send(json.dumps({
                    "type": "metrics_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": metrics_dict
                }))
        except Exception as e:
            logger.error(f"Error refreshing metrics: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Failed to refresh metrics"
            }))

    async def _handle_refresh_all(self, websocket: WebSocketServerProtocol):
        """Handle refresh all request"""
        try:
            # Refresh metrics
            await self._handle_refresh_metrics(websocket)
            
            # Update engines status
            await self._update_engines_status()
            
            await websocket.send(json.dumps({
                "type": "refresh_complete",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "All data refreshed successfully"
            }))
        except Exception as e:
            logger.error(f"Error refreshing all data: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Failed to refresh data"
            }))

    async def _handle_generate_report(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle generate report request"""
        try:
            timeframe = data.get("timeframe", "1h")
            
            if self.analytics_framework:
                report = await self.analytics_framework.generate_comprehensive_report(timeframe)
                
                await websocket.send(json.dumps({
                    "type": "report_generated",
                    "timestamp": datetime.utcnow().isoformat(),
                    "report": {
                        "report_id": report.report_id,
                        "report_type": report.report_type,
                        "generated_at": report.generated_at.isoformat(),
                        "summary": report.summary,
                        "recommendations": report.recommendations,
                        "confidence_score": report.confidence_score
                    }
                }))
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Failed to generate report"
            }))

    async def _handle_download_report(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle download report request"""
        try:
            report_id = data.get("report_id")
            
            if not report_id:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Report ID is required"
                }))
                return
            
            # In a real implementation, you would retrieve the report from storage
            # For now, we'll send a placeholder response
            await websocket.send(json.dumps({
                "type": "download_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "report_id": report_id,
                "download_url": f"/api/reports/{report_id}/download",
                "message": "Report ready for download"
            }))
            
        except Exception as e:
            logger.error(f"Error handling download request: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Failed to prepare report download"
            }))

    async def start_server(self):
        """Start the WebSocket server"""
        try:
            await self.initialize()
            
            server = await websockets.serve(
                self.handle_client_connection,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            )
            
            logger.info(f"Analytics WebSocket Server started on ws://{self.host}:{self.port}")
            print(f"🚀 Analytics WebSocket Server running on ws://{self.host}:{self.port}")
            
            # Keep the server running
            await server.wait_closed()
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def shutdown(self):
        """Gracefully shutdown the server"""
        try:
            # Close all client connections
            if self.clients:
                await asyncio.gather(
                    *[client.close() for client in self.clients],
                    return_exceptions=True
                )
            
            # Shutdown analytics framework
            if self.analytics_framework:
                await self.analytics_framework.shutdown()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Analytics WebSocket Server shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during server shutdown: {e}")

# Main execution
async def main():
    """Main function to run the WebSocket server"""
    server = AnalyticsWebSocketServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await server.shutdown()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
