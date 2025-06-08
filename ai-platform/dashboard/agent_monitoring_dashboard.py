"""
🏥 HUMANITARIAN AI PLATFORM - AGENT MONITORING WEB DASHBOARD
💝 Real-time web interface for agent collaboration monitoring

This module provides a comprehensive web dashboard for monitoring agent collaboration,
communication performance, and coordination effectiveness in real-time.

Features:
- Real-time metrics display with auto-refresh
- Interactive agent communication matrix
- Performance trend charts and graphs
- Alert management and notification center
- Humanitarian mission impact tracking
- Responsive design for desktop and mobile
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
import socketio
import weakref

# Import the collaboration monitor
from .AgentCollaborationMonitor import (
    AgentCollaborationMonitor, get_collaboration_monitor,
    MetricType, AlertLevel, CollaborationMetric
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentMonitoringDashboard:
    """
    🏥 Agent Monitoring Web Dashboard for Humanitarian AI Platform
    
    Provides comprehensive web-based monitoring interface:
    - Real-time agent collaboration metrics
    - Interactive performance visualization
    - Alert management and notifications
    - Humanitarian mission impact tracking
    - WebSocket-based live updates
    - RESTful API for external integrations
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080, collaboration_monitor: AgentCollaborationMonitor = None):
        self.host = host
        self.port = port
        self.collaboration_monitor = collaboration_monitor or get_collaboration_monitor()
        
        # Web server components
        self.app = web.Application()
        self.sio = socketio.AsyncServer(cors_allowed_origins="*")
        self.sio.attach(self.app)
        
        # Dashboard state
        self.connected_clients = set()
        self.dashboard_active = False
        self.update_interval = 5  # seconds
        self.update_task = None
        
        # Setup routes and handlers
        self._setup_routes()
        self._setup_websocket_handlers()
        self._setup_cors()
        
        logger.info("🏥 Agent Monitoring Dashboard initialized")
        logger.info(f"📊 Dashboard will be available at http://{host}:{port}")
    
    def _setup_routes(self):
        """Setup HTTP routes for the dashboard"""
        
        # API Routes
        self.app.router.add_get('/api/dashboard', self.get_dashboard_data)
        self.app.router.add_get('/api/agents', self.get_agents_summary)
        self.app.router.add_get('/api/agents/{agent_id}', self.get_agent_detail)
        self.app.router.add_get('/api/metrics/{metric_type}', self.get_metrics_data)
        self.app.router.add_get('/api/communication-matrix', self.get_communication_matrix)
        self.app.router.add_get('/api/alerts', self.get_alerts)
        self.app.router.add_post('/api/alerts/{alert_id}/resolve', self.resolve_alert)
        self.app.router.add_get('/api/health', self.health_check)
        
        # Static files and main dashboard page
        self.app.router.add_get('/', self.serve_dashboard)
        self.app.router.add_static('/static/', path='static/', name='static')
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers for real-time updates"""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            self.connected_clients.add(sid)
            logger.info(f"📡 Dashboard client connected: {sid}")
            
            # Send initial data to new client
            dashboard_data = self.collaboration_monitor.get_dashboard_data()
            await self.sio.emit('dashboard_update', dashboard_data, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            self.connected_clients.discard(sid)
            logger.info(f"📡 Dashboard client disconnected: {sid}")
        
        @self.sio.event
        async def subscribe_to_agent(sid, data):
            """Subscribe to specific agent updates"""
            agent_id = data.get('agent_id')
            if agent_id:
                await self.sio.enter_room(sid, f"agent_{agent_id}")
                logger.info(f"📡 Client {sid} subscribed to agent {agent_id}")
    
    def _setup_cors(self):
        """Setup CORS for cross-origin requests"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def get_dashboard_data(self, request):
        """API endpoint: Get complete dashboard data"""
        try:
            dashboard_data = self.collaboration_monitor.get_dashboard_data()
            return web.json_response(dashboard_data)
        except Exception as e:
            logger.error(f"❌ Error getting dashboard data: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_agents_summary(self, request):
        """API endpoint: Get summary of all agents"""
        try:
            dashboard_data = self.collaboration_monitor.get_dashboard_data()
            return web.json_response(dashboard_data['agent_summaries'])
        except Exception as e:
            logger.error(f"❌ Error getting agents summary: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_agent_detail(self, request):
        """API endpoint: Get detailed information for specific agent"""
        try:
            agent_id = request.match_info['agent_id']
            agent_summary = self.collaboration_monitor.get_agent_collaboration_summary(agent_id)
            return web.json_response({
                "agent_id": agent_id,
                "summary": agent_summary.__dict__,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"❌ Error getting agent detail: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_metrics_data(self, request):
        """API endpoint: Get specific metric type data"""
        try:
            metric_type_str = request.match_info['metric_type']
            
            # Validate metric type
            try:
                metric_type = MetricType(metric_type_str)
            except ValueError:
                return web.json_response({"error": f"Invalid metric type: {metric_type_str}"}, status=400)
            
            # Get query parameters
            limit = int(request.query.get('limit', 100))
            hours = int(request.query.get('hours', 1))
            
            # Get metrics from monitor
            dashboard_data = self.collaboration_monitor.get_dashboard_data()
            metrics_data = dashboard_data['recent_metrics'].get(metric_type_str, [])
            
            # Filter by time if specified
            if hours:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                metrics_data = [
                    metric for metric in metrics_data
                    if datetime.fromisoformat(metric['timestamp']) >= cutoff_time
                ]
            
            # Limit results
            metrics_data = metrics_data[-limit:] if limit else metrics_data
            
            return web.json_response({
                "metric_type": metric_type_str,
                "data": metrics_data,
                "count": len(metrics_data),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"❌ Error getting metrics data: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_communication_matrix(self, request):
        """API endpoint: Get agent communication matrix"""
        try:
            dashboard_data = self.collaboration_monitor.get_dashboard_data()
            return web.json_response(dashboard_data['communication_matrix'])
        except Exception as e:
            logger.error(f"❌ Error getting communication matrix: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_alerts(self, request):
        """API endpoint: Get current alerts"""
        try:
            # This would need to be implemented in the collaboration monitor
            # For now, return placeholder data
            return web.json_response({
                "active_alerts": [],
                "alert_count": 0,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"❌ Error getting alerts: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def resolve_alert(self, request):
        """API endpoint: Resolve an alert"""
        try:
            alert_id = request.match_info['alert_id']
            # Implementation would depend on alert management in collaboration monitor
            return web.json_response({
                "success": True,
                "alert_id": alert_id,
                "resolved_at": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"❌ Error resolving alert: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def health_check(self, request):
        """API endpoint: Health check"""
        try:
            return web.json_response({
                "status": "healthy",
                "dashboard_active": self.dashboard_active,
                "connected_clients": len(self.connected_clients),
                "monitoring_active": self.collaboration_monitor.monitoring_active,
                "timestamp": datetime.now().isoformat(),
                "humanitarian_mission": "active"
            })
        except Exception as e:
            logger.error(f"❌ Error in health check: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def serve_dashboard(self, request):
        """Serve the main dashboard HTML page"""
        # In a production environment, this would serve a proper HTML file
        # For now, return a simple HTML page with embedded dashboard
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 Humanitarian AI Platform - Agent Monitoring Dashboard</title>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .mission {{
            margin: 10px 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            padding: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            border-left: 5px solid #667eea;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        .metric-unit {{
            font-size: 0.8em;
            color: #999;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .status-healthy {{ background: #28a745; }}
        .status-warning {{ background: #ffc107; }}
        .status-critical {{ background: #dc3545; }}
        .agents-section {{
            padding: 30px;
            border-top: 1px solid #eee;
        }}
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
        }}
        .agent-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .agent-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #dee2e6;
        }}
        .agent-name {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .agent-stat {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 0.9em;
        }}
        .loading {{
            text-align: center;
            padding: 50px;
            color: #666;
        }}
        .last-updated {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Humanitarian AI Platform</h1>
            <div class="mission">💝 Agent Collaboration Monitoring Dashboard</div>
            <div class="mission">Ensuring optimal coordination for medical aid, children's surgeries, and poverty relief</div>
        </div>
        
        <div id="loading" class="loading">
            <div>🔄 Loading dashboard data...</div>
        </div>
        
        <div id="dashboard" style="display: none;">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Active Agents</div>
                    <div class="metric-value" id="active-agents">-</div>
                    <div class="metric-unit">agents online</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Messages Per Second</div>
                    <div class="metric-value" id="messages-per-second">-</div>
                    <div class="metric-unit">msg/sec</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Avg Coordination Latency</div>
                    <div class="metric-value" id="avg-latency">-</div>
                    <div class="metric-unit">milliseconds</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Error Rate</div>
                    <div class="metric-value" id="error-rate">-</div>
                    <div class="metric-unit">percent</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Coordination Success</div>
                    <div class="metric-value" id="coordination-success">-</div>
                    <div class="metric-unit">percent</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Humanitarian Impact</div>
                    <div class="metric-value" id="humanitarian-impact">-</div>
                    <div class="metric-unit">impact score</div>
                </div>
            </div>
            
            <div class="agents-section">
                <h2 class="section-title">👥 Agent Status</h2>
                <div id="agents-grid" class="agent-grid">
                    <!-- Agent cards will be populated here -->
                </div>
            </div>
        </div>
        
        <div class="last-updated">
            Last updated: <span id="last-updated">-</span>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        socket.on('connect', function() {{
            console.log('🔗 Connected to dashboard server');
        }});
        
        socket.on('dashboard_update', function(data) {{
            console.log('📊 Dashboard data received', data);
            updateDashboard(data);
        }});
        
        socket.on('disconnect', function() {{
            console.log('🔌 Disconnected from dashboard server');
        }});
        
        function updateDashboard(data) {{
            // Hide loading, show dashboard
            document.getElementById('loading').style.display = 'none';
            document.getElementById('dashboard').style.display = 'block';
            
            // Update system metrics
            const metrics = data.system_metrics;
            document.getElementById('active-agents').textContent = metrics.total_active_agents;
            document.getElementById('messages-per-second').textContent = metrics.total_messages_per_second.toFixed(2);
            document.getElementById('avg-latency').textContent = metrics.average_coordination_latency_ms.toFixed(1);
            document.getElementById('error-rate').textContent = metrics.system_error_rate_percent.toFixed(1);
            document.getElementById('coordination-success').textContent = metrics.coordination_success_rate_percent.toFixed(1);
            document.getElementById('humanitarian-impact').textContent = metrics.humanitarian_impact_score.toFixed(1);
            
            // Update agents grid
            updateAgentsGrid(data.agent_summaries);
            
            // Update last updated time
            document.getElementById('last-updated').textContent = new Date(data.timestamp).toLocaleString();
        }}
        
        function updateAgentsGrid(agents) {{
            const grid = document.getElementById('agents-grid');
            grid.innerHTML = '';
            
            Object.entries(agents).forEach(([agentId, summary]) => {{
                const statusClass = summary.status === 'active' ? 'status-healthy' : 
                                   summary.status === 'degraded' ? 'status-warning' : 'status-critical';
                
                const card = document.createElement('div');
                card.className = 'agent-card';
                card.innerHTML = `
                    <div class="agent-name">
                        <span class="status-indicator ${{statusClass}}"></span>
                        ${{agentId}}
                    </div>
                    <div class="agent-stat">
                        <span>Messages Sent:</span>
                        <span>${{summary.total_messages_sent}}</span>
                    </div>
                    <div class="agent-stat">
                        <span>Avg Response:</span>
                        <span>${{summary.average_response_time_ms.toFixed(1)}}ms</span>
                    </div>
                    <div class="agent-stat">
                        <span>Error Rate:</span>
                        <span>${{summary.error_rate_percent.toFixed(1)}}%</span>
                    </div>
                    <div class="agent-stat">
                        <span>Coordination Score:</span>
                        <span>${{summary.coordination_score.toFixed(1)}}</span>
                    </div>
                `;
                grid.appendChild(card);
            }});
        }}
        
        // Initial data load
        fetch('/api/dashboard')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(error => console.error('❌ Error loading dashboard:', error));
        
        // Periodic refresh as backup
        setInterval(() => {{
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('❌ Error refreshing dashboard:', error));
        }}, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
        """
        
        return web.Response(text=html_content, content_type='text/html')
    
    async def start_dashboard(self):
        """Start the web dashboard server"""
        if self.dashboard_active:
            logger.warning("⚠️ Dashboard server already active")
            return
        
        self.dashboard_active = True
        
        # Start the collaboration monitor if not already running
        if not self.collaboration_monitor.monitoring_active:
            await self.collaboration_monitor.start_monitoring()
        
        # Start periodic update broadcasts
        self.update_task = asyncio.create_task(self._broadcast_updates())
        
        # Setup alert callback
        self.collaboration_monitor.add_alert_callback(self._handle_alert)
        
        logger.info(f"🚀 Dashboard server starting on http://{self.host}:{self.port}")
        logger.info("💝 Serving humanitarian AI platform monitoring dashboard")
        
        # Start the web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"✅ Dashboard server running on http://{self.host}:{self.port}")
    
    async def stop_dashboard(self):
        """Stop the web dashboard server"""
        if not self.dashboard_active:
            return
        
        self.dashboard_active = False
        
        # Stop update broadcasts
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all clients
        for client_id in list(self.connected_clients):
            await self.sio.disconnect(client_id)
        
        logger.info("🛑 Dashboard server stopped")
    
    async def _broadcast_updates(self):
        """Periodically broadcast dashboard updates to connected clients"""
        while self.dashboard_active:
            try:
                if self.connected_clients:
                    # Get latest dashboard data
                    dashboard_data = self.collaboration_monitor.get_dashboard_data()
                    
                    # Broadcast to all connected clients
                    await self.sio.emit('dashboard_update', dashboard_data)
                    
                    logger.debug(f"📡 Broadcasted update to {len(self.connected_clients)} clients")
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error broadcasting updates: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _handle_alert(self, alert_data: Dict[str, Any]):
        """Handle alerts from collaboration monitor"""
        try:
            # Broadcast alert to all connected clients
            await self.sio.emit('alert', alert_data)
            logger.info(f"🚨 Alert broadcasted to dashboard clients: {alert_data['title']}")
        except Exception as e:
            logger.error(f"❌ Error broadcasting alert: {e}")

# Global dashboard instance
dashboard_instance = None

def get_dashboard(host: str = "localhost", port: int = 8080) -> AgentMonitoringDashboard:
    """Get or create global dashboard instance"""
    global dashboard_instance
    
    if dashboard_instance is None:
        dashboard_instance = AgentMonitoringDashboard(host, port)
    
    return dashboard_instance

# Example usage and testing
if __name__ == "__main__":
    async def test_dashboard():
        print("🏥 Testing Agent Monitoring Dashboard")
        print("💝 Starting web interface for humanitarian mission monitoring")
        
        # Initialize dashboard
        dashboard = AgentMonitoringDashboard(port=8080)
        
        try:
            # Start dashboard server
            await dashboard.start_dashboard()
            
            print(f"\n🌐 Dashboard available at: http://localhost:8080")
            print("💝 Monitoring agent coordination for humanitarian success")
            print("🔄 Dashboard will auto-refresh with real-time data")
            print("\nPress Ctrl+C to stop the dashboard")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            await dashboard.stop_dashboard()
            print("✅ Dashboard stopped")
    
    # Run test
    asyncio.run(test_dashboard())