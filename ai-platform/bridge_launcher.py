#!/usr/bin/env python3
"""
Platform3 Python Communication Bridge Launcher
Starts both REST API and WebSocket servers for Python-TypeScript communication
"""

import asyncio
import multiprocessing
import logging
import signal
import sys
import os
from datetime import datetime

# Add platform paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/bridge_launcher.log')
    ]
)
logger = logging.getLogger(__name__)

def run_rest_api_server():
    """Run the REST API server in a separate process"""
    try:
        import uvicorn
        from rest_api_server import app
        
        logger.info("🚀 Starting REST API Server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=1,
            loop="uvloop",
            http="httptools",
            access_log=True,
            reload=False
        )
    except Exception as e:
        logger.error(f"REST API Server failed: {e}")
        raise

def run_websocket_server():
    """Run the WebSocket server in a separate process"""
    try:
        from websocket_server import start_websocket_server
        
        logger.info("📡 Starting WebSocket Server...")
        
        # Run the WebSocket server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        server = loop.run_until_complete(start_websocket_server())
        loop.run_forever()
        
    except Exception as e:
        logger.error(f"WebSocket Server failed: {e}")
        raise

class BridgeLauncher:
    """Launcher for Platform3 Python-TypeScript Communication Bridge"""
    
    def __init__(self):
        self.rest_api_process = None
        self.websocket_process = None
        self.running = False
        
    def start(self):
        """Start both servers"""
        logger.info("🌉 Starting Platform3 Python-TypeScript Communication Bridge...")
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        try:
            # Start REST API server
            self.rest_api_process = multiprocessing.Process(
                target=run_rest_api_server,
                name="RestAPIServer"
            )
            self.rest_api_process.start()
            logger.info(f"✅ REST API Server started (PID: {self.rest_api_process.pid})")
            
            # Start WebSocket server
            self.websocket_process = multiprocessing.Process(
                target=run_websocket_server,
                name="WebSocketServer"
            )
            self.websocket_process.start()
            logger.info(f"✅ WebSocket Server started (PID: {self.websocket_process.pid})")
            
            self.running = True
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            logger.info("🔗 Communication Bridge fully operational!")
            logger.info("   REST API: http://localhost:8000")
            logger.info("   WebSocket: ws://localhost:8001")
            logger.info("   API Docs: http://localhost:8000/docs")
            
            # Monitor processes
            self._monitor_processes()
            
        except Exception as e:
            logger.error(f"Failed to start communication bridge: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop both servers"""
        logger.info("🛑 Stopping Platform3 Communication Bridge...")
        
        self.running = False
        
        if self.rest_api_process and self.rest_api_process.is_alive():
            logger.info("Stopping REST API Server...")
            self.rest_api_process.terminate()
            self.rest_api_process.join(timeout=10)
            if self.rest_api_process.is_alive():
                logger.warning("Force killing REST API Server...")
                self.rest_api_process.kill()
        
        if self.websocket_process and self.websocket_process.is_alive():
            logger.info("Stopping WebSocket Server...")
            self.websocket_process.terminate()
            self.websocket_process.join(timeout=10)
            if self.websocket_process.is_alive():
                logger.warning("Force killing WebSocket Server...")
                self.websocket_process.kill()
        
        logger.info("✅ Communication Bridge stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _monitor_processes(self):
        """Monitor both processes and restart if needed"""
        import time
        
        while self.running:
            try:
                # Check REST API process
                if self.rest_api_process and not self.rest_api_process.is_alive():
                    logger.error("REST API Server died, restarting...")
                    self.rest_api_process = multiprocessing.Process(
                        target=run_rest_api_server,
                        name="RestAPIServer"
                    )
                    self.rest_api_process.start()
                    logger.info(f"✅ REST API Server restarted (PID: {self.rest_api_process.pid})")
                
                # Check WebSocket process
                if self.websocket_process and not self.websocket_process.is_alive():
                    logger.error("WebSocket Server died, restarting...")
                    self.websocket_process = multiprocessing.Process(
                        target=run_websocket_server,
                        name="WebSocketServer"
                    )
                    self.websocket_process.start()
                    logger.info(f"✅ WebSocket Server restarted (PID: {self.websocket_process.pid})")
                
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
                time.sleep(5)

def main():
    """Main entry point"""
    launcher = BridgeLauncher()
    
    try:
        launcher.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Launcher failed: {e}")
        sys.exit(1)
    finally:
        launcher.stop()

if __name__ == "__main__":
    main()