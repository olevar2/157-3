"""
Platform3 Python-TypeScript Communication Bridge
FastAPI-based REST API server for Python AI engines
Provides high-performance endpoints for TypeScript services
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import uvicorn
import logging
import json
import time
import os
import sys
from contextlib import asynccontextmanager

# Add platform paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ai_platform_manager import get_platform_manager, AIPlatformManager
from shared.platform3_module_loader import Platform3ModuleLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/python_api_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class TradingSignalRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(default="1h", description="Timeframe for analysis")
    action: Optional[str] = Field(default=None, description="Requested action (buy/sell/hold)")
    current_price: Optional[float] = Field(default=None, description="Current market price")
    account_balance: Optional[float] = Field(default=None, description="Account balance")
    risk_level: str = Field(default="medium", description="Risk tolerance level")

class TradingSignalResponse(BaseModel):
    symbol: str
    action: str  # buy, sell, hold
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_score: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime
    latency_ms: float

class MarketAnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    indicators: List[str] = Field(default=["rsi", "macd", "bollinger", "sma", "ema"])
    lookback_periods: int = Field(default=100, ge=1, le=1000)

class MarketAnalysisResponse(BaseModel):
    symbol: str
    timeframe: str
    analysis: Dict[str, Any]
    signals: Dict[str, Any]
    market_condition: str
    volatility_score: float
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    timestamp: datetime
    latency_ms: float

class RiskAssessmentRequest(BaseModel):
    symbol: str
    position_size: float
    entry_price: float
    account_balance: float
    risk_percentage: float = Field(default=2.0, ge=0.1, le=10.0)
    leverage: float = Field(default=1.0, ge=1.0, le=100.0)

class RiskAssessmentResponse(BaseModel):
    symbol: str
    risk_score: float  # 0.0 to 1.0
    position_risk: float
    account_risk: float
    max_loss: float
    recommended_stop_loss: float
    position_sizing_recommendation: float
    risk_warnings: List[str]
    approved: bool
    timestamp: datetime
    latency_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    service: str
    version: str
    python_engines_active: int
    ai_platform_health: float
    uptime_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime

# Global platform manager
platform_manager: Optional[AIPlatformManager] = None

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global platform_manager
    logger.info("🚀 Starting Platform3 Python REST API Server...")
    
    # Initialize platform manager
    platform_manager = get_platform_manager()
    
    # Load Python modules
    Platform3ModuleLoader.setup_paths()
    
    logger.info("✅ Platform3 Python REST API Server ready")
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Platform3 Python REST API Server...")
    if platform_manager:
        platform_manager.shutdown()

# Create FastAPI app
app = FastAPI(
    title="Platform3 Python-TypeScript Communication Bridge",
    description="High-performance REST API for Python AI engines to communicate with TypeScript services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Performance tracking middleware
@app.middleware("http")
async def add_performance_headers(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    return response

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint for service monitoring"""
    start_time = time.time()
    
    try:
        if not platform_manager:
            raise HTTPException(status_code=503, detail="Platform manager not initialized")
        
        # Get platform health
        health_data = platform_manager.get_platform_health()
        status = platform_manager.get_platform_status()
        
        # Calculate process metrics
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return HealthCheckResponse(
            status="healthy",
            service="platform3-python-bridge",
            version="1.0.0",
            python_engines_active=status.active_models,
            ai_platform_health=health_data["overall_health_score"],
            uptime_seconds=health_data["uptime_seconds"],
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# Trading signals endpoint
@app.post("/api/v1/trading/signals", response_model=TradingSignalResponse)
async def get_trading_signals(request: TradingSignalRequest):
    """Generate AI-powered trading signals for a given symbol"""
    start_time = time.time()
    
    try:
        if not platform_manager:
            raise HTTPException(status_code=503, detail="Platform manager not initialized")
        
        logger.info(f"Generating trading signals for {request.symbol}")
        
        # Load Decision Master for trading signals
        decision_master = Platform3ModuleLoader.load_decision_master()
        if not decision_master:
            raise HTTPException(status_code=503, detail="Decision Master not available")
        
        # Prepare market data (mock implementation - replace with real data)
        market_data = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "current_price": request.current_price or 1.0950,  # Mock price
            "risk_level": request.risk_level,
            "account_balance": request.account_balance or 10000.0
        }
        
        # Get AI trading signal
        ai_signal = decision_master.DecisionMaster().generate_trading_signal(market_data)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TradingSignalResponse(
            symbol=request.symbol,
            action=ai_signal.get("action", "hold"),
            confidence=ai_signal.get("confidence", 0.5),
            price_target=ai_signal.get("price_target"),
            stop_loss=ai_signal.get("stop_loss"),
            take_profit=ai_signal.get("take_profit"),
            risk_score=ai_signal.get("risk_score", 0.3),
            reasoning=ai_signal.get("reasoning", "AI analysis"),
            timestamp=datetime.now(),
            latency_ms=latency_ms
        )
    
    except Exception as e:
        logger.error(f"Trading signal generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")

# Market analysis endpoint
@app.post("/api/v1/analysis/market", response_model=MarketAnalysisResponse)
async def analyze_market(request: MarketAnalysisRequest):
    """Perform comprehensive market analysis using Python AI engines"""
    start_time = time.time()
    
    try:
        if not platform_manager:
            raise HTTPException(status_code=503, detail="Platform manager not initialized")
        
        logger.info(f"Analyzing market for {request.symbol}")
        
        # Mock comprehensive market analysis (replace with real AI engine calls)
        analysis_data = {
            "rsi": {"value": 65.4, "signal": "overbought"},
            "macd": {"value": 0.0023, "signal": "bullish"},
            "bollinger": {"upper": 1.0985, "middle": 1.0945, "lower": 1.0905},
            "sma_20": 1.0943,
            "ema_12": 1.0948,
            "volume_profile": {"support": [1.0920, 1.0935], "resistance": [1.0970, 1.0985]}
        }
        
        # Calculate market condition
        market_condition = "trending_up"
        volatility_score = 0.65
        trend_strength = 0.72
        
        latency_ms = (time.time() - start_time) * 1000
        
        return MarketAnalysisResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            analysis=analysis_data,
            signals={"overall": "bullish", "strength": "moderate"},
            market_condition=market_condition,
            volatility_score=volatility_score,
            trend_strength=trend_strength,
            support_levels=[1.0920, 1.0935, 1.0943],
            resistance_levels=[1.0970, 1.0985, 1.0995],
            timestamp=datetime.now(),
            latency_ms=latency_ms
        )
    
    except Exception as e:
        logger.error(f"Market analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Risk assessment endpoint
@app.post("/api/v1/risk/assess", response_model=RiskAssessmentResponse)
async def assess_risk(request: RiskAssessmentRequest):
    """Assess trading risk using Python risk management engines"""
    start_time = time.time()
    
    try:
        if not platform_manager:
            raise HTTPException(status_code=503, detail="Platform manager not initialized")
        
        logger.info(f"Assessing risk for {request.symbol}")
        
        # Calculate risk metrics
        position_value = request.position_size * request.entry_price
        position_risk = (request.risk_percentage / 100) * request.account_balance
        account_risk = position_value / request.account_balance
        
        # Calculate stop loss recommendation
        risk_per_pip = position_risk / request.position_size
        recommended_stop_pips = 20  # Mock calculation
        recommended_stop_loss = request.entry_price - (recommended_stop_pips * 0.0001)
        
        # Risk score calculation
        risk_score = min(account_risk * 2, 1.0)
        
        # Risk warnings
        warnings = []
        if account_risk > 0.05:
            warnings.append("Position size exceeds 5% of account")
        if risk_score > 0.7:
            warnings.append("High risk trade - consider reducing position size")
        
        # Approval logic
        approved = risk_score < 0.8 and len(warnings) < 2
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RiskAssessmentResponse(
            symbol=request.symbol,
            risk_score=risk_score,
            position_risk=position_risk,
            account_risk=account_risk,
            max_loss=position_risk,
            recommended_stop_loss=recommended_stop_loss,
            position_sizing_recommendation=request.position_size * 0.8 if risk_score > 0.6 else request.position_size,
            risk_warnings=warnings,
            approved=approved,
            timestamp=datetime.now(),
            latency_ms=latency_ms
        )
    
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

# Platform status endpoint
@app.get("/api/v1/platform/status")
async def get_platform_status():
    """Get comprehensive Platform3 AI platform status"""
    try:
        if not platform_manager:
            raise HTTPException(status_code=503, detail="Platform manager not initialized")
        
        status = platform_manager.get_platform_status()
        health = platform_manager.get_platform_health()
        summary = platform_manager.get_platform_summary()
        
        return {
            "status": status.__dict__,
            "health": health,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Platform status request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status request failed: {str(e)}")

# AI model execution endpoint
@app.post("/api/v1/models/execute")
async def execute_model(
    model_id: str,
    function_name: str = "predict",
    parameters: Dict[str, Any] = None,
    timeout: Optional[int] = 30
):
    """Execute AI model with specified parameters"""
    try:
        if not platform_manager:
            raise HTTPException(status_code=503, detail="Platform manager not initialized")
        
        if parameters is None:
            parameters = {}
        
        # Execute model
        task_id = platform_manager.execute_prediction(
            model_id=model_id,
            function_name=function_name,
            parameters=parameters,
            timeout=timeout
        )
        
        return {
            "task_id": task_id,
            "model_id": model_id,
            "status": "submitted",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Model execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model execution failed: {str(e)}")

# Real-time data streaming endpoint (WebSocket would be ideal, but HTTP for now)
@app.get("/api/v1/stream/market/{symbol}")
async def stream_market_data(symbol: str, background_tasks: BackgroundTasks):
    """Get real-time market data stream (polling-based for HTTP compatibility)"""
    try:
        # Mock real-time data (replace with actual market data streaming)
        market_data = {
            "symbol": symbol,
            "bid": 1.0943,
            "ask": 1.0945,
            "spread": 0.0002,
            "change": 0.0012,
            "change_percent": 0.11,
            "volume": 125678,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "data": market_data,
            "next_update_seconds": 1,
            "stream_id": f"stream_{symbol}_{int(time.time())}"
        }
    
    except Exception as e:
        logger.error(f"Market data streaming failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Main server startup
if __name__ == "__main__":
    logger.info("🚀 Starting Platform3 Python REST API Bridge...")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Start server with optimal settings for trading performance
    uvicorn.run(
        "rest_api_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for consistency
        loop="uvloop",  # High-performance event loop
        http="httptools",  # High-performance HTTP parser
        access_log=True,
        reload=False  # Disable in production
    )