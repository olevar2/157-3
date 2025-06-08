"""
Analytics API Integration Layer
RESTful API endpoints for analytics framework integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import json
from pathlib import Path
import uvicorn

from AdvancedAnalyticsFramework import AdvancedAnalyticsFramework, AnalyticsReport, RealtimeMetric

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AnalyticsDataRequest(BaseModel):
    """Request model for streaming analytics data"""
    data: Dict[str, Any]
    source: str = "api"
    metadata: Optional[Dict[str, Any]] = None

class ReportGenerationRequest(BaseModel):
    """Request model for report generation"""
    timeframe: str = Field(default="24h", description="Time frame for the report (1h, 6h, 24h, 7d)")
    report_type: str = Field(default="comprehensive", description="Type of report to generate")
    include_recommendations: bool = Field(default=True, description="Whether to include recommendations")

class ScheduledReportRequest(BaseModel):
    """Request model for scheduling reports"""
    report_id: str
    timeframe: str = "24h"
    interval_hours: int = 24
    enabled: bool = True

class MetricResponse(BaseModel):
    """Response model for real-time metrics"""
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any]
    alert_threshold: Optional[float] = None

class EngineStatusResponse(BaseModel):
    """Response model for analytics engine status"""
    name: str
    status: str
    performance_score: float
    processed_items: int
    last_update: datetime

class ReportResponse(BaseModel):
    """Response model for analytics reports"""
    report_id: str
    report_type: str
    generated_at: datetime
    summary: str
    recommendations: List[str]
    confidence_score: float
    data_preview: Optional[Dict[str, Any]] = None

class AnalyticsStatsResponse(BaseModel):
    """Response model for analytics statistics"""
    total_engines: int
    active_engines: int
    total_metrics: int
    reports_generated: int
    uptime_hours: float
    last_update: datetime

# Initialize FastAPI app
app = FastAPI(
    title="Platform3 Advanced Analytics API",
    description="RESTful API for the Advanced Analytics and Reporting Framework",
    version="1.0.0",
    docs_url="/analytics/docs",
    redoc_url="/analytics/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analytics framework instance
analytics_framework: Optional[AdvancedAnalyticsFramework] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the analytics framework on startup"""
    global analytics_framework
    try:
        analytics_framework = AdvancedAnalyticsFramework()
        await analytics_framework.initialize()
        logger.info("Analytics API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize analytics framework: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global analytics_framework
    if analytics_framework:
        await analytics_framework.shutdown()
        logger.info("Analytics API shutdown complete")

def get_analytics_framework() -> AdvancedAnalyticsFramework:
    """Dependency to get analytics framework instance"""
    if not analytics_framework:
        raise HTTPException(status_code=503, detail="Analytics framework not initialized")
    return analytics_framework

# Health check endpoints
@app.get("/analytics/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Platform3 Advanced Analytics API"
    }

@app.get("/analytics/status")
async def get_system_status(framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)):
    """Get comprehensive system status"""
    try:
        metrics = framework.get_realtime_metrics()
        
        # Calculate statistics
        total_engines = len(framework.engines)
        active_engines = sum(1 for engine in framework.engines.values() if hasattr(engine, 'is_active'))
        total_metrics = len(metrics)
        
        # Get reports count (simplified)
        reports_dir = Path("reports")
        reports_generated = len(list(reports_dir.glob("*.json"))) if reports_dir.exists() else 0
        
        # Calculate uptime
        uptime_hours = 24  # Placeholder - in real implementation, track actual uptime
        
        return AnalyticsStatsResponse(
            total_engines=total_engines,
            active_engines=active_engines,
            total_metrics=total_metrics,
            reports_generated=reports_generated,
            uptime_hours=uptime_hours,
            last_update=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

# Real-time metrics endpoints
@app.get("/analytics/metrics", response_model=List[MetricResponse])
async def get_realtime_metrics(
    framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework),
    metric_filter: Optional[str] = Query(None, description="Filter metrics by name pattern")
):
    """Get current real-time metrics"""
    try:
        metrics = framework.get_realtime_metrics()
        
        # Filter metrics if requested
        if metric_filter:
            metrics = {k: v for k, v in metrics.items() if metric_filter.lower() in k.lower()}
        
        return [
            MetricResponse(
                metric_name=metric.metric_name,
                value=metric.value,
                timestamp=metric.timestamp,
                context=metric.context,
                alert_threshold=metric.alert_threshold
            )
            for metric in metrics.values()
        ]
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get real-time metrics")

@app.get("/analytics/metrics/{metric_name}", response_model=MetricResponse)
async def get_specific_metric(
    metric_name: str,
    framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)
):
    """Get a specific real-time metric"""
    try:
        metrics = framework.get_realtime_metrics()
        
        if metric_name not in metrics:
            raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
        
        metric = metrics[metric_name]
        return MetricResponse(
            metric_name=metric.metric_name,
            value=metric.value,
            timestamp=metric.timestamp,
            context=metric.context,
            alert_threshold=metric.alert_threshold
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting specific metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metric")

# Analytics data streaming endpoints
@app.post("/analytics/stream")
async def stream_analytics_data(
    request: AnalyticsDataRequest,
    background_tasks: BackgroundTasks,
    framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)
):
    """Stream analytics data for processing"""
    try:
        # Process data in background
        background_tasks.add_task(
            framework.stream_analytics_data,
            request.data,
            request.source
        )
        
        return {
            "status": "accepted",
            "message": "Analytics data queued for processing",
            "timestamp": datetime.utcnow().isoformat(),
            "source": request.source
        }
    except Exception as e:
        logger.error(f"Error streaming analytics data: {e}")
        raise HTTPException(status_code=500, detail="Failed to stream analytics data")

# Analytics engines endpoints
@app.get("/analytics/engines", response_model=List[EngineStatusResponse])
async def get_engines_status(framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)):
    """Get status of all analytics engines"""
    try:
        engines_status = []
        
        for name, engine in framework.engines.items():
            # Get engine metrics
            metrics = framework.get_realtime_metrics()
            performance_key = f"{name}_performance"
            
            performance_score = 0.0
            processed_items = 0
            
            if performance_key in metrics:
                performance_score = metrics[performance_key].value
            
            processed_key = f"{name}_processed"
            if processed_key in metrics:
                processed_items = int(metrics[processed_key].value)
            
            engines_status.append(EngineStatusResponse(
                name=name.replace('_', ' ').title(),
                status="active" if performance_score > 0 else "inactive",
                performance_score=performance_score,
                processed_items=processed_items,
                last_update=datetime.utcnow()
            ))
        
        return engines_status
    except Exception as e:
        logger.error(f"Error getting engines status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get engines status")

@app.get("/analytics/engines/{engine_name}")
async def get_engine_details(
    engine_name: str,
    framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)
):
    """Get detailed information about a specific analytics engine"""
    try:
        if engine_name not in framework.engines:
            raise HTTPException(status_code=404, detail=f"Engine '{engine_name}' not found")
        
        engine = framework.engines[engine_name]
        metrics = framework.get_realtime_metrics()
        
        # Get engine-specific metrics
        engine_metrics = {k: v for k, v in metrics.items() if k.startswith(engine_name)}
        
        return {
            "name": engine_name,
            "type": type(engine).__name__,
            "metrics": {
                k: {
                    "value": v.value,
                    "timestamp": v.timestamp.isoformat(),
                    "context": v.context
                }
                for k, v in engine_metrics.items()
            },
            "last_update": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting engine details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get engine details")

# Reports endpoints
@app.post("/analytics/reports/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks,
    framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)
):
    """Generate a new analytics report"""
    try:
        report = await framework.generate_comprehensive_report(request.timeframe)
        
        # Save report in background
        background_tasks.add_task(framework._save_report, report)
        
        return ReportResponse(
            report_id=report.report_id,
            report_type=report.report_type,
            generated_at=report.generated_at,
            summary=report.summary,
            recommendations=report.recommendations if request.include_recommendations else [],
            confidence_score=report.confidence_score,
            data_preview={
                "timeframe": request.timeframe,
                "engines_count": len(framework.engines),
                "metrics_count": len(framework.get_realtime_metrics())
            }
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

@app.get("/analytics/reports")
async def list_reports(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of reports to return"),
    offset: int = Query(0, ge=0, description="Number of reports to skip")
):
    """List available analytics reports"""
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return {"reports": [], "total": 0}
        
        # Get all report files
        report_files = sorted(
            reports_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Paginate
        total = len(report_files)
        report_files = report_files[offset:offset + limit]
        
        reports = []
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    reports.append({
                        "report_id": report_data.get("report_id"),
                        "report_type": report_data.get("report_type"),
                        "generated_at": report_data.get("generated_at"),
                        "summary": report_data.get("summary"),
                        "confidence_score": report_data.get("confidence_score"),
                        "file_size": report_file.stat().st_size
                    })
            except Exception as e:
                logger.error(f"Error reading report file {report_file}: {e}")
                continue
        
        return {
            "reports": reports,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to list reports")

@app.get("/analytics/reports/{report_id}")
async def get_report(report_id: str):
    """Get a specific analytics report"""
    try:
        reports_dir = Path("reports")
        report_file = reports_dir / f"{report_id}.json"
        
        if not report_file.exists():
            raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
        
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        return report_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        raise HTTPException(status_code=500, detail="Failed to get report")

@app.get("/analytics/reports/{report_id}/download")
async def download_report(report_id: str):
    """Download a specific analytics report"""
    try:
        reports_dir = Path("reports")
        report_file = reports_dir / f"{report_id}.json"
        
        if not report_file.exists():
            raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
        
        return FileResponse(
            path=str(report_file),
            filename=f"analytics_report_{report_id}.json",
            media_type="application/json"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail="Failed to download report")

@app.delete("/analytics/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete a specific analytics report"""
    try:
        reports_dir = Path("reports")
        report_file = reports_dir / f"{report_id}.json"
        
        if not report_file.exists():
            raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
        
        report_file.unlink()
        
        return {
            "status": "success",
            "message": f"Report '{report_id}' deleted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting report: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete report")

# Scheduled reports endpoints
@app.post("/analytics/reports/schedule")
async def schedule_report(
    request: ScheduledReportRequest,
    framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)
):
    """Schedule automatic report generation"""
    try:
        await framework.schedule_report(
            request.report_id,
            request.timeframe,
            request.interval_hours
        )
        
        return {
            "status": "success",
            "message": f"Report '{request.report_id}' scheduled successfully",
            "schedule": {
                "report_id": request.report_id,
                "timeframe": request.timeframe,
                "interval_hours": request.interval_hours,
                "enabled": request.enabled
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error scheduling report: {e}")
        raise HTTPException(status_code=500, detail="Failed to schedule report")

@app.get("/analytics/reports/scheduled")
async def list_scheduled_reports(framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)):
    """List all scheduled reports"""
    try:
        scheduled_reports = framework.scheduled_reports
        
        return {
            "scheduled_reports": [
                {
                    "report_id": report_id,
                    **schedule_data
                }
                for report_id, schedule_data in scheduled_reports.items()
            ],
            "total": len(scheduled_reports),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing scheduled reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to list scheduled reports")

@app.delete("/analytics/reports/scheduled/{report_id}")
async def unschedule_report(
    report_id: str,
    framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)
):
    """Remove a scheduled report"""
    try:
        if report_id not in framework.scheduled_reports:
            raise HTTPException(status_code=404, detail=f"Scheduled report '{report_id}' not found")
        
        del framework.scheduled_reports[report_id]
        
        return {
            "status": "success",
            "message": f"Scheduled report '{report_id}' removed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unscheduling report: {e}")
        raise HTTPException(status_code=500, detail="Failed to unschedule report")

# Analytics configuration endpoints
@app.get("/analytics/config")
async def get_analytics_config(framework: AdvancedAnalyticsFramework = Depends(get_analytics_framework)):
    """Get current analytics configuration"""
    try:
        return {
            "cache_timeout": framework.cache_timeout,
            "engines_count": len(framework.engines),
            "realtime_metrics_enabled": True,
            "redis_url": framework.redis_url,
            "reports_directory": "reports",
            "auto_cleanup_enabled": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analytics config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics configuration")

# Main function to run the API server
def run_server(host: str = "0.0.0.0", port: int = 8002, reload: bool = False):
    """Run the Analytics API server"""
    uvicorn.run(
        "AnalyticsAPI:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_server()
