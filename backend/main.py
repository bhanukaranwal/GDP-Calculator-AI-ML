"""
Ultra-Advanced GDP AI/ML Analytics Platform
Main FastAPI application entry point
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.config import settings
from core.database import engine, database
from core.monitoring import setup_monitoring, metrics
from api.routes import (
    gdp_routes,
    forecasting_routes,
    data_integration_routes,
    ai_routes,
    visualization_routes,
    auth_routes,
    admin_routes
)
from services.ml_service import MLService
from services.data_service import DataService
from services.notification_service import NotificationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('gdp_platform_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('gdp_platform_request_duration_seconds', 'Request duration')


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        response = await call_next(request)
        
        # Record metrics
        duration = asyncio.get_event_loop().time() - start_time
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
        REQUEST_DURATION.observe(duration)
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting GDP AI Platform...")
    
    # Initialize database
    await database.connect()
    logger.info("Database connected")
    
    # Initialize ML services
    ml_service = MLService()
    await ml_service.initialize()
    logger.info("ML services initialized")
    
    # Initialize data services
    data_service = DataService()
    await data_service.initialize()
    logger.info("Data services initialized")
    
    # Setup monitoring
    setup_monitoring()
    logger.info("Monitoring setup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GDP AI Platform...")
    await database.disconnect()
    await ml_service.cleanup()
    await data_service.cleanup()


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="GDP AI/ML Analytics Platform",
        description="Ultra-advanced economic intelligence system for GDP calculation, forecasting, and policy simulation",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(MetricsMiddleware)
    
    # Include routers
    app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(gdp_routes.router, prefix="/api/v1/gdp", tags=["GDP Calculation"])
    app.include_router(forecasting_routes.router, prefix="/api/v1/forecasting", tags=["AI/ML Forecasting"])
    app.include_router(data_integration_routes.router, prefix="/api/v1/data", tags=["Data Integration"])
    app.include_router(ai_routes.router, prefix="/api/v1/ai", tags=["AI Services"])
    app.include_router(visualization_routes.router, prefix="/api/v1/viz", tags=["Visualization"])
    app.include_router(admin_routes.router, prefix="/api/v1/admin", tags=["Administration"])
    
    # Health check endpoints
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": asyncio.get_event_loop().time()
        }
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        return Response(generate_latest(), media_type="text/plain")
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with platform information"""
        return {
            "message": "GDP AI/ML Analytics Platform",
            "version": "1.0.0",
            "documentation": "/docs",
            "health": "/health",
            "metrics": "/metrics"
        }
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "path": request.url.path
            }
        )
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        access_log=True,
        log_level="info"
    )
