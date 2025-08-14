"""
GDP Data Models and Schemas
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

# Database Models
class GDPRecord(Base):
    __tablename__ = "gdp_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), nullable=False, index=True)
    period = Column(String(20), nullable=False, index=True)
    gdp_value = Column(Float, nullable=False)
    method = Column(String(20), nullable=False)
    components = Column(JSON)
    confidence_interval = Column(JSON)
    quality_score = Column(Float)
    anomaly_flags = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ForecastRecord(Base):
    __tablename__ = "forecast_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code = Column(String(3), nullable=False, index=True)
    forecast_date = Column(DateTime, nullable=False)
    target_period = Column(String(20), nullable=False)
    predicted_value = Column(Float, nullable=False)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(20))
    ensemble_weights = Column(JSON)
    features_used = Column(JSON)
    prediction_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    source_type = Column(String(50), nullable=False)  # api, file, manual
    url = Column(String(500))
    api_key_required = Column(Boolean, default=False)
    update_frequency = Column(String(20))  # daily, weekly, monthly, quarterly
    last_update = Column(DateTime)
    status = Column(String(20), default='active')  # active, inactive, error
    configuration = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelPerformanceRecord(Base):
    __tablename__ = "model_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(50), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    country_code = Column(String(3), index=True)
    evaluation_date = Column(DateTime, nullable=False)
    mse = Column(Float)
    mae = Column(Float)
    mape = Column(Float)
    r2_score = Column(Float)
    directional_accuracy = Column(Float)
    training_time = Column(Float)
    prediction_time = Column(Float)
    feature_importance = Column(JSON)
    hyperparameters = Column(JSON)
    validation_method = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic Models
@dataclass
class ExpenditureComponents:
    """GDP Expenditure components"""
    consumption: float
    investment: float
    government_spending: float
    exports: float
    imports: float
    net_exports: float


@dataclass
class IncomeComponents:
    """GDP Income components"""
    wages_salaries: float
    corporate_profits: float
    rental_income: float
    interest_income: float
    proprietor_income: float
    taxes_minus_subsidies: float
    depreciation: float


@dataclass
class OutputComponents:
    """GDP Output components by sector"""
    agriculture: float
    manufacturing: float
    services: float
    construction: float
    mining: float
    utilities: float
    other_sectors: float


class GDPCalculationRequest(BaseModel):
    """Request model for GDP calculation"""
    country_code: str = Field(..., min_length=2, max_length=3)
    period: str = Field(..., regex=r'^\d{4}(-Q[1-4]|-\d{2})?$')
    method: str = Field(..., regex=r'^(expenditure|income|output)$')
    data: Dict[str, Any] = Field(...)
    apply_ai_corrections: bool = Field(default=True)
    include_uncertainty: bool = Field(default=True)
    
    @validator('country_code')
    def validate_country_code(cls, v):
        return v.upper()


class GDPCalculationResult(BaseModel):
    """Result model for GDP calculation"""
    gdp_value: float
    components: Dict[str, Any]
    country_code: str
    period: str
    method: str
    confidence_interval: Optional[Tuple[float, float]] = None
    quality_score: float
    anomaly_flags: Dict[str, bool]
    metadata: Dict[str, Any]
    calculation_timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ForecastRequest(BaseModel):
    """Request model for GDP forecasting"""
    country_code: str = Field(..., min_length=2, max_length=3)
    forecast_horizon: int = Field(default=4, ge=1, le=20)
    include_features: List[str] = Field(default_factory=list)
    model_preference: Optional[str] = None
    ensemble_method: str = Field(default='weighted_average')
    return_uncertainty: bool = Field(default=True)
    scenario_adjustments: Optional[Dict[str, float]] = None
    
    @validator('country_code')
    def validate_country_code(cls, v):
        return v.upper()


class ForecastResult(BaseModel):
    """Result model for GDP forecasting"""
    predictions: List[float]
    timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    model_info: Dict[str, Any]
    ensemble_method: str
    forecast_horizon: int
    uncertainty_estimates: Optional[List[float]] = None
    metadata: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelPrediction(BaseModel):
    """Individual model prediction"""
    model_name: str
    prediction: float
    confidence: float
    features_used: List[str]
    processing_time: float


class DataIntegrationRequest(BaseModel):
    """Request for data integration"""
    source_name: str
    data_type: str  # gdp, cpi, trade, etc.
    country_codes: List[str] = Field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_metadata: bool = Field(default=True)
    validate_data: bool = Field(default=True)


class DataQualityReport(BaseModel):
    """Data quality assessment report"""
    source: str
    completeness_score: float
    accuracy_score: float
    timeliness_score: float
    consistency_score: float
    overall_score: float
    issues_found: List[str]
    recommendations: List[str]
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VisualizationRequest(BaseModel):
    """Request for data visualization"""
    viz_type: str  # line, bar, map, 3d, etc.
    data_source: str
    country_codes: List[str] = Field(default_factory=list)
    time_range: Optional[Tuple[str, str]] = None
    aggregation: str = Field(default='none')  # none, quarterly, yearly
    include_forecast: bool = Field(default=False)
    style_preferences: Dict[str, Any] = Field(default_factory=dict)


class NLQueryRequest(BaseModel):
    """Natural language query request"""
    query: str = Field(..., min_length=10, max_length=500)
    context: Optional[Dict[str, Any]] = None
    include_visualization: bool = Field(default=True)
    response_format: str = Field(default='json')  # json, text, html


class NLQueryResponse(BaseModel):
    """Natural language query response"""
    answer: str
    data: Optional[Dict[str, Any]] = None
    visualizations: Optional[List[Dict[str, Any]]] = None
    confidence: float
    sources: List[str]
    suggestions: List[str]
    processing_time: float


class UserProfile(BaseModel):
    """User profile and preferences"""
    user_id: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    dashboard_config: Dict[str, Any] = Field(default_factory=dict)
    api_usage: Dict[str, Any] = Field(default_factory=dict)
    last_login: Optional[datetime] = None


class AlertConfig(BaseModel):
    """Alert configuration"""
    alert_id: str
    user_id: str
    condition: Dict[str, Any]
    notification_channels: List[str]
    is_active: bool = Field(default=True)
    created_at: datetime
    triggered_count: int = Field(default=0)


class SystemHealth(BaseModel):
    """System health status"""
    overall_status: str  # healthy, warning, critical
    components: Dict[str, str]
    metrics: Dict[str, float]
    alerts: List[str]
    last_check: datetime
    uptime: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Error Models
class APIError(BaseModel):
    """Standard API error response"""
    error: str
    message: str
    status_code: int
    timestamp: datetime
    path: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    invalid_value: Optional[Any] = None


# Response Models
class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool


# Configuration Models
class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: str
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=3600)


class CacheConfig(BaseModel):
    """Cache configuration"""
    redis_url: str
    default_ttl: int = Field(default=300)  # 5 minutes
    max_connections: int = Field(default=10)


class MLConfig(BaseModel):
    """ML model configuration"""
    model_storage_path: str
    auto_retrain: bool = Field(default=True)
    retrain_threshold: float = Field(default=0.05)
    ensemble_weights: Dict[str, float] = Field(default_factory=dict)
    feature_selection: bool = Field(default=True)


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    prometheus_enabled: bool = Field(default=True)
    grafana_enabled: bool = Field(default=True)
    log_level: str = Field(default='INFO')
    alert_channels: List[str] = Field(default_factory=list)