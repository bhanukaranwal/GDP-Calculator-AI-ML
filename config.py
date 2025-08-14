"""
Configuration management for GDP AI Platform
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic App Settings
    APP_NAME: str = "GDP AI/ML Analytics Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_SECRET: str = Field(..., env="JWT_SECRET")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    REDIS_URL: str = Field(..., env="REDIS_URL")
    NEO4J_URI: Optional[str] = Field(default=None, env="NEO4J_URI")
    NEO4J_USER: Optional[str] = Field(default=None, env="NEO4J_USER")
    NEO4J_PASSWORD: Optional[str] = Field(default=None, env="NEO4J_PASSWORD")
    
    # External APIs
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    WORLD_BANK_API_KEY: Optional[str] = Field(default=None, env="WORLD_BANK_API_KEY")
    IMF_API_KEY: Optional[str] = Field(default=None, env="IMF_API_KEY")
    OECD_API_KEY: Optional[str] = Field(default=None, env="OECD_API_KEY")
    MOSPI_API_KEY: Optional[str] = Field(default=None, env="MOSPI_API_KEY")
    
    # ML/AI Settings
    MODEL_STORAGE_PATH: str = Field(default="/app/models", env="MODEL_STORAGE_PATH")
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # Data Processing
    DATA_STORAGE_PATH: str = Field(default="/app/data", env="DATA_STORAGE_PATH")
    MAX_UPLOAD_SIZE: int = Field(default=100 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 100MB
    
    # Monitoring
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    GRAFANA_PORT: int = Field(default=3000, env="GRAFANA_PORT")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # CORS
    ALLOWED_HOSTS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="ALLOWED_HOSTS"
    )
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    
    # Celery (Background Tasks)
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Blockchain
    WEB3_PROVIDER_URI: Optional[str] = Field(default=None, env="WEB3_PROVIDER_URI")
    BLOCKCHAIN_PRIVATE_KEY: Optional[str] = Field(default=None, env="BLOCKCHAIN_PRIVATE_KEY")
    
    # Feature Flags
    ENABLE_BLOCKCHAIN: bool = Field(default=False, env="ENABLE_BLOCKCHAIN")
    ENABLE_VR_SUPPORT: bool = Field(default=False, env="ENABLE_VR_SUPPORT")
    ENABLE_VOICE_INTERFACE: bool = Field(default=False, env="ENABLE_VOICE_INTERFACE")
    ENABLE_FEDERATED_LEARNING: bool = Field(default=False, env="ENABLE_FEDERATED_LEARNING")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()