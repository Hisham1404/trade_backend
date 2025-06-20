from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = Field(..., description="Application name")
    APP_VERSION: str = Field(..., description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode")
    ENVIRONMENT: str = Field(default="production", description="Environment")
    
    # Security
    SECRET_KEY: str = Field(..., description="Secret key for JWT token generation")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration")
    
    # Database
    DATABASE_URL: str = Field(..., description="Database connection URL")
    DATABASE_URL_TEST: Optional[str] = Field(default=None, description="Test database URL")
    
    # External API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    FINNHUB_API_KEY: Optional[str] = Field(default=None, description="Finnhub API key")
    POLYGON_API_KEY: Optional[str] = Field(default=None, description="Polygon API key")
    NEWS_API_KEY: Optional[str] = Field(default=None, description="News API key")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"], description="Allowed CORS origins")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Rate limit per minute")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(default=False, description="Enable Prometheus monitoring")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables

# Create settings instance
settings = Settings() 