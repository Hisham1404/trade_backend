from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional, Union
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
    DATABASE_URL: str = "sqlite:///./trading_agent.db"
    DATABASE_URL_TEST: Optional[str] = Field(default=None, description="Test database URL")
    
    # External API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    FINNHUB_API_KEY: Optional[str] = Field(default=None, description="Finnhub API key")
    POLYGON_API_KEY: Optional[str] = Field(default=None, description="Polygon API key")
    NEWS_API_KEY: Optional[str] = Field(default=None, description="News API key")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    
    # Celery Configuration
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", description="Celery broker URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/1", description="Celery result backend URL")
    CELERY_TASK_SERIALIZER: str = Field(default="json", description="Task serialization format")
    CELERY_RESULT_SERIALIZER: str = Field(default="json", description="Result serialization format")
    CELERY_ACCEPT_CONTENT: Union[str, List[str]] = Field(default="json", description="Accepted content types")
    CELERY_TIMEZONE: str = Field(default="UTC", description="Celery timezone")
    CELERY_ENABLE_UTC: bool = Field(default=True, description="Enable UTC timezone")
    CELERY_TASK_TRACK_STARTED: bool = Field(default=True, description="Track task started state")
    CELERY_TASK_TIME_LIMIT: int = Field(default=3600, description="Task hard time limit in seconds")
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=3000, description="Task soft time limit in seconds")
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = Field(default=1, description="Worker prefetch multiplier")
    CELERY_TASK_ACKS_LATE: bool = Field(default=True, description="Acknowledge tasks late")
    CELERY_WORKER_SEND_TASK_EVENTS: bool = Field(default=True, description="Send task events")
    CELERY_TASK_SEND_SENT_EVENT: bool = Field(default=True, description="Send task sent events")
    CELERY_TASK_ALWAYS_EAGER: bool = Field(default=False, description="Execute tasks eagerly (for testing)")
    
    # Celery Beat Schedule Configuration
    CELERY_BEAT_SCHEDULE_HIGH_PRIORITY_INTERVAL: int = Field(default=60, description="High priority scraping interval in seconds")
    CELERY_BEAT_SCHEDULE_GENERAL_SCRAPING_INTERVAL: int = Field(default=1800, description="General scraping interval in seconds")
    CELERY_BEAT_SCHEDULE_OPTION_CHAIN_INTERVAL: int = Field(default=300, description="Option chain update interval in seconds")
    CELERY_BEAT_SCHEDULE_PARTICIPANT_FLOW_HOUR: int = Field(default=16, description="Participant flow update hour")
    CELERY_BEAT_SCHEDULE_PARTICIPANT_FLOW_MINUTE: int = Field(default=30, description="Participant flow update minute")
    
    # Redis Connection Pool Settings for Celery
    CELERY_BROKER_POOL_LIMIT: int = Field(default=10, description="Redis connection pool limit")
    CELERY_BROKER_CONNECTION_RETRY: bool = Field(default=True, description="Retry Redis connections")
    CELERY_BROKER_CONNECTION_MAX_RETRIES: int = Field(default=3, description="Max Redis connection retries")
    CELERY_RESULT_BACKEND_CONNECTION_RETRY: bool = Field(default=True, description="Retry result backend connections")
    CELERY_TASK_RESULT_EXPIRES: int = Field(default=3600, description="Task result expiration in seconds")
    
    # CORS
    ALLOWED_ORIGINS: Union[str, List[str]] = Field(default="http://localhost:3000,http://localhost:8080", description="Allowed CORS origins")
    
    @property
    def celery_accept_content_list(self) -> List[str]:
        """Get CELERY_ACCEPT_CONTENT as a list"""
        if isinstance(self.CELERY_ACCEPT_CONTENT, str):
            return [item.strip() for item in self.CELERY_ACCEPT_CONTENT.split(',')]
        return self.CELERY_ACCEPT_CONTENT if self.CELERY_ACCEPT_CONTENT else ["json"]
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Get ALLOWED_ORIGINS as a list"""
        if isinstance(self.ALLOWED_ORIGINS, str):
            return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',')]
        return self.ALLOWED_ORIGINS if self.ALLOWED_ORIGINS else ["http://localhost:3000", "http://localhost:8080"]
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Rate limit per minute")
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = Field(default=False, description="Enable Prometheus monitoring")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # Push notification settings
    FIREBASE_CREDENTIALS_PATH: Optional[str] = None
    APNS_KEY_PATH: Optional[str] = None
    APNS_KEY_ID: Optional[str] = None
    APNS_TEAM_ID: Optional[str] = None
    APNS_USE_SANDBOX: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables

# Create settings instance
settings = Settings() 