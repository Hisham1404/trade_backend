"""
Google Agent Development Kit (ADK) Configuration
Integration with Google AI Studio for trading agent capabilities
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ADKSettings(BaseSettings):
    """Google ADK configuration settings using Google AI Studio"""
    
    # Google AI Studio Configuration (not Vertex AI)
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    project_id: Optional[str] = Field(None, env="GOOGLE_PROJECT_ID")
    
    # ADK Agent Configuration
    agent_name: str = Field(default="TradingAgent", env="ADK_AGENT_NAME")
    agent_version: str = Field(default="1.0.0", env="ADK_AGENT_VERSION")
    agent_description: str = Field(default="AI Trading Agent for market analysis and decision making",env="ADK_AGENT_DESCRIPTION")
    
    # Model Configuration
    default_model: str = Field(default="gemini-1.5-flash", env="ADK_DEFAULT_MODEL")
    temperature: float = Field(default=0.1, env="ADK_TEMPERATURE")
    max_tokens: int = Field(default=8192, env="ADK_MAX_TOKENS")
    
    # Agent Capabilities
    enable_tools: bool = Field(default=True, env="ADK_ENABLE_TOOLS")
    enable_memory: bool = Field(default=True, env="ADK_ENABLE_MEMORY")
    enable_planning: bool = Field(default=True, env="ADK_ENABLE_PLANNING")
    
    # Safety and Filtering
    safety_settings: Dict[str, str] = Field(
        default={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
        }
    )
    
    # Logging and Monitoring
    enable_logging: bool = Field(default=True, env="ADK_ENABLE_LOGGING")
    log_level: str = Field(default="INFO", env="ADK_LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from the environment


class ADKToolConfig(BaseModel):
    """Configuration for ADK tools and capabilities"""
    
    # Trading-specific tools
    market_data_tools: bool = Field(default=True)
    technical_analysis_tools: bool = Field(default=True)
    portfolio_management_tools: bool = Field(default=True)
    news_analysis_tools: bool = Field(default=True)
    
    # External API integrations
    external_apis: Dict[str, bool] = Field(
        default={
            "yahoo_finance": True,
            "alpha_vantage": True,
            "news_feeds": True,
            "economic_calendar": True
        }
    )
    
    # Tool execution limits
    max_tool_calls_per_request: int = Field(default=10)
    tool_timeout_seconds: int = Field(default=30)


def get_adk_settings() -> ADKSettings:
    """Get ADK configuration settings"""
    return ADKSettings()


def get_tool_config() -> ADKToolConfig:
    """Get ADK tool configuration"""
    return ADKToolConfig()


def validate_adk_setup() -> Dict[str, Any]:
    """Validate ADK configuration and return status"""
    settings = get_adk_settings()
    tool_config = get_tool_config()
    
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "configuration": {}
    }
    
    # Validate required fields
    if not settings.google_api_key:
        validation_result["valid"] = False
        validation_result["errors"].append("GOOGLE_API_KEY is required")
    
    # Validate model configuration
    if settings.temperature < 0 or settings.temperature > 2:
        validation_result["warnings"].append("Temperature should be between 0 and 2")
    
    if settings.max_tokens < 1024:
        validation_result["warnings"].append("Max tokens might be too low for complex responses")
    
    # Configuration summary
    validation_result["configuration"] = {
        "agent_name": settings.agent_name,
        "agent_version": settings.agent_version,
        "model": settings.default_model,
        "tools_enabled": settings.enable_tools,
        "memory_enabled": settings.enable_memory,
        "planning_enabled": settings.enable_planning,
        "tool_count": len([k for k, v in tool_config.external_apis.items() if v])
    }
    
    return validation_result 