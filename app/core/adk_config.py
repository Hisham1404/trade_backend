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
    google_api_key: Optional[str] = Field(default=None, description="Google API Key from environment")
    project_id: Optional[str] = Field(default=None, description="Google Project ID (optional)")
    
    # ADK Agent Configuration
    agent_name: str = Field(default="TradingAgent", description="ADK Agent Name")
    agent_version: str = Field(default="1.0.0", description="ADK Agent Version")
    agent_description: str = Field(default="AI Trading Agent for market analysis and decision making", description="ADK Agent Description")
    
    # Model Configuration
    default_model: str = Field(default="gemini-1.5-flash", description="Default AI model")
    temperature: float = Field(default=0.1, description="AI model temperature")
    max_tokens: int = Field(default=8192, description="Maximum tokens")
    
    # Agent Capabilities
    enable_tools: bool = Field(default=True, description="Enable ADK tools")
    enable_memory: bool = Field(default=True, description="Enable ADK memory")
    enable_planning: bool = Field(default=True, description="Enable ADK planning")
    
    # Safety and Filtering
    safety_settings: Dict[str, str] = Field(
        default={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
        },
        description="Safety settings for AI model"
    )
    
    # Logging and Monitoring
    enable_logging: bool = Field(default=True, description="Enable logging")
    log_level: str = Field(default="INFO", description="Log level")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
        "env_prefix": "",
        "env_nested_delimiter": "__",
    }
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )
    
    def __init__(self, **kwargs):
        # Load Google API key from environment if not provided
        if not kwargs.get('google_api_key'):
            kwargs['google_api_key'] = os.getenv('GOOGLE_API_KEY')
        
        # Load other environment variables
        if not kwargs.get('project_id'):
            kwargs['project_id'] = os.getenv('GOOGLE_PROJECT_ID')
        
        if not kwargs.get('agent_name'):
            kwargs['agent_name'] = os.getenv('ADK_AGENT_NAME', 'TradingAgent')
        
        if not kwargs.get('agent_version'):
            kwargs['agent_version'] = os.getenv('ADK_AGENT_VERSION', '1.0.0')
        
        if not kwargs.get('agent_description'):
            kwargs['agent_description'] = os.getenv('ADK_AGENT_DESCRIPTION', 'AI Trading Agent for market analysis and decision making')
        
        if not kwargs.get('default_model'):
            kwargs['default_model'] = os.getenv('ADK_DEFAULT_MODEL', 'gemini-1.5-flash')
        
        if 'temperature' not in kwargs:
            temp_str = os.getenv('ADK_TEMPERATURE', '0.1')
            try:
                kwargs['temperature'] = float(temp_str)
            except ValueError:
                kwargs['temperature'] = 0.1
        
        if 'max_tokens' not in kwargs:
            tokens_str = os.getenv('ADK_MAX_TOKENS', '8192')
            try:
                kwargs['max_tokens'] = int(tokens_str)
            except ValueError:
                kwargs['max_tokens'] = 8192
        
        # Boolean environment variables
        for bool_field in ['enable_tools', 'enable_memory', 'enable_planning', 'enable_logging']:
            if bool_field not in kwargs:
                env_name = f"ADK_{bool_field.upper()}"
                env_value = os.getenv(env_name, 'true').lower()
                kwargs[bool_field] = env_value in ('true', '1', 'yes', 'on')
        
        if not kwargs.get('log_level'):
            kwargs['log_level'] = os.getenv('ADK_LOG_LEVEL', 'INFO')
        
        super().__init__(**kwargs)


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
        validation_result["errors"].append("GOOGLE_API_KEY environment variable is required for ADK functionality")
    
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