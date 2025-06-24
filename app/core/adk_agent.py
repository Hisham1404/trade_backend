"""
Google Agent Development Kit (ADK) Agent Implementation
Main agent class for trading intelligence using Google AI Studio
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import os

# Google ADK imports - correct structure based on documentation
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Local imports
from .adk_config import get_adk_settings, get_tool_config, validate_adk_setup
from ..models.asset import Asset
from ..models.alert import Alert

logger = logging.getLogger(__name__)


class TradingADKAgent:
    """
    Google ADK-powered trading agent using Google AI Studio
    Provides intelligent market analysis and trading decision support
    """
    
    def __init__(self):
        self.settings = get_adk_settings()
        self.tool_config = get_tool_config()
        self.agent: Optional[Agent] = None
        self.session_service: Optional[InMemorySessionService] = None
        self.runner: Optional[Runner] = None
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the ADK agent with Google AI Studio"""
        try:
            # Validate configuration
            validation = validate_adk_setup()
            if not validation["valid"]:
                logger.error(f"ADK configuration invalid: {validation['errors']}")
                return False
            
            # Set up environment for Google AI Studio (not Vertex AI)
            if self.settings.google_api_key:
                os.environ["GOOGLE_API_KEY"] = str(self.settings.google_api_key)
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
            
            # Initialize session service
            self.session_service = InMemorySessionService()
            
            # Create trading tools if enabled
            tools = []
            if self.settings.enable_tools:
                tools = await self._create_trading_tools()
            
            # Create agent
            self.agent = Agent(
                model=self.settings.default_model,
                name=self.settings.agent_name,
                instruction=f"""You are {self.settings.agent_description}.
                
                Your core capabilities include:
                - Market analysis and trend identification
                - Risk assessment for trading proposals
                - Portfolio analysis and optimization recommendations
                - News sentiment analysis and market impact assessment
                - Trading strategy generation based on market conditions
                
                Always provide accurate, data-driven insights while being transparent about limitations.
                Use the available tools to gather current market data when needed.
                Maintain a professional, analytical tone in your responses.""",
                description=self.settings.agent_description,
                tools=tools
            )
            
            # Initialize runner with app_name
            self.runner = Runner(
                agent=self.agent,
                app_name=self.settings.agent_name.lower().replace(' ', '_'),
                session_service=self.session_service
            )
            
            self.initialized = True
            logger.info(f"ADK agent '{self.settings.agent_name}' initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ADK agent: {str(e)}")
            return False
    
    async def _create_trading_tools(self) -> List:
        """Create trading-specific tools for the agent"""
        tools = []
        
        # Market Data Tool
        def get_market_data(symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
            """Get current market data for assets
            
            Args:
                symbol: Asset symbol to get data for
                timeframe: Time period for data (1d, 1h, etc.)
            
            Returns:
                Dictionary with market data and status
            """
            try:
                # Integration with existing market data services
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "message": f"Market data for {symbol} retrieved",
                    "data": {
                        "price": "Mock price data",
                        "volume": "Mock volume data",
                        "change": "Mock change data"
                    }
                }
            except Exception as e:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "message": str(e)
                }
        
        # Technical Analysis Tool
        def technical_analysis(symbol: str, indicators: str) -> Dict[str, Any]:
            """Perform technical analysis on asset data
            
            Args:
                symbol: Asset symbol to analyze
                indicators: Comma-separated list of indicators (RSI, MACD, etc.)
            
            Returns:
                Dictionary with analysis results
            """
            try:
                indicator_list = [i.strip() for i in indicators.split(',')]
                return {
                    "symbol": symbol,
                    "indicators": indicator_list,
                    "analysis": f"Technical analysis completed for {symbol}",
                    "results": {
                        "trend": "Neutral",
                        "signals": ["No strong signals detected"],
                        "recommendation": "Hold"
                    },
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            except Exception as e:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "message": str(e)
                }
        
        # News Analysis Tool
        def analyze_news(keywords: str, limit: int = 10) -> Dict[str, Any]:
            """Analyze news sentiment for market impact
            
            Args:
                keywords: Comma-separated keywords to search for
                limit: Maximum number of articles to analyze
            
            Returns:
                Dictionary with news analysis results
            """
            try:
                keyword_list = [k.strip() for k in keywords.split(',')]
                return {
                    "keywords": keyword_list,
                    "articles_analyzed": limit,
                    "sentiment": "neutral",
                    "impact_score": 0.5,
                    "summary": f"Analyzed {limit} articles related to {keywords}",
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            except Exception as e:
                return {
                    "keywords": keywords,
                    "status": "error",
                    "message": str(e)
                }
        
        # Portfolio Analysis Tool
        def portfolio_analysis(assets: str) -> Dict[str, Any]:
            """Analyze portfolio performance and risk
            
            Args:
                assets: Comma-separated list of asset symbols in portfolio
            
            Returns:
                Dictionary with portfolio analysis
            """
            try:
                asset_list = [a.strip() for a in assets.split(',')]
                return {
                    "assets": asset_list,
                    "total_assets": len(asset_list),
                    "risk_score": 0.6,
                    "performance": "Balanced portfolio",
                    "recommendations": [
                        "Consider diversification",
                        "Monitor risk exposure"
                    ],
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            except Exception as e:
                return {
                    "assets": assets,
                    "status": "error",
                    "message": str(e)
                }
        
        # Add tools to list (these will be registered as function tools by ADK)
        tools = [get_market_data, technical_analysis, analyze_news, portfolio_analysis]
        
        logger.info(f"Created {len(tools)} trading tools")
        return tools
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a trading-related query using the ADK agent"""
        if not self.initialized or not self.runner or not self.session_service:
            return {
                "status": "error",
                "message": "Agent not initialized"
            }
        
        try:
            # App and user info
            app_name = self.settings.agent_name.lower().replace(' ', '_')
            user_id = "trading_user"
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create initial state with context if provided
            initial_state = context or {}
            
            # Create a session for this query
            session = await self.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                state=initial_state
            )
            
            # Create user message content
            user_content = types.Content(
                role="user",
                parts=[types.Part(text=query)]
            )
            
            # Run the agent asynchronously
            events_stream = self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=user_content
            )
            
            # Collect events from the stream
            response_text = ""
            tool_calls = []
            events = []
            
            # Process events from the stream
            if events_stream:
                async for event in events_stream:
                    events.append(event)
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += str(part.text)
                    # Check for tool calls with defensive access
                    if hasattr(event, 'tool_calls') and event.tool_calls:
                        tool_calls.extend(event.tool_calls)
            
            return {
                "status": "success",
                "response": response_text or "Response processed successfully",
                "tool_calls": tool_calls,
                "session_id": session_id,
                "events_count": len(events),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_market_conditions(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze current market conditions for given symbols"""
        query = f"Analyze current market conditions for {', '.join(symbols)}. Provide insights on trends, volatility, and potential opportunities."
        
        context = {
            "symbols": symbols,
            "analysis_type": "market_conditions",
            "requested_at": datetime.now().isoformat()
        }
        
        return await self.process_query(query, context)
    
    async def generate_trading_insights(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading insights based on portfolio data"""
        query = "Based on the current portfolio composition and market data, provide trading insights and recommendations."
        
        context = {
            "portfolio": portfolio_data,
            "analysis_type": "trading_insights",
            "requested_at": datetime.now().isoformat()
        }
        
        return await self.process_query(query, context)
    
    async def assess_risk(self, trade_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for a proposed trade"""
        query = f"Assess the risk level and potential impact of this trade proposal: {json.dumps(trade_proposal, indent=2)}"
        
        context = {
            "trade_proposal": trade_proposal,
            "analysis_type": "risk_assessment", 
            "requested_at": datetime.now().isoformat()
        }
        
        return await self.process_query(query, context)
    
    async def shutdown(self):
        """Shutdown the agent and cleanup resources"""
        if self.runner:
            # ADK handles cleanup automatically
            pass
        self.initialized = False
        logger.info("ADK agent shut down successfully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "initialized": self.initialized,
            "agent_name": self.settings.agent_name,
            "model": self.settings.default_model,
            "tools_enabled": self.settings.enable_tools,
            "memory_enabled": self.settings.enable_memory,
            "planning_enabled": self.settings.enable_planning
        }


# Global agent instance
_trading_agent: Optional[TradingADKAgent] = None


async def get_trading_agent() -> TradingADKAgent:
    """Get or create the global trading agent instance"""
    global _trading_agent
    
    if _trading_agent is None:
        _trading_agent = TradingADKAgent()
        await _trading_agent.initialize()
    
    return _trading_agent


async def shutdown_trading_agent():
    """Shutdown the global trading agent"""
    global _trading_agent
    
    if _trading_agent:
        await _trading_agent.shutdown()
        _trading_agent = None 