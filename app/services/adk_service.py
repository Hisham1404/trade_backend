"""
Google ADK Service Layer
Integration service for Google Agent Development Kit with trading system
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from ..core.adk_agent import get_trading_agent, shutdown_trading_agent
from ..core.adk_config import validate_adk_setup
from ..models.asset import Asset
from ..models.alert import Alert

logger = logging.getLogger(__name__)


class ADKService:
    """Service layer for Google ADK integration"""
    
    def __init__(self):
        self.agent = None
        self.is_initialized = False
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the ADK service"""
        try:
            # Validate configuration first
            validation = validate_adk_setup()
            if not validation["valid"]:
                return {
                    "success": False,
                    "message": "ADK configuration invalid",
                    "errors": validation["errors"],
                    "timestamp": datetime.now().isoformat()
                }
            
            # Initialize agent
            self.agent = await get_trading_agent()
            
            if self.agent.initialized:
                self.is_initialized = True
                return {
                    "success": True,
                    "message": "ADK service initialized successfully",
                    "configuration": validation["configuration"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to initialize ADK agent",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"ADK service initialization failed: {str(e)}")
            return {
                "success": False,
                "message": f"Initialization error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current ADK service status"""
        if not self.is_initialized or not self.agent:
            return {
                "initialized": False,
                "message": "ADK service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            agent_status = self.agent.get_status()
            return {
                "initialized": self.is_initialized,
                "agent_status": agent_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting ADK status: {str(e)}")
            return {
                "initialized": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_market(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze market conditions for given symbols"""
        if not self.is_initialized or not self.agent:
            return {
                "success": False,
                "message": "ADK service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            result = await self.agent.analyze_market_conditions(symbols)
            return {
                "success": result["status"] == "success",
                "analysis": result,
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Market analysis error: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "symbols": symbols,
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_insights(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading insights based on portfolio data"""
        if not self.is_initialized or not self.agent:
            return {
                "success": False,
                "message": "ADK service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            result = await self.agent.generate_trading_insights(portfolio_data)
            return {
                "success": result["status"] == "success",
                "insights": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Insights generation error: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def assess_risk(self, trade_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for a trade proposal"""
        if not self.is_initialized or not self.agent:
            return {
                "success": False,
                "message": "ADK service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            result = await self.agent.assess_risk(trade_proposal)
            return {
                "success": result["status"] == "success",
                "risk_assessment": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Risk assessment error: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a general trading query"""
        if not self.is_initialized or not self.agent:
            return {
                "success": False,
                "message": "ADK service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            result = await self.agent.process_query(query, context)
            return {
                "success": result["status"] == "success",
                "query": query,
                "response": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    async def analyze_news_impact(self, keywords: List[str], limit: int = 10) -> Dict[str, Any]:
        """Analyze news impact on market using ADK"""
        if not self.is_initialized or not self.agent:
            return {
                "success": False,
                "message": "ADK service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            query = f"Analyze the market impact of recent news related to: {', '.join(keywords)}"
            context = {
                "keywords": keywords,
                "limit": limit,
                "analysis_type": "news_impact"
            }
            
            result = await self.agent.process_query(query, context)
            return {
                "success": result["status"] == "success",
                "news_analysis": result,
                "keywords": keywords,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"News analysis error: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "keywords": keywords,
                "timestamp": datetime.now().isoformat()
            }
    
    async def generate_trading_strategy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading strategy based on parameters"""
        if not self.is_initialized or not self.agent:
            return {
                "success": False,
                "message": "ADK service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            query = f"Generate a trading strategy based on these parameters: {parameters}"
            context = {
                "parameters": parameters,
                "analysis_type": "strategy_generation"
            }
            
            result = await self.agent.process_query(query, context)
            return {
                "success": result["status"] == "success",
                "strategy": result,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Strategy generation error: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "parameters": parameters,
                "timestamp": datetime.now().isoformat()
            }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Shutdown the ADK service"""
        try:
            if self.agent:
                await shutdown_trading_agent()
            
            self.is_initialized = False
            self.agent = None
            
            return {
                "success": True,
                "message": "ADK service shut down successfully",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"ADK service shutdown error: {str(e)}")
            return {
                "success": False,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global service instance
_adk_service: Optional[ADKService] = None


async def get_adk_service() -> ADKService:
    """Get or create the global ADK service instance"""
    global _adk_service
    
    if _adk_service is None:
        _adk_service = ADKService()
        await _adk_service.initialize()
    
    return _adk_service


async def shutdown_adk_service():
    """Shutdown the global ADK service"""
    global _adk_service
    
    if _adk_service:
        await _adk_service.shutdown()
        _adk_service = None 