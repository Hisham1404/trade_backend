"""
Broker Manager for unified broker API integration.

This module provides a unified interface for managing multiple broker connections
and coordinating position sizing, margin calculations, and order validation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from .base_broker import (
    BaseBroker, 
    BrokerType, 
    ProductType,
    BrokerCredentials,
    MarginInfo, 
    PositionInfo,
    BrokerError,
    AuthenticationError
)
from .zerodha_broker import ZerodhaBroker, create_zerodha_broker

logger = logging.getLogger(__name__)


class BrokerStatus(Enum):
    """Broker connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"


class BrokerManager:
    """
    Unified broker management system.
    
    Manages multiple broker connections and provides a unified interface
    for trading operations, margin calculations, and position management.
    """
    
    def __init__(self):
        """Initialize the broker manager."""
        self.brokers: Dict[str, BaseBroker] = {}
        self.broker_status: Dict[str, BrokerStatus] = {}
        self.broker_configs: Dict[str, Dict[str, Any]] = {}
        self.primary_broker: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
        # Health check settings
        self.health_check_interval = 300  # 5 minutes
        self.health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def add_broker(self, 
                        broker_id: str, 
                        broker_type: BrokerType, 
                        credentials: BrokerCredentials,
                        config: Optional[Dict[str, Any]] = None,
                        set_as_primary: bool = False) -> None:
        """
        Add a broker to the manager.
        
        Args:
            broker_id: Unique identifier for the broker
            broker_type: Type of broker (Zerodha, Upstox, etc.)
            credentials: Broker authentication credentials
            config: Additional broker configuration
            set_as_primary: Whether to set this as the primary broker
        """
        try:
            # Create broker instance based on type
            if broker_type == BrokerType.ZERODHA:
                broker = ZerodhaBroker(credentials, **(config or {}))
            else:
                raise ValueError(f"Unsupported broker type: {broker_type}")
            
            # Store broker and configuration
            self.brokers[broker_id] = broker
            self.broker_configs[broker_id] = config or {}
            self.broker_status[broker_id] = BrokerStatus.DISCONNECTED
            
            # Set as primary if requested or if it's the first broker
            if set_as_primary or self.primary_broker is None:
                self.primary_broker = broker_id
            
            self.logger.info(f"Added broker {broker_id} ({broker_type.value})")
            
        except Exception as e:
            self.logger.error(f"Failed to add broker {broker_id}: {str(e)}")
            raise BrokerError(f"Failed to add broker: {str(e)}")
    
    async def connect_broker(self, broker_id: str) -> bool:
        """
        Connect and authenticate a specific broker.
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            True if connection successful, False otherwise
        """
        if broker_id not in self.brokers:
            raise ValueError(f"Broker {broker_id} not found")
        
        try:
            self.broker_status[broker_id] = BrokerStatus.AUTHENTICATING
            broker = self.brokers[broker_id]
            
            # Initialize and authenticate
            await broker.initialize()
            
            self.broker_status[broker_id] = BrokerStatus.CONNECTED
            self.logger.info(f"Successfully connected to broker {broker_id}")
            return True
            
        except Exception as e:
            self.broker_status[broker_id] = BrokerStatus.ERROR
            self.logger.error(f"Failed to connect to broker {broker_id}: {str(e)}")
            return False
    
    async def connect_all_brokers(self) -> Dict[str, bool]:
        """
        Connect to all configured brokers.
        
        Returns:
            Dictionary mapping broker_id to connection success status
        """
        results = {}
        
        # Connect to brokers concurrently
        tasks = [
            self.connect_broker(broker_id) 
            for broker_id in self.brokers.keys()
        ]
        
        if tasks:
            connection_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for broker_id, result in zip(self.brokers.keys(), connection_results):
                if isinstance(result, Exception):
                    results[broker_id] = False
                    self.logger.error(f"Connection failed for {broker_id}: {str(result)}")
                else:
                    results[broker_id] = result
        
        return results
    
    async def disconnect_broker(self, broker_id: str) -> None:
        """Disconnect a specific broker."""
        if broker_id not in self.brokers:
            return
        
        try:
            await self.brokers[broker_id].close()
            self.broker_status[broker_id] = BrokerStatus.DISCONNECTED
            self.logger.info(f"Disconnected from broker {broker_id}")
        except Exception as e:
            self.logger.error(f"Error disconnecting from broker {broker_id}: {str(e)}")
    
    async def disconnect_all_brokers(self) -> None:
        """Disconnect from all brokers."""
        tasks = [
            self.disconnect_broker(broker_id) 
            for broker_id in self.brokers.keys()
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_connected_brokers(self) -> List[str]:
        """Get list of connected broker IDs."""
        return [
            broker_id for broker_id, status in self.broker_status.items()
            if status == BrokerStatus.CONNECTED
        ]
    
    def get_primary_broker(self) -> Optional[BaseBroker]:
        """Get the primary broker instance."""
        if self.primary_broker and self.primary_broker in self.brokers:
            if self.broker_status[self.primary_broker] == BrokerStatus.CONNECTED:
                return self.brokers[self.primary_broker]
        return None
    
    def set_primary_broker(self, broker_id: str) -> None:
        """Set the primary broker."""
        if broker_id not in self.brokers:
            raise ValueError(f"Broker {broker_id} not found")
        
        self.primary_broker = broker_id
        self.logger.info(f"Set {broker_id} as primary broker")
    
    async def get_margin_info(self, 
                             broker_id: Optional[str] = None, 
                             segment: Optional[str] = None) -> MarginInfo:
        """
        Get margin information from a broker.
        
        Args:
            broker_id: Specific broker to query (uses primary if None)
            segment: Market segment
            
        Returns:
            MarginInfo object
        """
        broker = self._get_broker(broker_id)
        return await broker.get_margin_info(segment)
    
    async def get_positions(self, broker_id: Optional[str] = None) -> List[PositionInfo]:
        """
        Get current positions from a broker.
        
        Args:
            broker_id: Specific broker to query (uses primary if None)
            
        Returns:
            List of PositionInfo objects
        """
        broker = self._get_broker(broker_id)
        return await broker.get_positions()
    
    async def calculate_margin_requirement(self, 
                                         symbol: str, 
                                         quantity: int, 
                                         product_type: ProductType,
                                         exchange: str = "NSE",
                                         broker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate margin requirement for a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to trade
            product_type: Product type
            exchange: Exchange
            broker_id: Specific broker to use (uses primary if None)
            
        Returns:
            Dictionary with margin details
        """
        broker = self._get_broker(broker_id)
        return await broker.calculate_margin_requirement(symbol, quantity, product_type, exchange)
    
    async def validate_order(self, 
                           symbol: str, 
                           quantity: int, 
                           product_type: ProductType,
                           order_type: str = "MARKET",
                           exchange: str = "NSE",
                           broker_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate an order without placing it.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to trade
            product_type: Product type
            order_type: Order type
            exchange: Exchange
            broker_id: Specific broker to use (uses primary if None)
            
        Returns:
            Validation result
        """
        broker = self._get_broker(broker_id)
        return await broker.validate_order(symbol, quantity, product_type, order_type, exchange)
    
    async def get_lot_size(self, 
                          symbol: str, 
                          exchange: str = "NSE",
                          broker_id: Optional[str] = None) -> int:
        """
        Get lot size for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange
            broker_id: Specific broker to use (uses primary if None)
            
        Returns:
            Lot size
        """
        broker = self._get_broker(broker_id)
        return await broker.get_lot_size(symbol, exchange)
    
    def _get_broker(self, broker_id: Optional[str] = None) -> BaseBroker:
        """Get a broker instance."""
        if broker_id:
            if broker_id not in self.brokers:
                raise ValueError(f"Broker {broker_id} not found")
            if self.broker_status[broker_id] != BrokerStatus.CONNECTED:
                raise BrokerError(f"Broker {broker_id} is not connected")
            return self.brokers[broker_id]
        else:
            # Use primary broker
            primary = self.get_primary_broker()
            if not primary:
                raise BrokerError("No primary broker available")
            return primary
    
    async def health_check_all_brokers(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all brokers."""
        results = {}
        
        tasks = [
            self._health_check_broker(broker_id)
            for broker_id in self.brokers.keys()
        ]
        
        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for broker_id, result in zip(self.brokers.keys(), health_results):
                if isinstance(result, Exception):
                    results[broker_id] = {
                        "status": "error",
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    results[broker_id] = result
        
        return results
    
    async def _health_check_broker(self, broker_id: str) -> Dict[str, Any]:
        """Perform health check on a specific broker."""
        if broker_id not in self.brokers:
            return {"status": "not_found", "timestamp": datetime.now().isoformat()}
        
        if self.broker_status[broker_id] != BrokerStatus.CONNECTED:
            return {"status": "disconnected", "timestamp": datetime.now().isoformat()}
        
        try:
            broker = self.brokers[broker_id]
            return await broker.health_check()
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._running:
            return
        
        self._running = True
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        self.logger.info("Started broker health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop periodic health monitoring."""
        self._running = False
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
        
        self.logger.info("Stopped broker health monitoring")
    
    async def _health_monitor_loop(self) -> None:
        """Health monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if not self._running:
                    break
                
                # Perform health check
                health_results = await self.health_check_all_brokers()
                
                # Update broker status based on health check
                for broker_id, health in health_results.items():
                    if health.get("status") == "healthy":
                        self.broker_status[broker_id] = BrokerStatus.CONNECTED
                    elif health.get("status") in ["unhealthy", "error"]:
                        self.broker_status[broker_id] = BrokerStatus.ERROR
                        self.logger.warning(f"Broker {broker_id} health check failed: {health.get('error', 'Unknown error')}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {str(e)}")
    
    def get_broker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all brokers."""
        return {
            broker_id: {
                "status": status.value,
                "type": broker.broker_type.value,
                "is_primary": broker_id == self.primary_broker,
                "authenticated": broker.is_authenticated() if status == BrokerStatus.CONNECTED else False
            }
            for broker_id, (broker, status) in 
            zip(self.brokers.keys(), zip(self.brokers.values(), self.broker_status.values()))
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_all_brokers()
        await self.start_health_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_health_monitoring()
        await self.disconnect_all_brokers()


# Factory functions for creating broker manager with pre-configured brokers

def create_broker_manager_with_zerodha(api_key: str, access_token: str) -> BrokerManager:
    """
    Create a broker manager with Zerodha broker configured.
    
    Args:
        api_key: Zerodha API key
        access_token: Zerodha access token
        
    Returns:
        Configured BrokerManager instance
    """
    manager = BrokerManager()
    
    credentials = BrokerCredentials(
        api_key=api_key,
        access_token=access_token
    )
    
    # Add Zerodha broker as primary
    asyncio.create_task(manager.add_broker(
        broker_id="zerodha",
        broker_type=BrokerType.ZERODHA,
        credentials=credentials,
        set_as_primary=True
    ))
    
    return manager 