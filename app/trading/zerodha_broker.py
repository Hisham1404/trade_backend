"""
Zerodha (Kite Connect) broker implementation.

This module provides integration with Zerodha's Kite Connect API for trading operations,
margin calculations, and position management.
"""

import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

from .base_broker import (
    BaseBroker, 
    BrokerType, 
    AuthType, 
    ProductType,
    BrokerCredentials,
    MarginInfo, 
    PositionInfo,
    BrokerError,
    AuthenticationError,
    InsufficientMarginError
)


class ZerodhaBroker(BaseBroker):
    """
    Zerodha Kite Connect API integration.
    
    Provides access to Zerodha's trading platform for margin calculations,
    position management, and order validation.
    """
    
    def __init__(self, credentials: BrokerCredentials, **kwargs):
        """
        Initialize Zerodha broker client.
        
        Args:
            credentials: Zerodha API credentials (api_key, api_secret, access_token)
        """
        super().__init__(credentials, **kwargs)
        
        if not credentials.api_key:
            raise ValueError("Zerodha API key is required")
        if not credentials.access_token:
            raise ValueError("Zerodha access token is required")
    
    @property
    def broker_type(self) -> BrokerType:
        return BrokerType.ZERODHA
    
    @property
    def base_url(self) -> str:
        return "https://api.kite.trade"
    
    @property
    def auth_type(self) -> AuthType:
        return AuthType.TOKEN_BASED
    
    async def authenticate(self) -> None:
        """
        Authenticate with Zerodha API.
        
        For Zerodha, we need a pre-generated access token from the login flow.
        This method validates the token by making a test API call.
        """
        try:
            # Test authentication by fetching user profile
            await self._make_authenticated_request("GET", "/user/profile")
            self._authenticated = True
            self.logger.info("Successfully authenticated with Zerodha")
        except Exception as e:
            self._authenticated = False
            raise AuthenticationError(f"Zerodha authentication failed: {str(e)}")
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Zerodha API."""
        return {
            "Authorization": f"token {self.credentials.api_key}:{self.credentials.access_token}",
            "X-Kite-Version": "3"
        }
    
    async def _make_authenticated_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an authenticated request to Zerodha API."""
        return await self._make_request(method, endpoint, **kwargs)
    
    async def get_margin_info(self, segment: Optional[str] = None) -> MarginInfo:
        """
        Get margin information from Zerodha.
        
        Args:
            segment: Market segment (equity, commodity, etc.)
            
        Returns:
            MarginInfo object with available margins
        """
        try:
            if segment:
                response = await self._make_authenticated_request("GET", f"/user/margins/{segment}")
            else:
                response = await self._make_authenticated_request("GET", "/user/margins")
            
            # Handle both single segment and all segments response
            margin_data = response.get("data", {})
            if segment:
                # Single segment response
                equity_data = margin_data
            else:
                # All segments response - use equity by default
                equity_data = margin_data.get("equity", {})
            
            return MarginInfo(
                total=float(equity_data.get("available", {}).get("cash", 0)),
                available=float(equity_data.get("available", {}).get("cash", 0)),
                utilized=float(equity_data.get("utilised", {}).get("cash", 0)),
                span=float(equity_data.get("utilised", {}).get("span", 0)),
                exposure=float(equity_data.get("utilised", {}).get("exposure", 0)),
                premium=float(equity_data.get("utilised", {}).get("option_premium", 0)),
                var_margin=float(equity_data.get("utilised", {}).get("var", 0)),
                additional_info=equity_data
            )
        except Exception as e:
            raise BrokerError(f"Failed to fetch margin info: {str(e)}")
    
    async def get_positions(self) -> List[PositionInfo]:
        """
        Get current positions from Zerodha.
        
        Returns:
            List of PositionInfo objects
        """
        try:
            response = await self._make_authenticated_request("GET", "/portfolio/positions")
            positions_data = response.get("data", {})
            
            positions = []
            for pos_type in ["day", "net"]:
                for position in positions_data.get(pos_type, []):
                    # Skip positions with zero quantity
                    if position.get("quantity", 0) == 0:
                        continue
                    
                    positions.append(PositionInfo(
                        symbol=position.get("tradingsymbol", ""),
                        exchange=position.get("exchange", ""),
                        product_type=position.get("product", ""),
                        quantity=int(position.get("quantity", 0)),
                        average_price=float(position.get("average_price", 0)),
                        market_value=float(position.get("value", 0)),
                        pnl=float(position.get("pnl", 0)),
                        realized_pnl=float(position.get("realised", 0)),
                        unrealized_pnl=float(position.get("unrealised", 0)),
                        day_change=float(position.get("day_change", 0)),
                        day_change_percent=float(position.get("day_change_percentage", 0)),
                        additional_info=position
                    ))
            
            return positions
        except Exception as e:
            raise BrokerError(f"Failed to fetch positions: {str(e)}")
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get current holdings from Zerodha.
        
        Returns:
            List of holding dictionaries
        """
        try:
            response = await self._make_authenticated_request("GET", "/portfolio/holdings")
            return response.get("data", [])
        except Exception as e:
            raise BrokerError(f"Failed to fetch holdings: {str(e)}")
    
    async def calculate_margin_requirement(self, 
                                         symbol: str, 
                                         quantity: int, 
                                         product_type: ProductType,
                                         exchange: str = "NSE") -> Dict[str, Any]:
        """
        Calculate margin requirement for a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to trade
            product_type: Product type (MIS, CNC, NRML)
            exchange: Exchange (NSE, BSE, etc.)
            
        Returns:
            Dictionary with margin details
        """
        try:
            # Prepare order basket for margin calculation
            basket = [{
                "exchange": exchange,
                "tradingsymbol": symbol,
                "transaction_type": "BUY",  # Use BUY for margin calculation
                "variety": "regular",
                "product": product_type.value,
                "order_type": "MARKET",
                "quantity": quantity
            }]
            
            response = await self._make_authenticated_request(
                "POST", 
                "/margins/basket",
                data={"basket": basket}
            )
            
            margin_data = response.get("data", {})
            initial_margin = margin_data.get("initial", {})
            final_margin = margin_data.get("final", {})
            
            return {
                "total": float(final_margin.get("total", 0)),
                "span": float(final_margin.get("span", 0)),
                "exposure": float(final_margin.get("exposure", 0)),
                "option_premium": float(final_margin.get("option_premium", 0)),
                "additional": float(final_margin.get("additional", 0)),
                "bo": float(final_margin.get("bo", 0)),
                "cash": float(final_margin.get("cash", 0)),
                "var": float(final_margin.get("var", 0)),
                "pnl": {
                    "realised": 0,
                    "unrealised": 0
                },
                "leverage": self._calculate_leverage(
                    float(final_margin.get("total", 1)),
                    quantity,
                    symbol  # Would need current price for accurate calculation
                ),
                "symbol": symbol,
                "quantity": quantity,
                "product_type": product_type.value,
                "exchange": exchange
            }
        except Exception as e:
            raise BrokerError(f"Failed to calculate margin requirement: {str(e)}")
    
    async def get_lot_size(self, symbol: str, exchange: str = "NSE") -> int:
        """
        Get lot size for a symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange
            
        Returns:
            Lot size for the symbol
        """
        try:
            # Simplified implementation with hardcoded lot sizes for common symbols
            lot_sizes = {
                "NIFTY": 50,
                "BANKNIFTY": 25,
                "RELIANCE": 250,
                "TCS": 150,
                "INFY": 300
            }
            
            # Return hardcoded lot size if available
            if symbol in lot_sizes:
                return lot_sizes[symbol]
            
            # Get instrument list (cached or from API) for other symbols
            instruments = await self._get_instruments(exchange)
            
            # Find the symbol in instruments
            for instrument in instruments:
                if instrument.get("tradingsymbol") == symbol:
                    return int(instrument.get("lot_size", 1))
            
            # Default to 1 if not found
            self.logger.warning(f"Lot size not found for {symbol}, defaulting to 1")
            return 1
            
        except Exception as e:
            self.logger.error(f"Failed to get lot size for {symbol}: {str(e)}")
            return 1
    
    async def validate_order(self, 
                           symbol: str, 
                           quantity: int, 
                           product_type: ProductType,
                           order_type: str = "MARKET",
                           exchange: str = "NSE") -> Dict[str, Any]:
        """
        Validate an order without placing it.
        
        Args:
            symbol: Trading symbol
            quantity: Quantity to trade
            product_type: Product type
            order_type: Order type (MARKET, LIMIT, etc.)
            exchange: Exchange
            
        Returns:
            Validation result
        """
        try:
            # Check margin requirement
            margin_req = await self.calculate_margin_requirement(
                symbol, quantity, product_type, exchange
            )
            
            # Get available margin
            margin_info = await self.get_margin_info("equity")
            
            # Validate sufficient margin
            required_margin = margin_req["total"]
            available_margin = margin_info.available
            
            is_valid = available_margin >= required_margin
            
            return {
                "valid": is_valid,
                "symbol": symbol,
                "quantity": quantity,
                "product_type": product_type.value,
                "exchange": exchange,
                "required_margin": required_margin,
                "available_margin": available_margin,
                "margin_shortfall": max(0, required_margin - available_margin),
                "leverage": margin_req.get("leverage", 1.0),
                "message": "Order is valid" if is_valid else f"Insufficient margin. Required: {required_margin}, Available: {available_margin}"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "message": f"Order validation failed: {str(e)}"
            }
    
    # Helper methods
    
    def _calculate_leverage(self, margin: float, quantity: int, symbol: str) -> float:
        """Calculate leverage based on margin and position value."""
        # Simplified leverage calculation
        # In production, would use current market price
        estimated_price = 1000  # Placeholder
        position_value = quantity * estimated_price
        
        if margin <= 0:
            return 1.0
        
        return min(position_value / margin, 10.0)  # Cap at 10x leverage
    
    async def _get_instruments(self, exchange: str) -> List[Dict[str, Any]]:
        """
        Get instruments list for the exchange.
        
        Note: In production, this should be cached as the file is large.
        """
        try:
            # For now, return empty list - in production would fetch from:
            # https://api.kite.trade/instruments/{exchange}
            self.logger.warning("Instruments list not implemented, returning empty list")
            return []
        except Exception as e:
            self.logger.error(f"Failed to fetch instruments: {str(e)}")
            return []
    
    async def get_user_profile(self) -> Dict[str, Any]:
        """Get user profile information."""
        try:
            response = await self._make_authenticated_request("GET", "/user/profile")
            return response.get("data", {})
        except Exception as e:
            raise BrokerError(f"Failed to fetch user profile: {str(e)}")
    
    async def get_funds(self) -> Dict[str, Any]:
        """Get funds information."""
        try:
            response = await self._make_authenticated_request("GET", "/user/margins")
            return response.get("data", {})
        except Exception as e:
            raise BrokerError(f"Failed to fetch funds: {str(e)}")


# Factory function for creating Zerodha broker instance
def create_zerodha_broker(api_key: str, access_token: str, **kwargs) -> ZerodhaBroker:
    """
    Create a Zerodha broker instance.
    
    Args:
        api_key: Zerodha API key
        access_token: Zerodha access token
        **kwargs: Additional arguments for BaseBroker
        
    Returns:
        Configured ZerodhaBroker instance
    """
    credentials = BrokerCredentials(
        api_key=api_key,
        access_token=access_token
    )
    
    return ZerodhaBroker(credentials, **kwargs) 