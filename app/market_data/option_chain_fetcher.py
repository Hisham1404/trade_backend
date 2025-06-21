"""
Option Chain Data Fetcher Module

This module provides functionality for fetching option chain data from NSE.
"""

import logging
import aiohttp
import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class NSEOptionChainFetcher:
    """Fetches option chain data from NSE"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
        }
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, symbol: str, is_equity: bool = False) -> str:
        """Generate cache key"""
        return f"{'equity' if is_equity else 'index'}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
    
    def _validate_option_chain_data(self, data: Dict) -> bool:
        """Basic validation of option chain data structure"""
        try:
            if not isinstance(data, dict):
                return False
            
            if 'records' not in data:
                return False
            
            records = data['records']
            if 'data' not in records or not isinstance(records['data'], list):
                return False
            
            return True
        except Exception:
            return False
    
    async def fetch_option_chain(self, symbol: str, is_equity: bool = False) -> Optional[Dict]:
        """Fetch option chain data for a symbol"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(symbol, is_equity)
            if cache_key in self.cache:
                logger.info(f"Returning cached data for {symbol}")
                return self.cache[cache_key]
            
            # Prepare URL
            if is_equity:
                url = f"{self.base_url}/api/option-chain-equities?symbol={symbol}"
            else:
                url = f"{self.base_url}/api/option-chain-indices?symbol={symbol}"
            
            logger.info(f"Fetching option chain data for {symbol} from {url}")
            
            # Note: Actual implementation would require proper session handling
            # and cookie management for NSE website
            # This is a placeholder that returns None
            
            # In a real implementation, you would:
            # 1. Get session cookies
            # 2. Make the API call
            # 3. Validate the response
            # 4. Cache the data
            
            logger.warning("Option chain fetching not implemented - returning None")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
            return None

class OptionChainProcessor:
    """Processes raw option chain data"""
    
    def calculate_metrics(self, option_data: List[Dict]) -> Dict[str, Any]:
        """Calculate key metrics from option chain data"""
        metrics = {
            'total_call_oi': 0,
            'total_put_oi': 0,
            'total_call_volume': 0,
            'total_put_volume': 0,
            'put_call_ratio': 0.0,
            'max_pain': 0.0,
            'iv_skew': {}
        }
        
        try:
            for strike_data in option_data:
                if 'CE' in strike_data:
                    ce = strike_data['CE']
                    metrics['total_call_oi'] += ce.get('openInterest', 0)
                    metrics['total_call_volume'] += ce.get('totalTradedVolume', 0)
                
                if 'PE' in strike_data:
                    pe = strike_data['PE']
                    metrics['total_put_oi'] += pe.get('openInterest', 0)
                    metrics['total_put_volume'] += pe.get('totalTradedVolume', 0)
            
            # Calculate PCR
            if metrics['total_call_oi'] > 0:
                metrics['put_call_ratio'] = metrics['total_put_oi'] / metrics['total_call_oi']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return metrics

class OptionChainService:
    """Main service interface for option chain operations"""
    
    def __init__(self):
        self.fetcher = NSEOptionChainFetcher()
        self.processor = OptionChainProcessor()
    
    async def get_option_chain_with_metrics(self, symbol: str, 
                                          is_equity: bool = False) -> Optional[Dict]:
        """Get option chain data with calculated metrics"""
        try:
            async with self.fetcher as fetcher:
                # Fetch raw data
                raw_data = await fetcher.fetch_option_chain(symbol, is_equity)
                
                if not raw_data:
                    return None
                
                # Process and add metrics
                if 'records' in raw_data and 'data' in raw_data['records']:
                    metrics = self.processor.calculate_metrics(
                        raw_data['records']['data']
                    )
                    raw_data['metrics'] = metrics
                
                return raw_data
                
        except Exception as e:
            logger.error(f"Error in option chain service: {str(e)}")
            return None 