"""
Market Hours Management Module

This module handles market hours, holidays, and trading session management
for Indian exchanges (NSE/BSE).
"""

import logging
from datetime import datetime, date, time, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import pytz

logger = logging.getLogger(__name__)

class MarketStatus(Enum):
    """Market status states"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PRE_OPEN = "PRE_OPEN"
    POST_CLOSE = "POST_CLOSE"
    HOLIDAY = "HOLIDAY"
    WEEKEND = "WEEKEND"

class MarketSession(Enum):
    """Trading session types"""
    REGULAR = "REGULAR"
    SPECIAL = "SPECIAL"
    MUHURAT = "MUHURAT"
    PRE_OPEN = "PRE_OPEN"
    POST_CLOSE = "POST_CLOSE"

@dataclass
class MarketInfo:
    """Market status information"""
    status: MarketStatus
    session: MarketSession
    is_trading_day: bool
    exchange: str
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    message: str = ""

class MarketHoursManager:
    """Manages market hours and trading calendar"""
    
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Market timings (IST)
        self.market_hours = {
            "NSE": {
                "pre_open_start": time(9, 0),
                "pre_open_end": time(9, 15),
                "regular_start": time(9, 15),
                "regular_end": time(15, 30),
                "post_close_start": time(15, 40),
                "post_close_end": time(16, 0)
            },
            "BSE": {
                "pre_open_start": time(9, 0),
                "pre_open_end": time(9, 15),
                "regular_start": time(9, 15),
                "regular_end": time(15, 30),
                "post_close_start": time(15, 40),
                "post_close_end": time(16, 0)
            }
        }
        
        # Basic holidays for 2024
        self.holidays_2024 = {
            date(2024, 1, 26): "Republic Day",
            date(2024, 3, 8): "Mahashivratri",
            date(2024, 3, 25): "Holi",
            date(2024, 3, 29): "Good Friday",
            date(2024, 4, 11): "Id-ul-Fitr",
            date(2024, 4, 17): "Ram Navami",
            date(2024, 5, 1): "Maharashtra Day",
            date(2024, 8, 15): "Independence Day",
            date(2024, 10, 2): "Gandhi Jayanti",
            date(2024, 11, 1): "Diwali",
            date(2024, 11, 15): "Guru Nanak Jayanti",
            date(2024, 12, 25): "Christmas"
        }
    
    def is_market_holiday(self, check_date: date, exchange: str = "NSE") -> bool:
        """Check if a date is a market holiday"""
        return check_date in self.holidays_2024
    
    def is_trading_day(self, check_date: date, exchange: str = "NSE") -> bool:
        """Check if a date is a trading day"""
        # Weekend check
        if check_date.weekday() in [5, 6]:  # Saturday, Sunday
            return False
        
        # Holiday check
        if self.is_market_holiday(check_date, exchange):
            return False
        
        return True
    
    def get_current_status(self, exchange: str = "NSE") -> MarketInfo:
        """Get current market status"""
        now_ist = datetime.now(self.ist)
        current_date = now_ist.date()
        current_time = now_ist.time()
        
        # Check if it's a trading day
        if not self.is_trading_day(current_date, exchange):
            if current_date.weekday() in [5, 6]:
                status = MarketStatus.WEEKEND
                message = "Market closed - Weekend"
            else:
                status = MarketStatus.HOLIDAY
                holiday_name = self.holidays_2024.get(current_date, "Holiday")
                message = f"Market closed - {holiday_name}"
            
            # Find next trading day
            next_day = current_date + timedelta(days=1)
            while not self.is_trading_day(next_day, exchange):
                next_day += timedelta(days=1)
            
            next_open = self.ist.localize(datetime.combine(
                next_day, 
                self.market_hours[exchange]["regular_start"]
            ))
            
            return MarketInfo(
                status=status,
                session=MarketSession.REGULAR,
                is_trading_day=False,
                exchange=exchange,
                next_open=next_open,
                message=message
            )
        
        # Trading day - check session
        hours = self.market_hours[exchange]
        
        if current_time < hours["pre_open_start"]:
            status = MarketStatus.CLOSED
            session = MarketSession.REGULAR
            message = "Market not yet open"
            next_open = self.ist.localize(datetime.combine(
                current_date, hours["pre_open_start"]
            ))
        elif current_time < hours["pre_open_end"]:
            status = MarketStatus.PRE_OPEN
            session = MarketSession.PRE_OPEN
            message = "Pre-open session"
            next_close = self.ist.localize(datetime.combine(
                current_date, hours["regular_end"]
            ))
        elif current_time < hours["regular_end"]:
            status = MarketStatus.OPEN
            session = MarketSession.REGULAR
            message = "Market open"
            next_close = self.ist.localize(datetime.combine(
                current_date, hours["regular_end"]
            ))
        elif current_time < hours["post_close_start"]:
            status = MarketStatus.CLOSED
            session = MarketSession.REGULAR
            message = "Market closed"
            next_open = self._get_next_open(current_date, exchange)
        elif current_time < hours["post_close_end"]:
            status = MarketStatus.POST_CLOSE
            session = MarketSession.POST_CLOSE
            message = "Post-close session"
            next_open = self._get_next_open(current_date, exchange)
        else:
            status = MarketStatus.CLOSED
            session = MarketSession.REGULAR
            message = "Market closed"
            next_open = self._get_next_open(current_date, exchange)
        
        return MarketInfo(
            status=status,
            session=session,
            is_trading_day=True,
            exchange=exchange,
            next_open=next_open if status == MarketStatus.CLOSED else None,
            next_close=next_close if status in [MarketStatus.OPEN, MarketStatus.PRE_OPEN] else None,
            message=message
        )
    
    def _get_next_open(self, current_date: date, exchange: str) -> datetime:
        """Get next market open time"""
        next_day = current_date + timedelta(days=1)
        while not self.is_trading_day(next_day, exchange):
            next_day += timedelta(days=1)
        
        return self.ist.localize(datetime.combine(
            next_day, 
            self.market_hours[exchange]["regular_start"]
        ))
    
    def should_fetch_data(self, exchange: str = "NSE") -> bool:
        """Check if we should fetch option data now"""
        status = self.get_current_status(exchange)
        return status.status in [MarketStatus.OPEN, MarketStatus.PRE_OPEN]
    
    def get_next_fetch_time(self, exchange: str = "NSE") -> Optional[datetime]:
        """Get next recommended fetch time"""
        status = self.get_current_status(exchange)
        
        if status.status == MarketStatus.OPEN:
            # Fetch every 5 minutes during market hours
            return datetime.now(self.ist) + timedelta(minutes=5)
        elif status.next_open:
            # Next fetch at market open
            return status.next_open
        else:
            return None
    
    def get_market_calendar(self, start_date: date, end_date: date, 
                          exchange: str = "NSE") -> List[Dict]:
        """Get market calendar for date range"""
        calendar = []
        current = start_date
        
        while current <= end_date:
            is_trading = self.is_trading_day(current, exchange)
            is_holiday = self.is_market_holiday(current, exchange)
            
            day_info = {
                "date": current.isoformat(),
                "is_trading_day": is_trading,
                "is_holiday": is_holiday,
                "day_name": current.strftime("%A")
            }
            
            if is_holiday and current in self.holidays_2024:
                day_info["holiday_name"] = self.holidays_2024[current]
            
            calendar.append(day_info)
            current += timedelta(days=1)
        
        return calendar

# Convenience functions
def get_market_status(exchange: str = "NSE") -> MarketInfo:
    """Get current market status"""
    manager = MarketHoursManager()
    return manager.get_current_status(exchange)

def is_market_open(exchange: str = "NSE") -> bool:
    """Check if market is currently open"""
    status = get_market_status(exchange)
    return status.status == MarketStatus.OPEN

def should_fetch_option_data(exchange: str = "NSE") -> bool:
    """Check if we should fetch option data now"""
    manager = MarketHoursManager()
    return manager.should_fetch_data(exchange) 