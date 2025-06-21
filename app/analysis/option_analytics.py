"""
Option Analytics Calculation Engine

This module provides comprehensive mathematical models and calculations for option chain analysis,
including PCR calculation, max pain point determination, implied volatility calculations,
Greeks computation, and statistical analysis functions optimized for real-time processing.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import brentq
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import warnings

# Suppress scipy warnings for optimization
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

@dataclass
class OptionGreeks:
    """Container for option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class AnalyticsResult:
    """Container for analytics calculation results"""
    pcr_oi: float
    pcr_volume: float
    max_pain: float
    iv_percentile: float
    support_levels: List[float]
    resistance_levels: List[float]
    trend_direction: str
    volatility_regime: str
    anomaly_score: float
    calculated_at: datetime

@dataclass
class TrendAnalysis:
    """Container for trend analysis results"""
    direction: str  # 'bullish', 'bearish', 'sideways'
    strength: float  # 0.0 to 1.0
    momentum: float
    confidence: float

class BlackScholesCalculator:
    """Black-Scholes model for option pricing and Greeks calculation"""
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholesCalculator._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def option_price(S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: str = 'call') -> float:
        """
        Calculate option price using Black-Scholes formula
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'
        """
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
        d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator._d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str = 'call') -> OptionGreeks:
        """Calculate option Greeks"""
        if T <= 0 or sigma <= 0:
            return OptionGreeks(0.0, 0.0, 0.0, 0.0, 0.0)
        
        d1 = BlackScholesCalculator._d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator._d2(S, K, T, r, sigma)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:  # put
            delta = -norm.cdf(-d1)
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:  # put
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return OptionGreeks(delta, gamma, theta, vega, rho)

    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = 'call') -> Optional[float]:
        """
        Calculate implied volatility using Brent's method
        
        Returns None if no solution is found
        """
        if T <= 0:
            return None
        
        def objective(sigma):
            try:
                bs_price = BlackScholesCalculator.option_price(S, K, T, r, sigma, option_type)
                return bs_price - market_price
            except:
                return float('inf')
        
        try:
            # Try to find implied volatility between 0.01% and 1000%
            iv = brentq(objective, 0.0001, 10.0, xtol=1e-6, maxiter=100)
            return iv if 0.0001 <= iv <= 10.0 else None
        except (ValueError, RuntimeError):
            return None

class PCRCalculator:
    """Put-Call Ratio calculations"""
    
    @staticmethod
    def calculate_pcr_oi(option_data: List[Dict]) -> float:
        """Calculate Put-Call Ratio based on Open Interest"""
        total_put_oi = 0
        total_call_oi = 0
        
        for strike_data in option_data:
            if 'PE' in strike_data and strike_data['PE']:
                total_put_oi += strike_data['PE'].get('openInterest', 0)
            
            if 'CE' in strike_data and strike_data['CE']:
                total_call_oi += strike_data['CE'].get('openInterest', 0)
        
        return total_put_oi / total_call_oi if total_call_oi > 0 else 0.0
    
    @staticmethod
    def calculate_pcr_volume(option_data: List[Dict]) -> float:
        """Calculate Put-Call Ratio based on Volume"""
        total_put_volume = 0
        total_call_volume = 0
        
        for strike_data in option_data:
            if 'PE' in strike_data and strike_data['PE']:
                total_put_volume += strike_data['PE'].get('totalTradedVolume', 0)
            
            if 'CE' in strike_data and strike_data['CE']:
                total_call_volume += strike_data['CE'].get('totalTradedVolume', 0)
        
        return total_put_volume / total_call_volume if total_call_volume > 0 else 0.0

class MaxPainCalculator:
    """Max Pain point calculation"""
    
    @staticmethod
    def calculate_max_pain(option_data: List[Dict], spot_price: float = None) -> float:
        """
        Calculate Max Pain point - the strike price where maximum number of options expire worthless
        """
        strike_pain = {}
        
        # Get all strike prices
        strikes = set()
        for strike_data in option_data:
            if 'strikePrice' in strike_data:
                strikes.add(float(strike_data['strikePrice']))
        
        if not strikes:
            return spot_price or 0.0
        
        strikes = sorted(strikes)
        
        # Calculate pain for each potential expiration price
        for expiry_price in strikes:
            total_pain = 0
            
            for strike_data in option_data:
                strike = float(strike_data.get('strikePrice', 0))
                
                # Call option pain
                if 'CE' in strike_data and strike_data['CE']:
                    call_oi = strike_data['CE'].get('openInterest', 0)
                    if expiry_price > strike:
                        total_pain += (expiry_price - strike) * call_oi
                
                # Put option pain
                if 'PE' in strike_data and strike_data['PE']:
                    put_oi = strike_data['PE'].get('openInterest', 0)
                    if expiry_price < strike:
                        total_pain += (strike - expiry_price) * put_oi
            
            strike_pain[expiry_price] = total_pain
        
        # Return strike with minimum pain
        return min(strike_pain.items(), key=lambda x: x[1])[0] if strike_pain else (spot_price or 0.0)

class SupportResistanceCalculator:
    """Support and Resistance level calculation"""
    
    @staticmethod
    def calculate_levels(option_data: List[Dict], num_levels: int = 3) -> Tuple[List[float], List[float]]:
        """
        Calculate support and resistance levels based on option activity
        
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        strike_activity = {}
        
        for strike_data in option_data:
            strike = float(strike_data.get('strikePrice', 0))
            
            total_oi = 0
            total_volume = 0
            
            # Sum call and put activity
            for option_type in ['CE', 'PE']:
                if option_type in strike_data and strike_data[option_type]:
                    total_oi += strike_data[option_type].get('openInterest', 0)
                    total_volume += strike_data[option_type].get('totalTradedVolume', 0)
            
            # Weight by both OI and volume
            activity_score = total_oi * 0.7 + total_volume * 0.3
            strike_activity[strike] = activity_score
        
        if not strike_activity:
            return [], []
        
        # Sort by activity score
        sorted_strikes = sorted(strike_activity.items(), key=lambda x: x[1], reverse=True)
        
        # Get top strikes as potential support/resistance
        top_strikes = [strike for strike, _ in sorted_strikes[:num_levels * 2]]
        top_strikes.sort()
        
        # Simple logic: lower strikes are support, higher are resistance
        mid_point = len(top_strikes) // 2
        support_levels = top_strikes[:mid_point][-num_levels:]  # Bottom half, take last N
        resistance_levels = top_strikes[mid_point:][:num_levels]  # Top half, take first N
        
        return support_levels, resistance_levels

class VolatilityAnalyzer:
    """Implied Volatility analysis"""
    
    @staticmethod
    def calculate_iv_percentile(current_iv: float, historical_iv: List[float]) -> float:
        """Calculate IV percentile ranking"""
        if not historical_iv or current_iv is None:
            return 50.0
        
        historical_iv = [iv for iv in historical_iv if iv is not None]
        if not historical_iv:
            return 50.0
        
        below_current = sum(1 for iv in historical_iv if iv < current_iv)
        return (below_current / len(historical_iv)) * 100
    
    @staticmethod
    def detect_volatility_regime(current_iv: float, historical_iv: List[float]) -> str:
        """Detect current volatility regime"""
        if not historical_iv or current_iv is None:
            return "unknown"
        
        historical_iv = [iv for iv in historical_iv if iv is not None]
        if len(historical_iv) < 10:
            return "insufficient_data"
        
        iv_mean = np.mean(historical_iv)
        iv_std = np.std(historical_iv)
        
        if current_iv > iv_mean + iv_std:
            return "high_volatility"
        elif current_iv < iv_mean - iv_std:
            return "low_volatility"
        else:
            return "normal_volatility"

class TrendDetector:
    """Statistical trend detection and analysis"""
    
    @staticmethod
    def detect_trend(pcr_history: List[float], window: int = 20) -> TrendAnalysis:
        """
        Detect trend based on PCR history using statistical methods
        """
        if len(pcr_history) < window:
            return TrendAnalysis("insufficient_data", 0.0, 0.0, 0.0)
        
        recent_data = np.array(pcr_history[-window:])
        
        # Linear regression for trend direction
        x = np.arange(len(recent_data))
        slope = np.polyfit(x, recent_data, 1)[0]
        
        # Calculate R-squared for trend strength
        y_pred = np.poly1d(np.polyfit(x, recent_data, 1))(x)
        ss_res = np.sum((recent_data - y_pred) ** 2)
        ss_tot = np.sum((recent_data - np.mean(recent_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Momentum calculation (rate of change)
        momentum = (recent_data[-1] - recent_data[0]) / len(recent_data)
        
        # Determine direction
        if slope > 0.001:
            direction = "bullish"  # PCR increasing (more puts)
        elif slope < -0.001:
            direction = "bearish"  # PCR decreasing (more calls)
        else:
            direction = "sideways"
        
        # Confidence based on R-squared and data consistency
        confidence = min(r_squared, 0.95)
        
        return TrendAnalysis(direction, abs(slope), momentum, confidence)

class AnomalyDetector:
    """Anomaly detection for unusual option activity"""
    
    @staticmethod
    def detect_anomalies(option_data: List[Dict], historical_stats: Dict = None) -> float:
        """
        Detect anomalies in option data and return anomaly score (0.0 to 1.0)
        """
        anomaly_indicators = []
        
        # Calculate current metrics
        total_volume = 0
        total_oi = 0
        unusual_activity_count = 0
        
        for strike_data in option_data:
            for option_type in ['CE', 'PE']:
                if option_type in strike_data and strike_data[option_type]:
                    volume = strike_data[option_type].get('totalTradedVolume', 0)
                    oi = strike_data[option_type].get('openInterest', 0)
                    
                    total_volume += volume
                    total_oi += oi
                    
                    # Check for unusual volume-to-OI ratio
                    if oi > 0 and volume / oi > 5:  # High volume relative to OI
                        unusual_activity_count += 1
        
        # Volume anomaly detection
        if historical_stats and 'avg_volume' in historical_stats:
            avg_volume = historical_stats['avg_volume']
            volume_ratio = total_volume / avg_volume if avg_volume > 0 else 1.0
            if volume_ratio > 3.0:  # 3x normal volume
                anomaly_indicators.append(min(volume_ratio / 10, 1.0))
        
        # Unusual activity anomaly
        if len(option_data) > 0:
            unusual_ratio = unusual_activity_count / len(option_data)
            if unusual_ratio > 0.2:  # More than 20% strikes showing unusual activity
                anomaly_indicators.append(unusual_ratio)
        
        # Calculate overall anomaly score
        if anomaly_indicators:
            return min(np.mean(anomaly_indicators), 1.0)
        
        return 0.0

class OptionAnalyticsEngine:
    """
    Main analytics engine that orchestrates all calculations
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize calculators
        self.bs_calculator = BlackScholesCalculator()
        self.pcr_calculator = PCRCalculator()
        self.max_pain_calculator = MaxPainCalculator()
        self.sr_calculator = SupportResistanceCalculator()
        self.volatility_analyzer = VolatilityAnalyzer()
        self.trend_detector = TrendDetector()
        self.anomaly_detector = AnomalyDetector()
    
    async def calculate_comprehensive_analytics(
        self,
        option_data: List[Dict],
        spot_price: float,
        days_to_expiry: float,
        historical_data: Dict = None
    ) -> AnalyticsResult:
        """
        Calculate comprehensive option analytics
        
        Args:
            option_data: List of option data dictionaries
            spot_price: Current underlying price
            days_to_expiry: Days to expiration
            historical_data: Historical data for percentile calculations
        
        Returns:
            AnalyticsResult with all calculated metrics
        """
        try:
            # Run calculations in parallel for performance
            loop = asyncio.get_event_loop()
            
            # Basic calculations
            pcr_oi_task = loop.run_in_executor(
                self.executor, self.pcr_calculator.calculate_pcr_oi, option_data
            )
            pcr_volume_task = loop.run_in_executor(
                self.executor, self.pcr_calculator.calculate_pcr_volume, option_data
            )
            max_pain_task = loop.run_in_executor(
                self.executor, self.max_pain_calculator.calculate_max_pain, option_data, spot_price
            )
            
            # Support/Resistance calculation
            sr_task = loop.run_in_executor(
                self.executor, self.sr_calculator.calculate_levels, option_data
            )
            
            # Wait for basic calculations
            pcr_oi = await pcr_oi_task
            pcr_volume = await pcr_volume_task
            max_pain = await max_pain_task
            support_levels, resistance_levels = await sr_task
            
            # Advanced analytics with historical context
            iv_percentile = 50.0  # Default
            trend_direction = "sideways"
            volatility_regime = "unknown"
            anomaly_score = 0.0
            
            if historical_data:
                # IV Percentile calculation
                current_iv = self._extract_average_iv(option_data)
                if current_iv and 'historical_iv' in historical_data:
                    iv_percentile = self.volatility_analyzer.calculate_iv_percentile(
                        current_iv, historical_data['historical_iv']
                    )
                    volatility_regime = self.volatility_analyzer.detect_volatility_regime(
                        current_iv, historical_data['historical_iv']
                    )
                
                # Trend detection
                if 'pcr_history' in historical_data:
                    trend_analysis = self.trend_detector.detect_trend(
                        historical_data['pcr_history'] + [pcr_oi]
                    )
                    trend_direction = trend_analysis.direction
                
                # Anomaly detection
                anomaly_score = self.anomaly_detector.detect_anomalies(
                    option_data, historical_data.get('volume_stats')
                )
            
            return AnalyticsResult(
                pcr_oi=pcr_oi,
                pcr_volume=pcr_volume,
                max_pain=max_pain,
                iv_percentile=iv_percentile,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_direction=trend_direction,
                volatility_regime=volatility_regime,
                anomaly_score=anomaly_score,
                calculated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analytics calculation: {str(e)}")
            # Return default values
            return AnalyticsResult(
                pcr_oi=0.0,
                pcr_volume=0.0,
                max_pain=spot_price,
                iv_percentile=50.0,
                support_levels=[],
                resistance_levels=[],
                trend_direction="unknown",
                volatility_regime="unknown",
                anomaly_score=0.0,
                calculated_at=datetime.now()
            )
    
    def calculate_option_greeks(
        self,
        spot_price: float,
        strike_price: float,
        days_to_expiry: float,
        implied_volatility: float,
        option_type: str = 'call'
    ) -> OptionGreeks:
        """Calculate option Greeks for a single option"""
        time_to_expiry = days_to_expiry / 365.0
        
        return self.bs_calculator.calculate_greeks(
            spot_price, strike_price, time_to_expiry, 
            self.risk_free_rate, implied_volatility, option_type
        )
    
    def calculate_implied_volatility(
        self,
        market_price: float,
        spot_price: float,
        strike_price: float,
        days_to_expiry: float,
        option_type: str = 'call'
    ) -> Optional[float]:
        """Calculate implied volatility for a single option"""
        time_to_expiry = days_to_expiry / 365.0
        
        return self.bs_calculator.implied_volatility(
            market_price, spot_price, strike_price, time_to_expiry,
            self.risk_free_rate, option_type
        )
    
    def _extract_average_iv(self, option_data: List[Dict]) -> Optional[float]:
        """Extract average implied volatility from option data"""
        iv_values = []
        
        for strike_data in option_data:
            for option_type in ['CE', 'PE']:
                if option_type in strike_data and strike_data[option_type]:
                    iv = strike_data[option_type].get('impliedVolatility')
                    if iv and iv > 0:
                        iv_values.append(iv)
        
        return np.mean(iv_values) if iv_values else None
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

# Convenience functions for quick calculations
def calculate_pcr(option_data: List[Dict]) -> Dict[str, float]:
    """Quick PCR calculation"""
    calculator = PCRCalculator()
    return {
        'pcr_oi': calculator.calculate_pcr_oi(option_data),
        'pcr_volume': calculator.calculate_pcr_volume(option_data)
    }

def calculate_max_pain(option_data: List[Dict], spot_price: float = None) -> float:
    """Quick Max Pain calculation"""
    calculator = MaxPainCalculator()
    return calculator.calculate_max_pain(option_data, spot_price)

def calculate_support_resistance(option_data: List[Dict]) -> Dict[str, List[float]]:
    """Quick Support/Resistance calculation"""
    calculator = SupportResistanceCalculator()
    support, resistance = calculator.calculate_levels(option_data)
    return {
        'support_levels': support,
        'resistance_levels': resistance
    } 