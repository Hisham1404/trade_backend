# Rate Limiter module
import time
from enum import Enum
from collections import defaultdict, deque

class RateLimitAlgorithm(Enum):
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"

class RateLimiter:
    def __init__(self, algorithm=RateLimitAlgorithm.SLIDING_WINDOW, requests_per_minute=60, burst_size=None):
        self.algorithm = algorithm
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self._sliding_windows = defaultdict(deque)
        
    def check_rate_limit(self, client_id):
        current_time = time.time()
        window = self._sliding_windows[client_id]
        
        # Remove old requests
        while window and window[0] < current_time - 60:
            window.popleft()
        
        # Check limit
        if len(window) >= self.requests_per_minute:
            return False
        
        window.append(current_time)
        return True

def check_rate_limit(client_id, limiter=None):
    if limiter is None:
        limiter = RateLimiter()
    return limiter.check_rate_limit(client_id)

def get_client_identifier(request):
    """Extract client identifier from request for rate limiting"""
    # Get client IP from request
    client_ip = request.client.host if request.client else "unknown"
    
    # Check for forwarded IP (if behind proxy)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    # Try to get user ID if authenticated
    user_id = getattr(request.state, 'user_id', None)
    if user_id:
        return f"user:{user_id}"
    
    # Fall back to IP address
    return f"ip:{client_ip}"
