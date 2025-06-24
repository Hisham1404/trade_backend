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
