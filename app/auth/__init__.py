# -*- coding: utf-8 -*-
"""
Authentication module for the Trading Intelligence Agent.

This module provides comprehensive authentication functionality.
"""

from app.auth.jwt_handler import JWTHandler
from app.auth.password_manager import PasswordManager
from app.auth.rate_limiter import RateLimiter
from app.auth.dependencies import get_current_user

# Export key functions and classes
__all__ = [
    "JWTHandler",
    "PasswordManager", 
    "RateLimiter",
    "get_current_user"
]
