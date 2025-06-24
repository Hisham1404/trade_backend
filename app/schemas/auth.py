"""
Authentication schemas for request validation and response formatting.

Defines Pydantic models for user registration, login, token management,
and authentication-related API responses.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, validator
from enum import Enum


class SubscriptionTier(str, Enum):
    """User subscription tiers."""
    BASIC = "basic"
    PREMIUM = "premium"
    PROFESSIONAL = "professional"


class UserRegistrationRequest(BaseModel):
    """Request schema for user registration."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, max_length=128, description="User's password")
    subscription_tier: SubscriptionTier = Field(
        default=SubscriptionTier.BASIC, 
        description="User's subscription tier"
    )
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError('Password must contain uppercase, lowercase, and numeric characters')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123",
                "subscription_tier": "basic"
            }
        }


class UserLoginRequest(BaseModel):
    """Request schema for user login."""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")
    remember_me: bool = Field(default=False, description="Extended session duration")
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123",
                "remember_me": False
            }
        }


class TokenResponse(BaseModel):
    """Response schema for token operations."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "refresh_token": "refresh_token_here",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class RefreshTokenRequest(BaseModel):
    """Request schema for token refresh."""
    refresh_token: str = Field(..., description="Valid refresh token")
    
    class Config:
        schema_extra = {
            "example": {
                "refresh_token": "refresh_token_here"
            }
        }


class UserResponse(BaseModel):
    """Response schema for user information."""
    id: int = Field(..., description="User's unique identifier")
    email: str = Field(..., description="User's email address")
    api_key: Optional[str] = Field(None, description="User's API key")
    subscription_tier: str = Field(..., description="User's subscription tier")
    is_active: bool = Field(default=True, description="Whether user account is active")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 123,
                "email": "user@example.com",
                "api_key": "api_key_abc123",
                "subscription_tier": "basic",
                "is_active": True,
                "created_at": "2024-01-01T12:00:00Z",
                "last_active": "2024-01-15T14:30:00Z"
            }
        }


class PasswordChangeRequest(BaseModel):
    """Request schema for password change."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    
    @validator('new_password')
    def validate_new_password_strength(cls, v):
        """Validate new password meets security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError('Password must contain uppercase, lowercase, and numeric characters')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "current_password": "OldPass123",
                "new_password": "NewSecurePass456"
            }
        }


class PasswordResetRequest(BaseModel):
    """Request schema for password reset."""
    email: EmailStr = Field(..., description="User's email address")
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }


class PasswordResetConfirmRequest(BaseModel):
    """Request schema for password reset confirmation."""
    reset_token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    
    @validator('new_password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError('Password must contain uppercase, lowercase, and numeric characters')
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "reset_token": "reset_token_here",
                "new_password": "NewSecurePass789"
            }
        }


class APIKeyResponse(BaseModel):
    """Response schema for API key operations."""
    api_key: str = Field(..., description="Generated API key")
    created_at: datetime = Field(..., description="API key creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="API key expiration timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "api_key": "api_key_abc123def456",
                "created_at": "2024-01-01T12:00:00Z",
                "expires_at": None
            }
        }


class UserSessionInfo(BaseModel):
    """Information about user's current session."""
    user_id: int = Field(..., description="User's ID")
    email: str = Field(..., description="User's email")
    subscription_tier: str = Field(..., description="User's subscription tier")
    session_start: datetime = Field(..., description="Session start time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    ip_address: str = Field(..., description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "email": "user@example.com",
                "subscription_tier": "basic",
                "session_start": "2024-01-15T10:00:00Z",
                "last_activity": "2024-01-15T14:30:00Z",
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0..."
            }
        }


class AuthErrorResponse(BaseModel):
    """Error response schema for authentication failures."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "authentication_failed",
                "message": "Invalid email or password",
                "details": {
                    "attempts_remaining": 3,
                    "lockout_duration": 300
                }
            }
        }


class RateLimitResponse(BaseModel):
    """Response schema for rate limit information."""
    limit: int = Field(..., description="Request limit per window")
    remaining: int = Field(..., description="Remaining requests in current window")
    reset: int = Field(..., description="Window reset timestamp")
    retry_after: Optional[int] = Field(None, description="Seconds to wait before retry")
    
    class Config:
        schema_extra = {
            "example": {
                "limit": 60,
                "remaining": 45,
                "reset": 1640995200,
                "retry_after": None
            }
        }


class LogoutRequest(BaseModel):
    """Request schema for user logout."""
    revoke_all_sessions: bool = Field(
        default=False, 
        description="Whether to revoke all user sessions"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "revoke_all_sessions": False
            }
        } 