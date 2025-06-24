"""
JWT (JSON Web Token) Handler for secure authentication.

Provides functionality for creating, verifying, and managing JWT tokens
for user authentication and session management.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models.user import User
from app.core.config import settings

# JWT Configuration
SECRET_KEY = getattr(settings, 'SECRET_KEY', 'your-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = getattr(settings, 'ACCESS_TOKEN_EXPIRE_MINUTES', 30)
REFRESH_TOKEN_EXPIRE_DAYS = getattr(settings, 'REFRESH_TOKEN_EXPIRE_DAYS', 7)

# HTTP Bearer token scheme
bearer_scheme = HTTPBearer()


class JWTHandler:
    """JWT token management class."""
    
    def __init__(self, secret_key: str = SECRET_KEY, algorithm: str = ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None,
        token_type: str = "access"
    ) -> str:
        """
        Create a JWT token with the given data and expiration.
        
        Args:
            data: Data to encode in the token
            expires_delta: Token expiration time delta
            token_type: Type of token ('access' or 'refresh')
            
        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()
        
        # Set expiration time
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            if token_type == "refresh":
                expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
            else:
                expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": token_type
        })
        
        # Create and return token
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string to verify
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
        except Exception:
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[str]:
        """
        Create a new access token from a valid refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token or None if refresh token invalid
        """
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            return None
        
        # Create new access token with same user data
        user_data = {k: v for k, v in payload.items() 
                    if k not in ["exp", "iat", "type"]}
        
        return self.create_token(user_data, token_type="access")


# Global JWT handler instance
jwt_handler = JWTHandler()


def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create an access token with user data.
    
    Args:
        data: User data to encode in token
        expires_delta: Optional custom expiration time
        
    Returns:
        JWT access token string
    """
    return jwt_handler.create_token(data, expires_delta, "access")


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a refresh token with user data.
    
    Args:
        data: User data to encode in token
        
    Returns:
        JWT refresh token string
    """
    return jwt_handler.create_token(data, token_type="refresh")


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a JWT token and return its payload.
    
    Args:
        token: JWT token string
        
    Returns:
        Token payload or None if invalid
    """
    return jwt_handler.verify_token(token)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials from request
        db: Database session
        
    Returns:
        Current authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Verify token
        payload = verify_token(token)
        if payload is None:
            raise credentials_exception
        
        # Extract user ID
        user_id: Union[str, int] = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        
        # Convert to int if string
        if isinstance(user_id, str):
            user_id = int(user_id)
            
    except (ValueError, TypeError):
        raise credentials_exception
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    # Check if user is active
    if not getattr(user, 'is_active', True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current active user.
    
    Args:
        current_user: Current user from get_current_user dependency
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not getattr(current_user, 'is_active', True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def create_tokens_for_user(user: User) -> Dict[str, Any]:
    """
    Create both access and refresh tokens for a user.
    
    Args:
        user: User object
        
    Returns:
        Dictionary containing access_token, refresh_token, and metadata
    """
    # User data for token
    token_data = {
        "user_id": user.id,
        "email": user.email,
        "subscription_tier": getattr(user, 'subscription_tier', 'basic')
    }
    
    # Create tokens
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    } 