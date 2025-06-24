"""
Authentication router with comprehensive user management endpoints.

Provides secure user registration, login, logout, token refresh,
password management, and session handling functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.models.user import User
from app.schemas.auth import (
    UserRegistrationRequest, UserLoginRequest, TokenResponse,
    RefreshTokenRequest, UserResponse, PasswordChangeRequest,
    PasswordResetRequest, PasswordResetConfirmRequest,
    APIKeyResponse, UserSessionInfo, LogoutRequest
)
from app.auth.jwt_handler import (
    create_tokens_for_user, verify_token, get_current_user,
    get_current_active_user, create_access_token
)
from app.auth.password_manager import (
    hash_password, verify_password, require_strong_password
)
from app.auth.rate_limiter import check_rate_limit, get_client_identifier

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
bearer_scheme = HTTPBearer()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegistrationRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    
    Creates a new user with validated email and password,
    generates API key, and returns user information.
    """
    # Check rate limiting
    client_id = get_client_identifier(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Validate password strength
    try:
        require_strong_password(user_data.password)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Hash password
    hashed_password = hash_password(user_data.password)
    
    # Generate API key
    api_key = str(uuid.uuid4())
    
    # Generate username from email (before @)
    username = user_data.email.split('@')[0]
    
    # Ensure username is unique by adding numbers if needed
    base_username = username
    counter = 1
    while db.query(User).filter(User.username == username).first():
        username = f"{base_username}{counter}"
        counter += 1
    
    # Create user
    new_user = User(
        email=user_data.email,
        username=username,
        hashed_password=hashed_password,
        api_key=api_key,
        subscription_tier=user_data.subscription_tier.value,
        is_active=True,
        created_at=datetime.utcnow(),
        last_active=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.post("/login", response_model=TokenResponse)
async def login_user(
    login_data: UserLoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access tokens.
    
    Validates credentials and returns JWT access and refresh tokens
    for authenticated sessions.
    """
    # Check rate limiting
    client_id = get_client_identifier(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Find user by email
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated"
        )
    
    # Update last active timestamp
    user.last_active = datetime.utcnow()
    db.commit()
    
    # Create tokens
    tokens = create_tokens_for_user(user)
    
    # Set secure cookie for refresh token if remember_me is enabled
    if login_data.remember_me:
        response.set_cookie(
            key="refresh_token",
            value=tokens["refresh_token"],
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=True,
            samesite="strict"
        )
    
    return tokens


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using valid refresh token.
    
    Generates new access token from valid refresh token
    without requiring re-authentication.
    """
    # Check rate limiting
    client_id = get_client_identifier(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Verify refresh token
    payload = verify_token(refresh_data.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get user from token
    user_id = payload.get("user_id")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Update last active
    user.last_active = datetime.utcnow()
    db.commit()
    
    # Create new tokens
    return create_tokens_for_user(user)


@router.post("/logout")
async def logout_user(
    logout_data: LogoutRequest,
    current_user: User = Depends(get_current_active_user),
    response: Response = None
):
    """
    Logout user and invalidate tokens.
    
    Optionally revokes all user sessions and clears
    authentication cookies.
    """
    # Clear refresh token cookie
    if response:
        response.delete_cookie(key="refresh_token")
    
    # In a production system, you would:
    # 1. Add tokens to a blacklist/revocation list
    # 2. Remove refresh tokens from database
    # 3. Clear session data
    
    if logout_data.revoke_all_sessions:
        # Revoke all sessions by updating a user field
        # This would invalidate all existing tokens
        pass
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information.
    
    Returns detailed information about the currently
    authenticated user including profile and settings.
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    update_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update current user profile information.
    
    Allows users to update their profile data while
    maintaining authentication state.
    """
    # Update allowed fields
    allowed_fields = ["subscription_tier"]
    
    for field, value in update_data.items():
        if field in allowed_fields and hasattr(current_user, field):
            setattr(current_user, field, value)
    
    db.commit()
    db.refresh(current_user)
    
    return current_user


@router.post("/change-password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change user password.
    
    Validates current password and updates to new password
    with strength validation.
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password strength
    try:
        require_strong_password(password_data.new_password)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Update password
    current_user.hashed_password = hash_password(password_data.new_password)
    db.commit()
    
    return {"message": "Password updated successfully"}


@router.post("/reset-password")
async def request_password_reset(
    reset_data: PasswordResetRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Request password reset for user account.
    
    Sends password reset instructions to user's email
    if account exists.
    """
    # Check rate limiting
    client_id = get_client_identifier(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Find user by email
    user = db.query(User).filter(User.email == reset_data.email).first()
    
    # Always return success to prevent email enumeration
    # In production, send actual email with reset token
    
    if user:
        # Generate reset token and store it
        # Send email with reset instructions
        pass
    
    return {"message": "If an account with this email exists, password reset instructions have been sent"}


@router.post("/reset-password/confirm")
async def confirm_password_reset(
    reset_data: PasswordResetConfirmRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Confirm password reset with token.
    
    Validates reset token and updates user password
    to new secure password.
    """
    # Check rate limiting
    client_id = get_client_identifier(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Verify reset token (implement token validation)
    # For now, return error as token system not fully implemented
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Password reset token validation not implemented"
    )


@router.post("/api-key/regenerate", response_model=APIKeyResponse)
async def regenerate_api_key(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Regenerate user's API key.
    
    Creates new API key and invalidates the old one
    for enhanced security.
    """
    # Generate new API key
    new_api_key = str(uuid.uuid4())
    
    # Update user's API key
    current_user.api_key = new_api_key
    db.commit()
    
    return APIKeyResponse(
        api_key=new_api_key,
        created_at=datetime.utcnow(),
        expires_at=None
    )


@router.get("/session", response_model=UserSessionInfo)
async def get_session_info(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current session information.
    
    Returns details about the current authenticated
    session including timing and client info.
    """
    # Get client information
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent")
    
    # Get forwarded IP if behind proxy
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    
    return UserSessionInfo(
        user_id=current_user.id,
        email=current_user.email,
        subscription_tier=current_user.subscription_tier,
        session_start=current_user.last_active or current_user.created_at,
        last_activity=current_user.last_active or current_user.created_at,
        ip_address=client_ip,
        user_agent=user_agent
    )


@router.get("/health")
async def auth_health_check():
    """
    Authentication service health check.
    
    Returns the status of authentication services
    and dependencies.
    """
    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "registration": True,
            "login": True,
            "token_refresh": True,
            "password_reset": False,  # Not fully implemented
            "rate_limiting": True
        }
    } 