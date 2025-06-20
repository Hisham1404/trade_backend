from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import secrets

from app.database.connection import get_db
from app.models.user import User
from app.core.config import settings

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/register")
async def register_user(
    username: str,
    email: str, 
    password: str,
    full_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == email) | (User.username == username)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Generate API key
    api_key = f"trading_agent_{secrets.token_urlsafe(32)}"
    
    # Hash password (simplified - in production use proper hashing)
    hashed_password = f"hashed_{password}"  # TODO: Implement proper password hashing
    
    # Create new user
    new_user = User(
        username=username,
        email=email,
        full_name=full_name,
        hashed_password=hashed_password,
        api_key=api_key,
        created_at=datetime.utcnow()
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "message": "User registered successfully",
        "user_id": new_user.id,
        "username": new_user.username,
        "api_key": new_user.api_key
    }

@router.post("/login")
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login user and return API key"""
    
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == form_data.username) | (User.email == form_data.username)
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Verify password (simplified)
    if user.hashed_password != f"hashed_{form_data.password}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Update last login
    user.last_login_at = datetime.utcnow()
    db.commit()
    
    return {
        "access_token": user.api_key,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }

@router.get("/me")
async def get_current_user_info(
    api_key: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Get current user information"""
    
    user = db.query(User).filter(User.api_key == api_key).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "subscription_tier": user.subscription_tier,
        "created_at": user.created_at,
        "last_login_at": user.last_login_at
    } 