from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.connection import get_db

router = APIRouter()

@router.get("/")
async def get_users():
    """Get all users (admin endpoint)"""
    return {"message": "Users endpoint - to be implemented"}

@router.get("/{user_id}")
async def get_user(user_id: int):
    """Get user by ID"""
    return {"message": f"Get user {user_id} - to be implemented"} 