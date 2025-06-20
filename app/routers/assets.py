from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.connection import get_db

router = APIRouter()

@router.get("/")
async def get_assets():
    """Get portfolio assets"""
    return {"message": "Assets endpoint - to be implemented"}

@router.post("/")
async def add_asset():
    """Add asset to portfolio"""
    return {"message": "Add asset - to be implemented"} 