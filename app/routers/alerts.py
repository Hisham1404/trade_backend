from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.connection import get_db

router = APIRouter()

@router.get("/")
async def get_alerts():
    """Get user alerts"""
    return {"message": "Alerts endpoint - to be implemented"}

@router.post("/")
async def create_alert():
    """Create new alert"""
    return {"message": "Create alert - to be implemented"} 