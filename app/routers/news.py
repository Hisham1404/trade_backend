from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.connection import get_db

router = APIRouter()

@router.get("/")
async def get_news():
    """Get news items"""
    return {"message": "News endpoint - to be implemented"}

@router.get("/search")
async def search_news():
    """Search news by symbol or keyword"""
    return {"message": "News search - to be implemented"} 