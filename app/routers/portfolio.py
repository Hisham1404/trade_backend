from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.connection import get_db

router = APIRouter()

@router.get("/")
async def get_portfolios():
    """Get user portfolios"""
    return {"message": "Portfolio endpoint - to be implemented"}

@router.post("/")
async def create_portfolio():
    """Create new portfolio"""
    return {"message": "Create portfolio - to be implemented"}

@router.get("/{portfolio_id}")
async def get_portfolio(portfolio_id: int):
    """Get portfolio by ID"""
    return {"message": f"Get portfolio {portfolio_id} - to be implemented"} 