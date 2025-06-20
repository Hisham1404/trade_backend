"""
API routes for managing news feeds and sources.
"""
from fastapi import APIRouter

router = APIRouter()

# Basic health check for the feeds router
@router.get("/")
async def get_feeds_status():
    return {"status": "ok", "router": "feeds"} 