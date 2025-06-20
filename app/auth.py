from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User

oauth2_scheme_global = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(
    api_key: str = Depends(oauth2_scheme_global),
    db: Session = Depends(get_db)
):
    """Dependency that retrieves the current user based on an API key token."""
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return user 