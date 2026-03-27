from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt # Using PyJWT
from .security import decode_token
from ..schemas.auth import TokenData, UserInDB
from ..database import get_async_database
from typing import Optional

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "access":
            raise credentials_exception
            
        token_data = TokenData(username=username)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise credentials_exception
    except Exception:
        raise credentials_exception

    db = get_async_database()
    # Asnyc call to MongoDB using Motor
    user_doc = await db.users.find_one({"username": token_data.username})
    
    if user_doc is None:
        raise credentials_exception
        
    user_doc["_id"] = str(user_doc["_id"])
    return UserInDB(**user_doc)

def require_role(allowed_roles: list[str]):
    async def role_checker(current_user: UserInDB = Depends(get_current_user)):
        # Safely get the role avoiding AttributeError
        user_role = getattr(current_user, "role", None)
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return role_checker
