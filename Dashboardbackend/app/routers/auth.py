from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import jwt
from datetime import datetime, timedelta, timezone
from ..schemas.auth import UserInDB, UserResponse, Token, RefreshRequest, AuthResponse

IST = timezone(timedelta(hours=5, minutes=30))
from ..core.security import verify_password, create_access_token, create_refresh_token, decode_token, ACCESS_TOKEN_EXPIRE_MINUTES
from ..core.deps import get_current_user
from ..database import get_async_database

router = APIRouter()

@router.post("/login", response_model=AuthResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = get_async_database()
    # Async DB fetch
    user_doc = await db.users.find_one({"username": form_data.username})

    if not user_doc or not verify_password(form_data.password, user_doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate tokens
    access_token = create_access_token(data={"sub": user_doc["username"], "role": user_doc["role"]})
    refresh_token = create_refresh_token(data={"sub": user_doc["username"], "role": user_doc["role"]})

    expires_at = datetime.now(IST) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    return {
        "success": True,
        "data": {
            "access_token": access_token,
            "token_type": "bearer",
            "refresh_token": refresh_token,
            "expires_at": expires_at
        }
    }

@router.post("/refresh", response_model=AuthResponse)
async def refresh_access_token(request: RefreshRequest):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode and validate the refresh token
        payload = decode_token(request.refresh_token)
        username: str = payload.get("sub")
        token_type: str = payload.get("type")

        if username is None or token_type != "refresh":
            raise credentials_exception

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        raise credentials_exception

    # Usually you also query DB here to check if the user is completely active/exists. Let's do a quick DB check.
    db = get_async_database()
    user_doc = await db.users.find_one({"username": username})
    if not user_doc:
        raise credentials_exception

    # Generate new access token
    new_access_token = create_access_token(data={"sub": user_doc["username"], "role": user_doc["role"]})
    new_refresh_token = create_refresh_token(data={"sub": user_doc["username"], "role": user_doc["role"]})

    expires_at = datetime.now(IST) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    return {
        "success": True,
        "data": {
            "access_token": new_access_token,
            "token_type": "bearer",
            "refresh_token": new_refresh_token,
            "expires_at": expires_at
        }
    }

@router.post("/logout")
async def logout(current_user: UserInDB = Depends(get_current_user)):
    # Since we are using stateless JWTs, we just return a success response
    # The client is responsible for deleting the tokens.
    return {"message": "Successfully logged out. Please clear your local tokens."}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    return current_user
