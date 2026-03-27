from passlib.context import CryptContext
from ..database import get_database
import uuid
from datetime import datetime, timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_session(user_id: str):
    session_id = str(uuid.uuid4())
    db = get_database()
    db.sessions.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(days=1)
    })
    return session_id

def get_user_from_session(session_id: str):
    db = get_database()
    session = db.sessions.find_one({"session_id": session_id})
    if session and session['expires_at'] > datetime.now():
        return db.users.find_one({"_id": session['user_id']}) # Assuming user_id is linked
    return None

def delete_session(session_id: str):
    db = get_database()
    db.sessions.delete_one({"session_id": session_id})
