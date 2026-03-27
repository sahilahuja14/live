import asyncio
import os
import sys

# Add backend directory to sys.path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import connect_db, get_database, close_db
from app.core.security import get_password_hash
from datetime import datetime, timezone

def get_password_hash(password: str) -> str:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(password)

async def seed_users():
    connect_db()
    db = get_database()
    
    users = [
        {
            "username": "Finance",
            "hashed_password": get_password_hash("finance12345"),
            "role": "operation",
            "department_id": "dept_ops_1",
            "created_at": datetime.now(timezone.utc)
        }
    ]
    db.users.insert_one(users[0])
    # try:
    #     # Check if users already exist to avoid duplication
    #     if db.users.count_documents({}) == 0:
    #         db.users.insert_many(users)
    #         print("Successfully seeded users: admin, sales_user, finance_user")
    #     else:
    #         print("Users already exist. Skipping seed.")
    #         # For testing, we can force delete and recreate:
    #         # db.users.delete_many({})
    #         # db.users.insert_many(users)
    #         # print("Re-seeded users")
    # except Exception as e:
    #     print(f"Error seeding users: {e}")
    # finally:
    #     close_db()

if __name__ == "__main__":
    asyncio.run(seed_users())
