import os
import datetime
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

# Set loop appropriately depending on environment
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.security import get_password_hash

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "devdb")

def seed_users_templates():
    if not MONGO_URI:
        print("Error: MONGO_URI environment variable is not set.")
        return

    print(f"Connecting to MongoDB at {MONGO_URI}...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    print("Clearing existing users and templates...")
    db.users.delete_many({})
    db.templates.delete_many({})

    # 1. Create a Super Admin User
    print("Creating admin user...")
    admin_user = {
        "username": "admin",
        "hashed_password": get_password_hash("admin"),
        "role": "superadmin",
        "department_id": None,
        "created_at": datetime.datetime.utcnow()
    }
    db.users.insert_one(admin_user)
    print("User 'admin' created with password 'admin' (Role: superadmin).")

    # 1.1 Create a Sales User
    print("Creating sales user...")
    sales_user = {
        "username": "sales",
        "hashed_password": get_password_hash("sales"),
        "role": "sales",
        "department_id": None,
        "created_at": datetime.datetime.utcnow()
    }
    db.users.insert_one(sales_user)
    print("User 'sales' created with password 'sales' (Role: sales).")

    # 1.2 Create a Finance User
    print("Creating finance user...")
    finance_user = {
        "username": "finance",
        "hashed_password": get_password_hash("finance"),
        "role": "finance",
        "department_id": None,
        "created_at": datetime.datetime.utcnow()
    }
    db.users.insert_one(finance_user)
    print("User 'finance' created with password 'finance' (Role: finance).")

    # 2. Create a default Dashboard Template
    print("Creating default template...")
    default_template = {
        "name": "Default Overview",
        "widgets": [
            {
                "id": "w1",
                "type": "kpi",
                "config": {
                    "title": "Total Queries",
                    "dataKey": "open",
                    "value": "queries.open",
                    "colSpan": 3,
                    "rowSpan": 1
                }
            },
            {
                "id": "w2",
                "type": "kpi",
                "config": {
                    "title": "Total Bookings",
                    "dataKey": "booking",
                    "value": "bookings.booking",
                    "colSpan": 3,
                    "rowSpan": 1
                }
            }
        ],
        "assigned_departments": [],
        "assigned_roles": ["superadmin", "user", "sales", "finance", "operations"],
        "created_at": datetime.datetime.utcnow(),
        "created_by": "admin"
    }
    db.templates.insert_one(default_template)
    print("Default template 'Default Overview' created.")

    print("Seeding of users and templates completed successfully.")

if __name__ == "__main__":
    seed_users_templates()
