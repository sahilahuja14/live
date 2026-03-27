from pymongo import MongoClient, AsyncMongoClient
import os
import asyncio

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

MODE_DBS = {
    "Air": os.getenv("AIR_DB_NAME", "air_db"),
    "Ocean": os.getenv("OCEAN_DB_NAME", "ocean_db"),
    "Road": os.getenv("ROAD_DB_NAME", "road_db"),
    "Courier": os.getenv("COURIER_DB_NAME", "courier_db")
}

class Database:
    client: MongoClient = None
    async_client: AsyncMongoClient = None

db = Database()

if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not set")

if not DB_NAME:
    raise RuntimeError("DB_NAME is not set")

def get_db_client():
    return db.client

def get_database():
    return db.client.get_database(DB_NAME)

def get_async_database():
    return db.async_client.get_database(DB_NAME)

def get_analytics_database():
    return get_database()

def get_async_analytics_database():
    return get_async_database()

def get_mode_database(mode: str):
    db_name = MODE_DBS.get(mode)
    if not db_name:
        raise ValueError(f"Unknown mode: {mode}")
    return db.client.get_database(db_name)

def get_async_mode_database(mode: str):
    db_name = MODE_DBS.get(mode)
    if not db_name:
        raise ValueError(f"Unknown mode: {mode}")
    return db.async_client.get_database(db_name)

def get_all_async_mode_databases():
    return {mode: db.async_client.get_database(name) for mode, name in MODE_DBS.items()}

def get_all_mode_databases():
    return {mode: db.client.get_database(name) for mode, name in MODE_DBS.items()}

def connect_db():
    try:
        # Sync Client
        db.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db.client.admin.command('ping')
        print("Connected to MongoDB (Sync)")

        # Async Client
        db.async_client = AsyncMongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # We'll verify async connection in the startup check
        print("Initialized MongoDB (Async)")
            
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise e

def close_db():
    if db.client:
        db.client.close()
    if db.async_client:
        db.async_client.close()
    print("Disconnected from MongoDB")
