from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

MONGO_URI = "mongodb://127.0.0.1:27017"
DATABASE_NAME = "vigilvoice"

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

try:
    client.admin.command("ping")
    print("[MongoDB] Connected successfully")
except ConnectionFailure as e:
    print(f"[MongoDB] Connection failed: {e}")
    raise

db = client[DATABASE_NAME]

speakers_collection = db["speakers"]
security_events_collection = db["security_events"]

# Useful indexes
speakers_collection.create_index("speaker_id", unique=True)
speakers_collection.create_index("name")

security_events_collection.create_index("speaker_id")
security_events_collection.create_index("timestamp")

print(f"[MongoDB] Database: {DATABASE_NAME}")