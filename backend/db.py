from pymongo import MongoClient
from datetime import datetime

MONGO_URI = "mongodb+srv://devanshprasad798_db_user:uDkX62Keuf7w4aVx@civisense.ni0g7jc.mongodb.net/?retryWrites=true&w=majority"


client = MongoClient(MONGO_URI)

db = client["civisense"]

predictions_col = db["predictions"]
model_health_col = db["model_health"]

def log_prediction(image_name, detections):
    doc = {
        "image_name": image_name,
        "detections": detections,
        "timestamp": datetime.utcnow()
    }
    predictions_col.insert_one(doc)


def log_model_health(health_data):
    doc = health_data.copy()   # IMPORTANT
    doc["timestamp"] = datetime.utcnow()
    model_health_col.insert_one(doc)

