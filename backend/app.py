from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.db import predictions_col, log_prediction, log_model_health
from backend.risk_engine import compute_severity

from ultralytics import YOLO
from PIL import Image
import io
import json
import numpy as np
import os
import uuid
import cv2

# -------------------------
# PATH SETUP (CRITICAL FIX)
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "drift_images")
UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "backend", "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "models", "damage_detector.pt")
BASELINE_STATS_PATH = os.path.join(DATA_DIR, "baseline_stats.json")
LIVE_STATS_PATH = os.path.join(DATA_DIR, "live_stats.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 640

# -------------------------
# APP INIT
# -------------------------
app = FastAPI(
    title="CIVISENSE API",
    description="AI-powered Urban Damage Intelligence & Model Health Monitoring",
    version="1.0.0"
)

# -------------------------
# LOAD MODEL
# -------------------------
model = YOLO(MODEL_PATH)

# MUST MATCH TRAINED CLASSES
CIVISENSE_CLASSES = {
    0: "Alligator",
    1: "Edge Cracking",
    2: "Lateral-Crack",
    3: "Longitudinal-Crack",
    4: "Ravelling",
    5: "Rutting",
    6: "Striping",
    7: "pothole"
}

# -------------------------
# STATIC FILES
# -------------------------
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# -------------------------
# INIT LIVE STATS FILE
# -------------------------
if not os.path.exists(LIVE_STATS_PATH):
    with open(LIVE_STATS_PATH, "w") as f:
        json.dump(
            {
                "total_images": 0,
                "total_detections": 0,
                "confidences": [],
                "areas": []
            },
            f,
            indent=4
        )

# -------------------------
# HELPERS
# -------------------------
def mean(arr):
    return float(np.mean(arr)) if arr else 0.0

def update_live_stats(detections):
    with open(LIVE_STATS_PATH) as f:
        stats = json.load(f)

    stats["total_images"] += 1
    stats["total_detections"] += len(detections)

    for det in detections:
        stats["confidences"].append(det["confidence"])
        x1, y1, x2, y2 = det["bbox"]
        area = ((x2 - x1) * (y2 - y1)) / (IMG_SIZE * IMG_SIZE)
        stats["areas"].append(area)

    stats["confidences"] = stats["confidences"][-100:]
    stats["areas"] = stats["areas"][-100:]

    with open(LIVE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)

def draw_and_save(image_path, detections):
    image = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["class"]} {det["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, image)

    return f"/outputs/{filename}"

# -------------------------
# ROOT
# -------------------------
@app.get("/")
def root():
    return {"status": "CIVISENSE backend running"}

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        upload_path = os.path.join(
            UPLOAD_DIR, f"{uuid.uuid4().hex}_{image.filename}"
        )
        pil_img.save(upload_path)

        results = model(pil_img)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = CIVISENSE_CLASSES.get(cls, "unknown")
                severity, level = compute_severity(conf, [x1, y1, x2, y2], class_name)

                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "severity": round(severity, 4),
                    "risk_level": level,
                    "bbox": [x1, y1, x2, y2]
                })

        update_live_stats(detections)
        log_prediction(image.filename, detections)

        annotated_image = draw_and_save(upload_path, detections)

        return {
            "num_detections": len(detections),
            "detections": detections,
            "annotated_image": annotated_image
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------------
# MODEL HEALTH
# -------------------------
@app.get("/model-health")
def model_health():
    with open(BASELINE_STATS_PATH) as f:
        baseline = json.load(f)

    with open(LIVE_STATS_PATH) as f:
        current = json.load(f)

    conf_drift = abs(mean(current["confidences"]) - mean(baseline["confidences"]))
    area_drift = abs(mean(current["areas"]) - mean(baseline["areas"]))
    freq_drift = abs(
        (current["total_detections"] / max(current["total_images"], 1)) -
        (baseline["total_detections"] / max(baseline["total_images"], 1))
    )
    freq_drift = min(freq_drift, 1.0)

    drift_score = round((conf_drift + area_drift + freq_drift) / 3, 4)

    status = (
        "STABLE" if drift_score < 0.05
        else "WARNING" if drift_score < 0.15
        else "RETRAIN_SUGGESTED"
    )

    health_data = {
        "drift_score": drift_score,
        "confidence_drift": round(conf_drift, 4),
        "area_drift": round(area_drift, 4),
        "frequency_drift": round(freq_drift, 4),
        "status": status
    }

    log_model_health(health_data)
    return health_data

# -------------------------
# ANALYTICS
# -------------------------
@app.get("/analytics/summary")
def analytics_summary():
    total_images = predictions_col.count_documents({})

    pipeline = [
        {"$unwind": "$detections"},
        {"$group": {"_id": "$detections.class", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]

    damage_stats = list(predictions_col.aggregate(pipeline))

    high_risk_pipeline = [
        {"$unwind": "$detections"},
        {"$match": {"detections.risk_level": "HIGH"}},
        {"$count": "count"}
    ]

    high_risk_result = list(predictions_col.aggregate(high_risk_pipeline))
    high_risk = high_risk_result[0]["count"] if high_risk_result else 0

    return {
        "total_images_processed": total_images,
        "damage_distribution": damage_stats,
        "high_risk_detections": high_risk
    }

@app.get("/alerts/high-risk")
def high_risk_alerts(limit: int = 5):
    pipeline = [
        {"$unwind": "$detections"},
        {"$match": {"detections.risk_level": "HIGH"}},
        {"$sort": {"timestamp": -1}},
        {"$limit": limit},
        {
            "$project": {
                "_id": 0,
                "image_name": 1,
                "class": "$detections.class",
                "confidence": "$detections.confidence",
                "severity": "$detections.severity",
                "timestamp": 1
            }
        }
    ]

    alerts = list(predictions_col.aggregate(pipeline))
    return {"alerts": alerts}
