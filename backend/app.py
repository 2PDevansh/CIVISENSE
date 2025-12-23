from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import json
import numpy as np
import os

from risk_engine import compute_severity
from db import log_prediction, log_model_health

# -------------------------
# APP INIT
# -------------------------
app = FastAPI(
    title="CIVISENSE API",
    description="AI-powered Urban Damage Intelligence & Model Health Monitoring",
    version="1.0.0"
)

# -------------------------
# PATHS
# -------------------------
MODEL_PATH = "../models/damage_detector.pt"
BASELINE_STATS_PATH = "../data/drift_images/baseline_stats.json"
LIVE_STATS_PATH = "../data/drift_images/live_stats.json"

IMG_SIZE = 640

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
    return float(np.mean(arr)) if len(arr) else 0.0

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

    # keep rolling window (last 100 detections)
    stats["confidences"] = stats["confidences"][-100:]
    stats["areas"] = stats["areas"][-100:]

    with open(LIVE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)

# -------------------------
# ROOT
# -------------------------
@app.get("/")
def root():
    return {"status": "CIVISENSE backend running"}

# -------------------------
# PREDICT ENDPOINT
# -------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(img)
        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_name = CIVISENSE_CLASSES.get(cls, "unknown")
                severity, level = compute_severity(
                    conf,
                    [x1, y1, x2, y2],
                    class_name
                )

                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "severity": round(severity, 4),
                    "risk_level": level,
                    "bbox": [x1, y1, x2, y2]
                })

        # update drift stats
        update_live_stats(detections)

        # log prediction to MongoDB
        log_prediction(image.filename, detections)

        return {
            "num_detections": len(detections),
            "detections": detections
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# -------------------------
# MODEL HEALTH ENDPOINT
# -------------------------
@app.get("/model-health")
def model_health():
    try:
        with open(BASELINE_STATS_PATH) as f:
            baseline = json.load(f)

        with open(LIVE_STATS_PATH) as f:
            current = json.load(f)

        conf_drift = abs(
            mean(current["confidences"]) -
            mean(baseline["confidences"])
        )

        area_drift = abs(
            mean(current["areas"]) -
            mean(baseline["areas"])
        )

        freq_drift = abs(
            (current["total_detections"] / max(current["total_images"], 1)) -
            (baseline["total_detections"] / max(baseline["total_images"], 1))
        )

        # cap frequency influence
        freq_drift = min(freq_drift, 1.0)

        drift_score = round(
            (conf_drift + area_drift + freq_drift) / 3,
            4
        )

        if drift_score < 0.05:
            status = "STABLE"
        elif drift_score < 0.15:
            status = "WARNING"
        else:
            status = "RETRAIN_SUGGESTED"

        health_data = {
            "drift_score": drift_score,
            "confidence_drift": round(conf_drift, 4),
            "area_drift": round(area_drift, 4),
            "frequency_drift": round(freq_drift, 4),
            "status": status
        }

        # log model health to MongoDB
        log_model_health(health_data)

        return health_data

    except Exception as e:
     print("ERROR:", e)
     raise e

