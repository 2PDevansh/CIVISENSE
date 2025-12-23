import os
import json
from ultralytics import YOLO

MODEL_PATH = "../models/damage_detector.pt"
IMAGE_DIR = "../data/drift_images/kaggle"
OUTPUT_FILE = "../data/drift_images/kaggle_stats.json"

IMG_SIZE = 640

model = YOLO(MODEL_PATH)

stats = {
    "total_images": 0,
    "total_detections": 0,
    "confidences": [],
    "areas": []
}

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    results = model(img_path)

    stats["total_images"] += 1

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            stats["total_detections"] += 1

            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            area = ((x2 - x1) * (y2 - y1)) / (IMG_SIZE * IMG_SIZE)

            stats["confidences"].append(conf)
            stats["areas"].append(area)

# Save stats
with open(OUTPUT_FILE, "w") as f:
    json.dump(stats, f, indent=4)

print(" Kaggle drift inference completed")
print(f"Images processed: {stats['total_images']}")
print(f"Detections found: {stats['total_detections']}")
