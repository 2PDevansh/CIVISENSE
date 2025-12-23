import os
from ultralytics import YOLO
from risk_engine import compute_severity

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "../models/damage_detector.pt"
IMAGE_DIR = "../data/test_images"

CIVISENSE_CLASSES = {
    0: "pothole",
    1: "crack",
    2: "surface_damage"
}

# -------------------------
# LOAD MODEL
# -------------------------
model = YOLO(MODEL_PATH)

# -------------------------
# RUN INFERENCE
# -------------------------
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    results = model(img_path)

    print(f"\n📸 Results for {img_name}")

    detections_found = False

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            detections_found = True

            cls = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()

            class_name = CIVISENSE_CLASSES.get(cls, "unknown")

            severity, level = compute_severity(conf, xyxy, class_name)

            print(
                f"  ➤ {class_name:<15} | "
                f"conf={conf:.2f} | "
                f"severity={severity:.4f} | "
                f"level={level}"
            )

    if not detections_found:
        print("  ⚠️ No damage detected")
