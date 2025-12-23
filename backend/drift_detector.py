import json
import numpy as np

BASELINE_FILE = "../data/drift_images/baseline_stats.json"
CURRENT_FILE = "../data/drift_images/kaggle_stats.json"

with open(BASELINE_FILE) as f:
    baseline = json.load(f)

with open(CURRENT_FILE) as f:
    current = json.load(f)

def mean(arr):
    return np.mean(arr) if len(arr) > 0 else 0

baseline_conf = mean(baseline["confidences"])
current_conf = mean(current["confidences"])

baseline_area = mean(baseline["areas"])
current_area = mean(current["areas"])

baseline_freq = baseline["total_detections"] / max(baseline["total_images"], 1)
current_freq = current["total_detections"] / max(current["total_images"], 1)

conf_drift = abs(current_conf - baseline_conf)
area_drift = abs(current_area - baseline_area)
freq_drift = abs(current_freq - baseline_freq)

drift_score = round((conf_drift + area_drift + freq_drift) / 3, 4)

if drift_score < 0.05:
    status = " STABLE"
elif drift_score < 0.15:
    status = " WARNING"
else:
    status = " RETRAIN SUGGESTED"

print(" DRIFT REPORT")
print(f"Confidence drift: {conf_drift:.4f}")
print(f"Area drift: {area_drift:.4f}")
print(f"Frequency drift: {freq_drift:.4f}")
print(f"\nOverall drift score: {drift_score}")
print(f"Model health status: {status}")
