def compute_severity(confidence, box, class_name):
    x1, y1, x2, y2 = box
    IMG_SIZE = 640

    area = ((x2 - x1) * (y2 - y1)) / (IMG_SIZE * IMG_SIZE)

    weights = {
        "pothole": 1.0,
        "crack": 0.6,
        "surface_damage": 0.4
    }

    severity = confidence * area * weights.get(class_name, 0.5)

    if severity < 0.05:
        level = "LOW"
    elif severity < 0.15:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return round(severity, 4), level
