import cv2
import os
import uuid

OUTPUT_DIR = "outputs/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def draw_and_save(image_path, detections):
    """
    detections: list of dicts with bbox, class, confidence
    """
    image = cv2.imread(image_path)
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)

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

    cv2.imwrite(save_path, image)
    return save_path
