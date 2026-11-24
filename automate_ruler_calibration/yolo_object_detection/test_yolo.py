

# python test_yolo.py
from ultralytics import YOLO
import cv2

MODEL_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/yolo_runs/ruler_detector_v2-4/weights/best.pt"
IMAGE_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/labeling_images/cali_c1_15Oct2025.jpg"
OUTPUT_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/test_detection.jpg"

model   = YOLO(MODEL_PATH)
results = model(IMAGE_PATH, verbose=True)

# Print raw box coordinates and confidence
for r in results:
    for box in r.boxes:
        print(f"  conf={float(box.conf):.3f}  xyxy={box.xyxy[0].tolist()}")

# Save annotated image
annotated = results[0].plot()
cv2.imwrite(OUTPUT_PATH, annotated)
print(f"Saved to {OUTPUT_PATH}")
