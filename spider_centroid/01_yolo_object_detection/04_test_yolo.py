

# python test_yolo.py
from ultralytics import YOLO
import cv2

MODEL_PATH = "/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/yolo_object_detection/yolo_runs/spider_detector_bb_v5/weights/best.pt"
IMAGE_KNOWN_PATH = "/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/test_material/image_edit_test_results/sh089_02_c2_02Nov2025_1.jpg"
IMAGE_KNOWN_PATH2 = "/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/test_material/image_edit_test_results/edited_noblob.png"
# IMAGE_KNOWN_PATH3 = "/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/output/sh131_02_c1_02Nov2025_1.jpg"
OUTPUT_PATH = "/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/test_material/image_edit_test_results/retrained_detection_noblob.jpg"

model   = YOLO(MODEL_PATH)
results = model(IMAGE_KNOWN_PATH2, verbose=True)

# Print raw box coordinates and confidence
for r in results:
    for box in r.boxes:
        print(f"  conf={float(box.conf):.3f}  xyxy={box.xyxy[0].tolist()}")

# Save annotated image
annotated = results[0].plot()
cv2.imwrite(OUTPUT_PATH, annotated)
print(f"Saved to {OUTPUT_PATH}")
