# pip install ultralytics labelme # for YOLO object detection

# ──────────────────────────────────────────────────────────────────────────────
# RUN CODE in bash/terminal
# python yolo_extract_frames.py

# yolo_extract_frames.py — run this once to get images for labeling
import cv2
from pathlib import Path
import glob

# list avi calibration files in image processing folder - there are 25. 
avi_files = glob.glob("/media/peterparker/9BFA-B40E/jumping_spider/image_processing/yolo_avi_collection/*.avi")
print(f"Found {len(avi_files)} AVI files:")
for f in avi_files:
    print(f"  {f}")

# avi_files = [
#     "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_02Nov2025.avi",
#     "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_15Oct2025.avi",
#     "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_12Oct2025.avi",
#     "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_20Sept2025.avi",
#     "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_01Oct2025.avi",
#     "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_22Sept2025.avi",
# ]

out_dir = Path("/media/peterparker/9BFA-B40E/jumping_spider/image_processing/labeling_images")
out_dir.mkdir(exist_ok=True)

for avi in avi_files:
    cap = cv2.VideoCapture(avi)
    ret, frame = cap.read()
    cap.release()
    if ret:
        name = Path(avi).stem + ".jpg"
        cv2.imwrite(str(out_dir / name), frame)
        print(f"Saved {name}")

# ──────────────────────────────────────────────────────────────────────────────

# THEN:
# in bash (terminal), run the following code and draw a rectangle around each ruler
# labelme /Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/labeling_images/
# Use angled ruler to label the ruler tightly (no background space)