# RUN CODE in bash/terminal
# python convert_yolo_labels.py


# convert_yolo_labels.py
import json
import glob
from pathlib import Path
import cv2
import numpy as np
import shutil

label_dir  = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/yolo_object_detection/labeling_images")
output_dir = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/yolo_object_detection/yolo_dataset")

# YOLO expects this structure:
# yolo_dataset/
#   images/train/   ← jpg files
#   images/val/     ← jpg files (use 1-2 images for val)
#   labels/train/   ← txt files
#   labels/val/     ← txt files

for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

json_files = sorted(glob.glob(str(label_dir / "*.json")))
# Use last image as val, rest as train
splits = {f: "val" if i == len(json_files)-1 else "train"
          for i, f in enumerate(json_files)}

for json_path, split in splits.items():
    with open(json_path) as f:
        data = json.load(f)

    img_h = data["imageHeight"]
    img_w = data["imageWidth"]
    img_name = Path(data["imagePath"]).name

    # Copy image
    src_img = label_dir / img_name
    dst_img = output_dir / "images" / split / img_name
    
    shutil.copy(src_img, dst_img)

    # Write YOLO label: class_id cx cy w h (all normalised 0-1)
    label_path = output_dir / "labels" / split / (Path(img_name).stem + ".txt")
    with open(label_path, "w") as f:
        for shape in data["shapes"]:
            if shape["label"] == "spider" and shape["shape_type"] == "rectangle":
                pts = np.array(shape["points"], dtype=np.float32)

                # adapted previous conver_yolo_labels to use rectangle/BB instead of OBB
                x1, y1 = pts[0]
                x2, y2 = pts[1]

                # Make sure coords are ordered correctly regardless of draw direction
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])

                cx = (x_min + x_max) / 2 / img_w
                cy = (y_min + y_max) / 2 / img_h
                w  = (x_max - x_min) / img_w
                h  = (y_max - y_min) / img_h

                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

print("Done. Dataset ready.")
