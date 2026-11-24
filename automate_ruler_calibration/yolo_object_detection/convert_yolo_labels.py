# RUN CODE in bash/terminal
# python convert_yolo_labels.py


# convert_yolo_labels.py
import json
import glob
from pathlib import Path
import cv2
import numpy as np

label_dir  = Path("/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/labeling_images")
output_dir = Path("/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/yolo_dataset")

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
    import shutil
    shutil.copy(src_img, dst_img)

    # Write YOLO label: class_id cx cy w h (all normalised 0-1)
    label_path = output_dir / "labels" / split / (Path(img_name).stem + ".txt")
    with open(label_path, "w") as f:
        for shape in data["shapes"]:
            if shape["label"] == "ruler" and shape["shape_type"] == "oriented_rectangle":
                pts = np.array(shape["points"], dtype=np.float32)
                # Normalize all 4 points to 0-1
                pts[:, 0] /= img_w
                pts[:, 1] /= img_h
                # YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
                coords = pts.flatten()
                f.write(f"0 {' '.join(f'{v:.6f}' for v in coords)}\n")

print("Done. Dataset ready.")
