# RUN CODE in bash/terminal
# python convert_yolo_labels.py


# convert_yolo_labels.py
import json
import glob
import shutil
from pathlib import Path
import cv2
import numpy as np

label_dir  = Path("/media/peterparker/9BFA-B40E/jumping_spider/image_processing_git/spider_image_processing/image_processing/yolo_platform_detection/labeling_images")
output_dir = Path("/media/peterparker/9BFA-B40E/jumping_spider/image_processing_git/spider_image_processing/image_processing/yolo_platform_detection/yolo_dataset")

# Map label names -> class ids
CLASS_MAP = {
    "platform_a": 0,
    "platform_b": 1,
}

# YOLO expects this structure:
# yolo_dataset/
#   images/train/   ← jpg files
#   images/val/     ← jpg files (use 1-2 images for val)
#   labels/train/   ← txt files
#   labels/val/     ← txt files

for split in ["train", "val"]:
    (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

def shape_to_obb_points(shape):
    """
    Convert a labelme shape (rectangle OR polygon, any number of points)
    into 4 ordered corner representing an oriented bounding box. 

    - rectangle: labelmestores only 2 points (opposite corners), so we
    expand to the full 4-corner box first.

    - polygon: fit the minimum-area rotated rectangle around all vertices, 
    regardless of how many points there are.  
    """
    pts = np.array(shape["points"], dtype=np.float32)

    if shape["shape_type"] == "rectangle":
        #labelme rectangles are stored as [ [x1, y1], [x2, y2] ] (opposite corners)
        (x1, y1), (x2, y2) = pts[0], pts[1]
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        box = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ], dtype=np.float32)
        return box
    
    elif shape["shape_type"] == "polygon":
        # Works for any number of vertices (3, 5, 12, ...)
        rect = cv2.minAreaRect(pts)          # ((cx,cy),(w,h),angle)
        box = cv2.boxPoints(rect)            # 4 corner points, float32
        return box.astype(np.float32)
    
    else:
        return None #unsupported shape type (e.g. "circle", "line")


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

            label = shape["label"]
            if label not in CLASS_MAP:
                continue #skip labels we don't care about

            box = shape_to_obb_points(shape)
            if box is None:
                print(f" Skipping unsupported shape_type "
                      f"'{shape['shape_type']}' for label '{label}' in {json_path}")
                continue

            class_id = CLASS_MAP[label]

            #Normalize to 0-1
            box[:, 0] /= img_w
            box[:, 1] /= img_h

            coords = box.flatten()
            f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in coords)}\n")

            # if shape["label"] == "platform_a" and shape["shape_type"] == "rectangle":
            #     pts = np.array(shape["points"], dtype=np.float32)
            #     # Normalize all 4 points to 0-1
            #     pts[:, 0] /= img_w
            #     pts[:, 1] /= img_h
            #     # YOLO OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
            #     coords = pts.flatten()
            #     f.write(f"0 {' '.join(f'{v:.6f}' for v in coords)}\n")
            # elif shape["label"] == "platform_a" and shape["shape_type"] == "polygon":
            #     sdf
            # elif shape["label"] == "platform_b" and shape["shape_type"] == "polygon":


print("Done. Dataset ready.")
