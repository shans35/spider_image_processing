# python test_yolo.py
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from typing import NamedTuple
from pathlib import Path
import numpy as np
import random
import cv2

# invoke yolo on one frame from every video
# display raw image, yolo applied, and sam applied using matplotlib
# save into debug directory

MODEL_PATH = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/yolo_object_detection/yolo_runs/spider_detector_bb_v5/weights/best.pt")
AVI_DIR = Path("/media/peterparker/2597-CC61/misc/Phidippus_videos_for_final_data_analysis")
OUTPUT_DIR = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/debug_detection")

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}

model   = YOLO(MODEL_PATH)

class YoloResult(NamedTuple):
    box: np.ndarray
    img: np.ndarray
    crop: np.ndarray

def load_image(image_path: str):
    """Take a .avi file and save a random frame from it from 20%-80% in."""
    
    frame_index = 0
    ext = Path(image_path).suffix.lower()

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(image_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {image_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 20% into video
        start_index = (total_frames // 10)*2
        # 80% into video
        end_index = total_frames - total_frames // 20
               
        frame_range = random.randint(start_index, end_index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range)
        ret, img = cap.read()
        cap.release()

        if not ret or img is None:
            raise ValueError(
                f"Could not read frame {frame_index} from '{image_path}' "
                f"(total frames: {total_frames})"
            )
        return img

def invoke_yolo(avi_path: str)-> YoloResult:
    """Runs YOLO inference on a single frame path and returns the bbox."""

    img = load_image(avi_path)

    if img is None:
        raise ValueError(
            f"Current img is None"
        )

    yolo_result = model(img, verbose=False)

    if not yolo_result or len(yolo_result[0].boxes) == 0:
        raise RuntimeError(
            f"No objects detected by YOLO in frame from current img: {avi_path}"
        )
    

    x, y, w, h = yolo_result[0].boxes.xywh[0].detach().cpu().numpy()

    box = np.array(
        [
            int(x - w / 2),
            int(y - h / 2),
            int(x + w / 2),
            int(y + h / 2),
        ],
        dtype=np.float32,
    )

    # xyxy gives corners directly
    x1, y1, x2, y2 = yolo_result[0].boxes.xyxy[0].detach().cpu().numpy()

    # Clamp to image bounds and convert to ints for slicing
    if img is not None: 
        h, w = img.shape[:2]
        x1, x2 = max(0, int(x1)), min(w, int(x2))
        y1, y2 = max(0, int(y1)), min(h, int(y2))

        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        cropped_img = img[y1:y2, x1:x2]

    return YoloResult(box, img, cropped_img)

def generate_debug(res: YoloResult, name: str) -> None:
    """Save a raw | YOLO box | crop figure to OUTPUT_DIR."""
    x1, y1, x2, y2 = res.box.astype(int)

    boxed = res.img.copy()
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # matplotlib wants RGB, OpenCV frames are BGR
    panels = [
        ("Raw", cv2.cvtColor(res.img, cv2.COLOR_BGR2RGB)),
        ("YOLO", cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)),
        ("Crop", cv2.cvtColor(res.crop, cv2.COLOR_BGR2RGB)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}_debug.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

def generate_empty_debug(res: YoloResult, name: str, reason: str) -> None:
    """Save a placeholder figure when detection fails."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, title in zip(axes, ["Raw", "YOLO", "Crop"]):
        ax.text(0.5, 0.5, "FAILED", ha="center", va="center", fontsize=20, color="red")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{name}: {reason}", fontsize=10)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}_debug_FAILED.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
   

def main(): 
    sorted_avis = sorted(AVI_DIR.glob("*.avi"))
    errors = []
    for avi in sorted_avis:
        if avi.suffix.lower() not in VIDEO_EXTS:
            continue
        try:
            res = invoke_yolo(str(avi))
            generate_debug(res, avi.stem)
        except Exception as e: 
            errors.append({
                "video": str(avi.stem),
                "error": str(e)
            })
    #        generate_empty_debug(avi.stem, str(e))
            continue

    # report at the end
    if errors:
        print(f"\n{len(errors)} of {len(sorted_avis)} videos failed:\n")
        for item in errors:
            print(f"- {item['video']}: {item['error']!r}")
            # print(item["traceback"])  # uncomment for full details

    return errors

if __name__ == "__main__":
    main()

