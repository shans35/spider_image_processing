from pathlib import Path
import cv2
import argparse
import sys
import random

AVI_DIR = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/input/yolo_avi_collection_spiders")
OUTPUT_DIR = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/output")

def load_image(image_path: str):
    """Take a .avi file and save a random frame from it from 10%-95% in."""
    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}
    frame_index = 0

    ext = Path(image_path).suffix.lower()

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(image_path)
        if not cap.isOpened():
             raise FileNotFoundError(f"Cannot open video: {image_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = random.randint((total_frames // 10)*2, total_frames - total_frames // 20)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, img = cap.read()
        cap.release()

        if not ret or img is None:
            cap.release()
            raise ValueError(
                f"Could not read frame {frame_index} from '{image_path}' "
                f"(total frames: {total_frames})"
            )
        else:
            out_path = OUTPUT_DIR / f"{Path(image_path).stem}.jpg"
            cv2.imwrite(str(out_path), img)
            print(f"[1] Loaded image {img.shape[1]}×{img.shape[0]} px")
    
def main(): 
    for avi in AVI_DIR.iterdir():
        load_image(str(avi))

if __name__ == "__main__":
    main()