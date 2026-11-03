from pathlib import Path
import cv2
import argparse
import sys
import random

AVI_DIR = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/random guy")
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
        # 20% into video
        start_index = (total_frames // 10)*2
        # 80% into video
        end_index = total_frames - total_frames // 20
        # between the two
        mid_index = (start_index + end_index) // 2
        
        frame_indeces = [
            random.randint(start_index, mid_index - 1), # sample from 1st half
            random.randint(mid_index, end_index) # # sample from 2nd half
            ]

        # sample twice for two screen caps per avi
        for i in range(1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indeces[i])

            ret, img = cap.read()

            if not ret or img is None:
                cap.release()
                raise ValueError(
                    f"Could not read frame {frame_index} from '{image_path}' "
                    f"(total frames: {total_frames})"
                )
            else:
                out_path = OUTPUT_DIR / f"{Path(image_path).stem}_{i + 1}.jpg"
                cv2.imwrite(str(out_path), img)
                print(f"[1] Loaded image {img.shape[1]}×{img.shape[0]} px")
        cap.release()
    
def main(): 
    for avi in AVI_DIR.iterdir():
        load_image(str(avi))

if __name__ == "__main__":
    main()

# Then using this on labeling images generated
# labelme /media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/input/yolo_avi_collection_spiders/additional_input/screencaps