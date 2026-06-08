# June 8, 2026
# Ana Curtis
# Sophie Hanson

import cv2
from pathlib import Path

# Objective: extract a frame from one of the avi's and identify the landing platform. 

#---- Helpers ----#

# Platform image grayitization
def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#---- LOAD ATTEMPT ----#

def load_image(image_path: str, frame_index: int = 0):
    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}
    ext = Path(image_path).suffix.lower()

    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(image_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {image_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, img = cap.read()
        cap.release()
        if not ret or img is None:
            raise ValueError(
                f"Could not read frame {frame_index} from '{image_path}' "
                f"(total frames: {total_frames})"
            )
        print(f"[1] Loaded AVI frame {frame_index}/{total_frames-1} — "
              f"{img.shape[1]}×{img.shape[0]} px")
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        print(f"[1] Loaded image {img.shape[1]}×{img.shape[0]} px")
 
    return img
        

print(cv2.__version__)

if __name__ == '__main__':
    INPUT_DIR = "/media/peterparker/9BFA-B40E/jumping_spider/image-processing/input"
    OUTPUT_DIR = "/media/peterparker/9BFA-B40E/jumping_spider/image-processing/output"
    