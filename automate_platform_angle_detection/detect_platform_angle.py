# June 8, 2026
# Ana Curtis

# ~~~~~~~some helpful tools/keyboard shortcuts~~~~~~~
# pip show opencv-python            (to check if a package is installed)
# Ctrl+Shft+P                       (to open python interpreter and select environment)
# Shft+Enter                        (to run individual lines of code in script)
# exit()                            (to switch from Python >>> in terminal to (base))
# Ctrl+D                            to clear terminal / switch
# Ctrl+/                            to comment out multiple lines at once

# HI ANA, I figured out the problem with cv2. It was installed in python 3.9.7, not the version we were on.
# I found this by looking at the results of "pip show opencv-python"
# I used the python interpreter to swtich our environment so cv2 works now!
#debugged
# - sophie

# to run:
# (base) python detect_platform_angle.py

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


import cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import numpy as np

HOME_DIR = "/media/peterparker/9BFA-B40E/jumping_spider/spider_image_processing"

# Objective: extract a frame from one of the avi's and identify the landing platform. 

#---- Helpers ----#

# Platform image grayitization
def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#---- IMAGE-LOADING FUNCTIONS ----# - should be generalized for use outside of this context

# image_collection_path = "/media/peterparker/9BFA-B40E/jumping_spider/image-processing/input"
# files_in_dir = []

image_path = "/media/peterparker/9BFA-B40E/jumping_spider/image-processing/input/tw-00811-01_19_c2_06Apr2026.avi"
output_dir = "/media/peterparker/9BFA-B40E/jumping_spider/image-processing/output"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}
ext = Path(image_path).suffix.lower()
frame_index = 0

def load_image(image_path: str, frame_index: int = 0):
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(image_path)
        
        # if not cap.isOpened():
        #     raise FileNotFoundError(f"Cannot open video: {image_path}")
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
        
def save_fig(fig, output_dir: Path, name: str):
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [debug] saved → {path}")

def preprocess(img):
    """prepare image for edge detection - grayscale, gaussian blur, canny edges"""
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    grayed_and_blurred = cv2.Canny(blurred, 20, 60, apertureSize=3)
    return grayed_and_blurred

def detect_platform_x(img):
    """ """
        
    lines = cv2.HoughLinesP(
    img,
    rho=1,
    theta=np.pi / 360,      # 0.5° resolution
    threshold=80,
    minLineLength=img.shape[1] // 8,
    #minLineLength=max(10, img.shape[1] // 4), # requires lines to span at least 25% of image width
    maxLineGap=20
    )
        
    if lines is not None: 
        for line in lines:
             #unpack 1s array inside loop
             x1, y1, x2, y2 = line[0]
             #draw line on original image
             cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
             # attempt to omit jumping platform by limiting to right 3/4 of image
             if x1 > img.shape[1] // 4:
                #
                angle_degrees = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
                actual_angle = 90 - angle_degrees
                print(angle_degrees)
    return img

    
if __name__ == '__main__':
    INPUT_DIR = Path("/media/peterparker/'9BFA-B40E'/jumping_spider/image-processing/input")
    OUTPUT_DIR = Path("/media/peterparker/'9BFA-B40E'/jumping_spider/image-processing/output")

#---- CODE TO RUN ----#

img = load_image(image_path)
#save_fig(img, output_dir, "your_mom.png")
plt.imshow(img)
plt.axis('off')
plt.show()

newimg = preprocess(img)
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(newimg, cmap='gray')
plt.axis('off')
plt.show()

# crop image to remove black space - specific to these videos

xmin, ymin, xmax, ymax = 0, 112, 1024, 368
# xmin = 341 when jumping platform is omitted 
roi_img = newimg[ymin:ymax, xmin:xmax]

plt.imshow(roi_img, cmap='gray')
plt.axis('off')
plt.show()

lined_img = detect_platform_x(roi_img)
plt.imshow(lined_img, cmap='gray')
plt.axis('off')
plt.text(200, 100, "hough my goodness!", color = 'white')
plt.show()

MODEL_PATH = "{HOME_DIR}/automate_platform_angle_detection/yolo_platform_detection/yolo_runs/platform_detector_obb_v1-3/weights/best.pt"

def use_yolo():
    img_h, img_w = img.shape[:2]
    model = YOLO(MODEL_PATH)
    results = model(img, verbose=False)
    boxes = results[0].obb

    if boxes is None or len(boxes) == 0:
        print(" [warn] YOLO found no platform, using full image.")
        return img, (0, 0, img_w, img_h), np.ones((img_h, img_w), dtype=np.uint8) * 255

    best       = boxes[boxes.conf.argmax()]
    pts        = best.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
    conf       = float(best.conf[0])
    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))
    print(f"  [yolo] conf={conf:.2f} x={x1}–{x2}, y={y1}–{y2}")
