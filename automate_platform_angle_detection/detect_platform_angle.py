# June 8, 2026
# Ana Curtis

""" For use with Elias Lab phiddipus jumping spider data collection practices.
Trained YOLO object detection model to identify takeoff and landing platforms. 
Objective: extract a frame from each one of the avi's to identify the platforms and their angles 
to the end of producing a .csv file delineating filename along with the data. 

Inputs: 
YOLO object detection model - (trained on 25 images)
Collection of avi's 

Outputs: 
Folder of visualizations of detected objects for debugging purposes
.csv file: 

filename | takeoff platform angle | landing platform angle

"""

#______________________________

#=        = IMPORTS =         =
#______________________________

import cv2
from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO
import statistics
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import sys

HOME_DIR = "/media/peterparker/9BFA-B40E/jumping_spider/spider_image_processing"

MODEL_PATH = r"\elias-lab\spider_image_processing\automate_platform_angle_detection\yolo_platform_detection\yolo_runs\platform_detector_obb_v1-3\weights\best.pt"


@dataclass(slots=True)
class VideoResult:
    filename: str 
    takeoff_angle: float | None = None
    landing_angle: float | None = None
    error: str | None = None

#---- Helpers ----#

def to_gray(img):
    """Return a grayscale image."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def gray2bgr(img):
    """Return an image with BGR channels."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#---- IMAGE-LOADING FUNCTIONS ----# 
"""Note: Should be generalized for use outside of this context (as with everything else)."""

image_collection_path = "/media/peterparker/9BFA-B40E/jumping_spider/image-processing/input"

avi_image_path = r"\elias-lab\sh154_14_c1_16Mar2026.avi"
output_dir = r"\elias-lab\spider_image_processing\automate_platform_angle_detection\output"

frame_index = 0

def load_image(image_path: str, frame_index: int = 0):
    """Take a .avi file and extract a random frame from the middle of the video."""
    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".m4v"}
    ext = Path(avi_image_path).suffix.lower()

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
    """Save debug image."""
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [debug] saved → {path}")

def preprocess(img):
    """Prepare image for edge detection - grayscale, gaussian blur, canny edges."""
    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    grayed_and_blurred = cv2.Canny(blurred, 20, 60, apertureSize=3)
    return grayed_and_blurred

model = YOLO(MODEL_PATH)
""" Important. """

def use_yolo(image):
    """Apply YOLO model to screencap to detect platforms. Return a list of tuples: [(cropped bounding box img, label)] ."""

    image_results = []
    img_h, img_w = image.shape[:2]
    results = model(image, verbose=False)
    boxes = results[0].obb

    if boxes is None or len(boxes) == 0:
        print(" [warn] YOLO found no platform, using full image.")
        image_results.append([image, "no_id"])
    
    for box in boxes:
        pts        = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2)

        # typecast pts to int
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:,0].max()), int(pts[:, 1].max())

        print(f"  [yolo] {box[0]} x={x1:.0f}–{x2:.0f}, y={y1:.0f}–{y2:.0f}")
        print(f" x1: {x1} x2: {x2} y1: {y1} y2: {y2}")
        cropped     = image[y1:y2, x1:x2]

        # adds labels to corresponding images in dict
        cls_id = int(box.cls)
        label = model.names[cls_id]

        # changes label names
        if label == 'platform_a':
            label = 'takeoff_platform'
        else:
            label = 'landing_platform'

        image_results.append([cropped, label])

    return image_results

def detect_lines(img, label):
    """ Returns a list including the img with hough lines applied, its label, and its dominant platform angle. """

    label_data = []
    lines = cv2.HoughLinesP(
    img,
    rho=1,
    theta=np.pi / 360,      # 0.5° resolution
    threshold=80,
    minLineLength=img.shape[1] // 10/8, # requires lines to span at least 80% of image width
    maxLineGap=20
    )
        
    if lines is not None: 
        lines_angle_collection = []
        for line in lines:
             #unpack 1s array inside loop
             x1, y1, x2, y2 = line[0]
             #draw line on original image
             cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
             angle_degrees = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
             if angle_degrees < 0:
                angle_degrees = angle_degrees + 180
        print(angle_degrees)
        lines_angle_collection.append(float(angle_degrees))
        dominant_angle = statistics.mean(lines_angle_collection)
        label_data.append(img)
        label_data.append(label)
        label_data.append(dominant_angle)   
    else:
        label_data.append(img)
        label_data.append(label)
        label_data.append(0)   

    return label_data 

def visualize_images(images):
    """Take in a list ([img, label, angle]) produced by detect_lines() to produce a visualization. Use detect_lines()."""

    results = []
    for i in range(len(images)):
        img = images[i][0]
        img_label = images[i][1]
        angle = images[i][2]

        # gives the function the images
        edges = to_gray(preprocess(images[i][0]))
        hough = detect_lines(edges.copy(), img_label)[0]  # .copy() so drawing doesn't corrupt the edge map you're also displaying
        results.append([img, edges, hough])

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])  # keep indexing 2D-consistent

    row_title_y = [2] + [1.45] * (n_rows - 1)

    titles = ["Original", "Edges", "Hough lines"]
    for row, label in enumerate(results):

        # row-level heading — larger + bold, sits above the three subplot titles for this row
        row_label = images[row][1].replace('_', ' ').title()
        axes[row, 1].text(
            0.2, row_title_y[row],
            row_label,
            transform=axes[row, 1].transAxes,
            ha='center', va='bottom',
            fontsize=15, fontweight='bold', color='black'
        )
        
        for col, image in enumerate(label):
            ax = axes[row, col]
            ax.imshow(image, cmap='gray' if col > 0 else None)
            ax.set_title(titles[col])
            ax.text(
            0.6, 1.16, f"{float(angle)}",       # x centered, just above the axes (title sits a bit higher)
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=9, color='gray'
            )
            ax.axis('off')

    plt.subplots_adjust(hspace=0.9, top=0.88)
    save_fig(fig, OUTPUT_PATH, 'plaecholder for avi title + _visual.jpg')  

# ---------- ---------- #
    
if __name__ == '__main__':
    INPUT_DIR = Path("/media/peterparker/'9BFA-B40E'/jumping_spider/image-processing/input")
    OUTPUT_PATH = Path(r"C:\elias-lab\spider_image_processing\automate_platform_angle_detection\output")

#---- Main Pipeline ----#

#def main detect_angles(avi_image_path)
    joe = VideoResult(avi_image_path)
    joe.filename = avi_image_path
    
    img = load_image(avi_image_path)
    # label_list is 2 indeces of img, label """WHAT HAPPENS WHEN THERES NO PLATFORM?"""
    label_list = use_yolo(img)

    # label is a tuple including platform img and its label
    img_label_angle = []
    for label in label_list:
        # img_label_angle includes img, label, and angle in it
        img, label = label[0], label[1]
        img = preprocess(img)
        img_label_angle_entry = detect_lines(img, label)
        img_label_angle.append(img_label_angle_entry)
        if label == 'takeoff_platform':
            joe.takeoff_angle = img_label_angle_entry[2]
        else:
            joe.landing_angle = img_label_angle_entry[2]
    visualize_images(img_label_angle)    






