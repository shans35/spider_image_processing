# May 18, 2026
# Sophie Hanson
# Claude Code


"""
ruler_calibration.py
====================
Automated ruler digitization pipeline to compute pixels-per-mm.

Run this code AFTER training YOLO model (or use pre-trained YOLO model)
  1. yolo_extract_frames.py
  2. convert_yolo_labels.py
  3. create ruler.yaml config file within "yolo_dataset"
  4. train_yolo.py

Pipeline steps:
  1. Load & preprocess image
  2. Detect ruler angle via Hough lines → deskew
  3. Detect ruler ROI using YOLO object detection model → crop + inward pad
  4. Assess focus region via Laplacian variance
  5. Detect tick marks via edge projection profile
  6. Cluster tick spacings (major/minor) → compute px/mm
  7. Compute composite confidence score
  8. Produce debug visualisations at every step

Usage:
  python "ruler_calibration.py"
  or
  python ruler_calibration.py --image path/to/ruler.jpg [--output debug_out/] [--dpi 96]

Dependencies:
  pip install opencv-python numpy scipy matplotlib scikit-learn
  pip install ultralytics labelme # for YOLO object detection
"""

import argparse
import sys
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import variation
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_fig(fig, output_dir: Path, name: str):
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [debug] saved → {path}")


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Load & preprocess
# ──────────────────────────────────────────────────────────────────────────────

def load_image(image_path: str, frame_index: int = 0):
    """
    Load a single frame from an AVI (or any video), or a plain image file.
    For a single-frame AVI, frame_index=0 is always correct.
    """
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


def preprocess(img):
    """CLAHE on luminance channel to boost low-contrast rulers."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


def debug_load(img, enhanced, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(bgr2rgb(img))
    axes[0].set_title("Original image", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(bgr2rgb(enhanced))
    axes[1].set_title("After CLAHE enhancement", fontsize=13)
    axes[1].axis("off")
    fig.suptitle("Step 1 – Load & Preprocess", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, output_dir, "step1_load_preprocess.png")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Angle detection & deskew
# ──────────────────────────────────────────────────────────────────────────────

def detect_angle(img):
    """
    Uses probabilitstic Hough on Canny edges.
    Returns dominant angle (degrees) relative to horizontal.
    Confidence is based on angular consensus among detected lines.
    """
    gray = to_gray(img)
    # Gentle blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 60, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,      # 0.5° resolution
        threshold=80,
        minLineLength=img.shape[1] // 8,
        # minLineLength=max(10, img.shape[1] // 4), # requires lines to span at least 25% of image width
        maxLineGap=20
    )

    if lines is None or len(lines) < 3:
        print("     [warn] Too few Hough lines detected; assuming angle = 0°")
        return 0.0, 0.0, edges, []
    
    angles = []
    line_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep near-horizontal lines (ruler is roughly horizontal)
        if abs(angle) < 15:
            angles.append(angle)
            line_list.append((x1, y1, x2, y2))

    if not angles:
        return 0.0, 0.0, edges, []
    
    angles = np.array(angles)
    # Weighted median (rboust to outliers)
    dominant_angle = float(np.median(angles))
    # Angular std --> angle confidence (lower std = higher confidence)
    angle_std = float(np.std(angles))
    angle_confidence = max(0.0, 1.0 - min(angle_std / 10.0, 1.0))

    print(f"[2] Detected angle = {dominant_angle:.2f} deg (std={angle_std:.2f} deg, "
          f"conf={angle_confidence:.2f}, n_lines={len(angles)})")
    return dominant_angle, angle_confidence, edges, line_list


def deskew(img, angle_deg):
    """Rotate image so ruler is perfectly horizontal."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    # Compute new canvas size to avoid clipping
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def debug_angle(img, angle_deg, edges, line_list, rotated, output_dir):
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(bgr2rgb(img))
    colors = plt.cm.cool(np.linspace(0, 1, max(len(line_list), 1)))
    for i, (x1, y1, x2, y2) in enumerate(line_list[:40]):
        ax0.plot([x1, x2], [y1, y2], color=colors[min(i, len(colors)-1)], lw=1.5, alpha=0.7)
    ax0.set_title(f"Hough lines detected\n(showing up to 40)", fontsize=11)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(edges, cmap="gray")
    ax1.set_title("Canny edges", fontsize=11)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(bgr2rgb(rotated))
    ax2.set_title(f"Deskewed (angle corrected: {angle_deg:.2f}°)", fontsize=11)
    ax2.axis("off")

    fig.suptitle("Step 2 – Angle Detection & Deskew", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, output_dir, "step2_angle_deskew.png")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – ROI detection, crop & inward pad
# ──────────────────────────────────────────────────────────────────────────────

# detect ruler using trained YOLO object detection
def detect_ruler_roi(img, pad_fraction=0.05, edge_exclude=10, 
                     output_dir=None, vertical_pad=0):
    """
    vertical_pad: extra pixels to expand top and bottom of ROI box.
                  Use this for vertical rulers before rotation — expands
                  what becomes the left/right extent after 90° rotation.
    """
    from ultralytics import YOLO
    MODEL_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/yolo_runs/ruler_detector_obb_v1/weights/best.pt"

    img_h, img_w = img.shape[:2]
    model   = YOLO(MODEL_PATH)
    results = model(img, verbose=False)
    boxes   = results[0].obb

    if boxes is None or len(boxes) == 0:
        print("  [warn] YOLO found no ruler; using full image")
        return img, (0, 0, img_w, img_h), np.ones((img_h, img_w), dtype=np.uint8) * 255

    best       = boxes[boxes.conf.argmax()]
    pts        = best.xyxyxyxy[0].cpu().numpy().reshape(4, 2)
    conf       = float(best.conf[0])
    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))
    print(f"  [yolo] conf={conf:.2f} x={x1}–{x2}, y={y1}–{y2}")

    # Expand x for vertical rulers (becomes height after 90° CW rotation)
    x1 = max(x1 - vertical_pad, 0)
    x2 = min(x2 + vertical_pad, img_w)

    # edge_exclude only on x if NOT a vertical ruler (vertical_pad handles x already)
    if vertical_pad > 0:
        x0 = x1   # no additional edge exclusion on expanded sides
        x1 = x2
    else:
        x0 = max(x1 + edge_exclude, 0)
        x1 = min(x2 - edge_exclude, img_w)

    y0 = max(y1, 0)
    y1 = max(y2, 0)

    if x1 <= x0 or y1 <= y0:
        print("  [warn] YOLO box collapsed; using full image")
        return img, (0, 0, img_w, img_h), np.ones((img_h, img_w), dtype=np.uint8) * 255

    cropped     = img[y0:y1, x0:x1]
    padded_mask = np.ones((y1-y0, x1-x0), dtype=np.uint8) * 255

    print(f"[3] Ruler ROI: x={x0}–{x1}, y={y0}–{y1}, w={x1-x0}, h={y1-y0}")
    return cropped, (x0, y0, x1-x0, y1-y0), padded_mask

# def detect_ruler_roi(img, pad_fraction=0.05, edge_exclude=10, output_dir=None, image_path=None):
#     import json
#     img_h, img_w = img.shape[:2]

#     JSON_DIR = Path("/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/labeling_images")

#     # ── Option 1: use manually labeled JSON coordinates ───────────────
#     if image_path:
#         stem      = Path(image_path).stem
#         json_path = JSON_DIR / f"{stem}.json"
#         if json_path.exists():
#             with open(json_path) as f:
#                 data = json.load(f)
#             for shape in data["shapes"]:
#                 if shape["label"] == "ruler":
#                     pts = np.array(shape["points"], dtype=np.float32)
#                     x0  = max(int(np.min(pts[:, 0])) + edge_exclude, 0)
#                     x1  = min(int(np.max(pts[:, 0])) - edge_exclude, img_w)
#                     y0  = max(int(np.min(pts[:, 1])), 0)
#                     y1  = min(int(np.max(pts[:, 1])), img_h)
#                     if x1 > x0 and y1 > y0:
#                         cropped     = img[y0:y1, x0:x1]
#                         padded_mask = np.ones((y1-y0, x1-x0), dtype=np.uint8) * 255
#                         print(f"  [json] Loaded label: {json_path.name}")
#                         print(f"[3] Ruler ROI: x={x0}–{x1}, y={y0}–{y1}, w={x1-x0}, h={y1-y0}")
#                         return cropped, (x0, y0, x1-x0, y1-y0), padded_mask
#         else:
#             print(f"  [warn] No JSON found for {stem} — falling back to edge detection")

#     # ── Option 2: edge energy fallback for unlabeled images ──────────
#     gray     = to_gray(img)
#     blurred  = cv2.GaussianBlur(gray, (3, 3), 0)
#     sobelx   = cv2.Sobel(blurred.astype(np.float32), cv2.CV_64F, 1, 0, ksize=1)
#     edge_mag = np.abs(sobelx)

#     row_energy = edge_mag.sum(axis=1)
#     row_smooth = cv2.GaussianBlur(
#         row_energy.reshape(-1, 1).astype(np.float32), (51, 1), 0).flatten()
#     row_smooth /= row_smooth.max() + 1e-9

#     active = row_smooth > 0.30
#     best_start, best_end, cur_start, best_len = 0, img_h, None, 0
#     for i, v in enumerate(active):
#         if v and cur_start is None:
#             cur_start = i
#         elif not v and cur_start is not None:
#             if (i - cur_start) > best_len:
#                 best_len = i - cur_start
#                 best_start, best_end = cur_start, i
#             cur_start = None
#     if cur_start is not None and (img_h - cur_start) > best_len:
#         best_start, best_end = cur_start, img_h

#     y0 = max(best_start - 10, 0)
#     y1 = min(best_end   + 10, img_h)

#     col_smooth = cv2.GaussianBlur(
#         edge_mag[y0:y1, :].sum(axis=0).reshape(1, -1).astype(np.float32),
#         (1, 51), 0).flatten()
#     col_smooth /= col_smooth.max() + 1e-9

#     peak_mean      = np.mean(np.sort(col_smooth)[-len(col_smooth)//4:])
#     drop_threshold = peak_mean * 0.45

#     ruler_start = 0
#     for i in range(len(col_smooth) - 10):
#         if np.all(col_smooth[i:i+10] > drop_threshold):
#             ruler_start = i
#             break

#     ruler_end = len(col_smooth)
#     for i in range(ruler_start, len(col_smooth) - 20):
#         if np.all(col_smooth[i:i+20] < drop_threshold):
#             ruler_end = i
#             break

#     x0 = max(ruler_start + edge_exclude, 0)
#     x1 = min(ruler_end   - edge_exclude, img_w)

#     if x1 <= x0 or y1 <= y0:
#         print("  [warn] ROI collapsed; using full image")
#         return img, (0, 0, img_w, img_h), np.ones((img_h, img_w), dtype=np.uint8) * 255

#     cropped     = img[y0:y1, x0:x1]
#     padded_mask = np.ones((y1-y0, x1-x0), dtype=np.uint8) * 255
#     print(f"[3] Ruler ROI (edge fallback): x={x0}–{x1}, y={y0}–{y1}, w={x1-x0}, h={y1-y0}")
#     return cropped, (x0, y0, x1-x0, y1-y0), padded_mask


# def detect_ruler_roi(img, pad_fraction=0.05, edge_exclude=10, output_dir=None):
#     gray = to_gray(img)
#     img_h, img_w = gray.shape

#     # Try both threshold directions with full erode fallback loop,
#     # pick whichever yields the widest valid contour
#     best_contour = None
#     best_width   = 0

#     for binary, label in [
#         (cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV  + cv2.THRESH_OTSU)[1], "INV"),
#         (cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY      + cv2.THRESH_OTSU)[1], "NORM"),
#     ]:
#         for erode_iter in [3, 2, 1, 0]:
#             if erode_iter > 0:
#                 ek     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#                 eroded = cv2.morphologyEx(binary, cv2.MORPH_ERODE, ek,
#                                           iterations=erode_iter)
#             else:
#                 eroded = binary.copy()

#             ck     = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
#             closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, ck, iterations=2)

#             contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
#                                            cv2.CHAIN_APPROX_SIMPLE)
#             if not contours:
#                 continue

#             def ruler_score(c):
#                 cx, cy, cw, ch = cv2.boundingRect(c)
#                 area = cv2.contourArea(c)
#                 if ch == 0 or area < 500:
#                     return 0
#                 aspect = cw / ch
#                 if aspect < 1.5 or aspect > 20.0:
#                     return 0
#                 if ch < 10:
#                     return 0
#                 # Prefer contours that are wider AND taller
#                 # and whose center is in the middle 80% of image height
#                 cy_center = cy + ch / 2
#                 if cy_center < img_h * 0.10 or cy_center > img_h * 0.90:
#                     return 0
#                 return area * (cw / img_w)  # weight by fraction of image width covered

#             scored = sorted([(ruler_score(c), c) for c in contours],
#                             key=lambda x: x[0], reverse=True)
#             if scored[0][0] == 0:
#                 continue

#             # bx, by, bw, bh = cv2.boundingRect(scored[0][1])
#             # if bw >= img_w * 0.10 and bw > best_width:
#             #     best_contour = scored[0][1]
#             #     best_width   = bw
#             #     print(f"  [roi] New best: threshold={label}, erode_iter={erode_iter}, "
#             #           f"w={bw}, h={bh}")
#             # break   # found valid contour at this erode level, move to next threshold

#             bx, by, bw, bh = cv2.boundingRect(scored[0][1])
#             # Reject if contour covers >85% of image — that's background, not ruler
#             # this will be wrong for some images where the ruler spans the entire width or height
#             if bw >= img_w * 0.10 and bw > best_width and bh < img_h * 0.85:
#                 best_contour = scored[0][1]
#                 best_width   = bw
#                 print(f"  [roi] New best: threshold={label}, erode_iter={erode_iter}, "
#                       f"w={bw}, h={bh}")
#             break

#     if best_contour is None:
#         print("  [warn] No valid ruler contour found; using full image")
#         return img, (0, 0, img_w, img_h), np.ones((img_h, img_w), dtype=np.uint8) * 255

#     best = best_contour

#     x, y, w, h = cv2.boundingRect(best)

#     # ── Tighten x using column projection (exclude background on right) ──
#     roi_strip = gray[max(y, 0):min(y+h, img_h), max(x, 0):min(x+w, img_w)]
#     col_sum   = (roi_strip < roi_strip.mean()).sum(axis=0).astype(float)
#     col_sum   = col_sum / (col_sum.max() + 1e-9)
#     x, y, w, h = cv2.boundingRect(best)

#     # ── Use horizontal edge energy profile to find true ruler x-extent ──
#     roi_gray = gray[max(y, 0):min(y+h, img_h), max(x, 0):min(x+w, img_w)]
#     sobelx   = cv2.Sobel(roi_gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=1)
#     edge_col = np.abs(sobelx).sum(axis=0).astype(float)

#     # Smooth over ~5% of width to get energy envelope
#     smooth_w    = max(5, int(len(edge_col) * 0.05))
#     edge_smooth = np.convolve(edge_col, np.ones(smooth_w)/smooth_w, mode='same')
#     edge_smooth = edge_smooth / (edge_smooth.max() + 1e-9)

#     # Use the peak energy region to define the threshold —
#     # robust regardless of which side of the frame the ruler is on
#     peak_mean      = np.mean(np.sort(edge_smooth)[-len(edge_smooth)//4:])
#     drop_threshold = peak_mean * 0.45

#     # Find start: first column that rises above threshold
#     # and stays above for at least 10 consecutive columns
#     ruler_start = 0
#     for i in range(len(edge_smooth) - 10):
#         if np.all(edge_smooth[i:i+10] > drop_threshold):
#             ruler_start = i
#             break

#     # Find end: first column that drops below threshold
#     # and stays below for at least 20 consecutive columns
#     ruler_end = len(edge_smooth)
#     for i in range(ruler_start, len(edge_smooth) - 20):
#         if np.all(edge_smooth[i:i+20] < drop_threshold):
#             ruler_end = i
#             break

#     x_tight_l = max(x + ruler_start, 0)
#     x_tight_r = min(x + ruler_end,   img_w)
#     print(f"  [roi] Edge energy drop at x={ruler_end}px "
#           f"(threshold={drop_threshold:.2f}, peak_mean={peak_mean:.2f})")

#     # ── Tighten y using row energy ────────────────────────────────────
#     edge_row  = np.abs(sobelx).sum(axis=1).astype(float)
#     edge_row  = edge_row / (edge_row.max() + 1e-9)
#     active_rows = np.where(edge_row > 0.20)[0]
#     if len(active_rows) > 5:
#         y_tight_top = max(y + int(active_rows[0])  - 5, 0)
#         y_tight_bot = min(y + int(active_rows[-1]) + 5, img_h)
#     else:
#         y_tight_top = max(y - 20, 0)
#         y_tight_bot = min(y + h,  img_h)

#     x0 = max(x_tight_l + edge_exclude, 0)
#     x1 = min(x_tight_r - edge_exclude, img_w)
#     y0 = y_tight_top
#     y1 = y_tight_bot

#     if x1 <= x0 or y1 <= y0:
#         print("  [warn] Crop collapsed; using full image")
#         return img, (0, 0, img_w, img_h), np.ones((img_h, img_w), dtype=np.uint8) * 255

#     cropped     = img[y0:y1, x0:x1]
#     padded_mask = np.ones((y1-y0, x1-x0), dtype=np.uint8) * 255

#     print(f"[3] Ruler ROI: x={x0}–{x1}, y={y0}–{y1}, w={x1-x0}, h={y1-y0}")
#     return cropped, (x0, y0, x1-x0, y1-y0), padded_mask

# for vertical rulers, hard-code which ones are vertical because the below function wasn't working
def normalise_ruler_orientation(img, image_path=None):
    """
    Hard-coded orientation correction based on filename.
    Rotates known vertical rulers 90° CW.
    """
    # Files known to contain vertical rulers
    vertical_files = {
        "cali_c1_03Nov2025.avi",
        "cali_c1_31Mar2026.avi",
        "cali_c1_31Oct2025.avi",
        "cali_c2_04Oct2025.avi",
        "cali_c2_12Nov2025.avi",
        "cali_c2_31Oct2025.avi",
        "cali_c3_27Sept2025.avi",
        "cali_c4_27Sept2025.avi"
    }
    filename = Path(image_path).name if image_path else ""
    if filename in vertical_files:
        h, w = img.shape[:2]
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print(
            f"  [orient] Hard-coded vertical ruler "
            f"→ rotated 90° CW "
            f"({w}×{h} → {rotated.shape[1]}×{rotated.shape[0]})"
        )
        return rotated, True
    print("  [orient] Hard-coded horizontal ruler — no rotation")
    return img, False

# # for vertical rulers
# def normalise_ruler_orientation(img):
#     """
#     Detect ruler orientation by comparing horizontal vs vertical edge energy
#     in the cropped image. If horizontal edges dominate, the ruler is vertical
#     and needs 90° CW rotation.
#     """
#     gray   = to_gray(img)
#     # Use only the top portion where tick marks are clearest
#     h, w   = gray.shape
#     sample = gray[:min(h, h//2), :]  # top half

#     sobelx = cv2.Sobel(sample.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(sample.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
#     v_energy = np.abs(sobelx).sum()  # vertical edges → horizontal ruler
#     h_energy = np.abs(sobely).sum()  # horizontal edges → vertical ruler

#     ratio = h_energy / (v_energy + 1e-9)
#     print(f"  [orient] h_energy/v_energy ratio = {ratio:.3f} "
#           f"({'vertical ruler' if ratio > 1.2 else 'horizontal ruler'})")

#     if ratio > 1.2:
#         rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#         print(f"  [orient] Rotated 90° CW — "
#               f"was {w}×{h}, now {rotated.shape[1]}×{rotated.shape[0]}")
#         return rotated, True
#     return img, False

def debug_roi(img, roi_coords, cropped, padded_mask, output_dir):
    x, y, w, h = roi_coords
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vis = bgr2rgb(img.copy())
    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                              edgecolor="lime", facecolor="none")
    axes[0].imshow(vis)
    axes[0].add_patch(rect)
    axes[0].set_title("Detected ROI (green box)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(bgr2rgb(cropped))
    axes[1].set_title("Cropped & padded ROI", fontsize=11)
    axes[1].axis("off")

    # Add a black border so an all-white mask is still visible
    mask_display = padded_mask.copy()
    mask_display[0, :]  = 0
    mask_display[-1, :] = 0
    mask_display[:, 0]  = 0
    mask_display[:, -1] = 0
    # Black background with white mask
    black_bg = np.zeros_like(mask_display)
    display = np.stack([black_bg, black_bg, black_bg], axis=-1)
    display[mask_display > 0] = [255, 255, 255]
    axes[2].imshow(display)
    axes[2].set_title("Padded ruler mask (white = active)", fontsize=11)
    axes[2].axis("off")

    fig.suptitle("Step 3 – ROI Detection, Crop & Inward Pad",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, output_dir, "step3_roi_crop.png")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Focus assessment (Laplacian variance)
# ──────────────────────────────────────────────────────────────────────────────

def assess_focus(cropped, n_strips=40):
    """
    Slides a horizontal strip across the image height and computes
    Laplacian variance (sharpness proxy) for each strip.
    Returns: focus_scores array, best_strip_idx, focus_mask (y-range).
    """
    # Reduce strips if image is too small
    h_img = to_gray(cropped).shape[0]
    n_strips = min(n_strips, h_img // 2)
    n_strips = max(n_strips, 2)

    gray = to_gray(cropped)
    h, w = gray.shape
    strip_h = max(h // n_strips, 1)
    
    # weight the focus score toward strips with strong vertical edges (tick marks)
    # rather than general texture, so the focus band will prefer the region where tick marks are clearest
    scores = []
    for i in range(n_strips):
        y0 = i * strip_h
        y1 = min(y0 + strip_h, h)
        strip = gray[y0:y1, :]
        lap = cv2.Laplacian(strip, cv2.CV_64F)
        lap_var = float(lap.var())
        # Also score by vertical edge energy — favours tick mark regions
        sobelx = cv2.Sobel(strip, cv2.CV_64F, 1, 0, ksize=1)
        v_edge = float(np.abs(sobelx).mean())
        # Combined score: vertical edges weighted higher than general sharpness
        scores.append(lap_var * 0.3 + v_edge * 0.7)

    scores = np.array(scores)
    # Find contiguous high-focus region (top 5th percentile)
    threshold = np.percentile(scores, 5)
    in_focus = scores >= threshold

    # Find the longest contiguous in-focus run
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, f in enumerate(in_focus):
        if f:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    y_focus_start = best_start * strip_h
    y_focus_end = min((best_start + best_len) * strip_h, h)

    focus_score = float(np.mean(scores[best_start: best_start + best_len]) /
                        (np.max(scores) + 1e-9))
    print(f"[4] Focus region: y={y_focus_start}–{y_focus_end}px "
          f"(focus_score={focus_score:.2f})")
    return scores, y_focus_start, y_focus_end, focus_score, strip_h


def debug_focus(cropped, scores, y_focus_start, y_focus_end, strip_h, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(bgr2rgb(cropped))
    axes[0].axhline(y_focus_start, color="cyan", lw=2, linestyle="--",
                    label="Focus region start")
    axes[0].axhline(y_focus_end, color="magenta", lw=2, linestyle="--",
                    label="Focus region end")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("Focus region overlay", fontsize=11)
    axes[0].axis("off")

    strip_centers = np.arange(len(scores)) * strip_h + strip_h / 2
    axes[1].fill_between(strip_centers, scores, alpha=0.4, color="steelblue")
    axes[1].plot(strip_centers, scores, color="steelblue", lw=2)
    axes[1].axvspan(y_focus_start, y_focus_end, alpha=0.25, color="lime",
                    label="Selected focus band")
    axes[1].set_xlabel("Y position (px)", fontsize=10)
    axes[1].set_ylabel("Laplacian variance (sharpness)", fontsize=10)
    axes[1].set_title("Sharpness profile (horizontal strips)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Step 4 – Focus Assessment (Laplacian Variance)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, output_dir, "step4_focus.png")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – Tick mark detection
# ──────────────────────────────────────────────────────────────────────────────

def get_short_tick_band(gray, padded_mask, fraction=0.20):
    """
    Returns the row slice that contains ONLY short ticks.
    Tall ticks (1mm, 5mm, 10mm) span the full ruler height.
    Short ticks (0.5mm) only appear in the top/bottom fraction.
    
    Strategy: find the row where cumulative vertical edge energy
    first drops off — that's where short ticks end and only tall
    ticks continue.
    """
    from scipy.ndimage import uniform_filter1d

    h = gray.shape[0]
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    edge_abs = np.abs(sobelx)
    
    # Energy per row, smoothed
    row_energy = edge_abs.sum(axis=1)
    row_energy_smooth = uniform_filter1d(row_energy, size=5)
    row_energy_norm = row_energy_smooth / (row_energy_smooth.max() + 1e-9)
    
    # Find the first local minimum after the initial peak —
    # this is where short ticks stop contributing
    top_band = max(5, int(h * fraction))
    
    # Look for drop-off in the first 40% of rows
    search = row_energy_norm[:int(h * 0.4)]
    if len(search) > 3:
        # Find where energy drops below 50% of its peak in this region
        peak = search.max()
        drop = np.where(search < peak * 0.5)[0]
        if len(drop) > 0 and drop[0] > 3:
            top_band = int(drop[0])
    
    print(f"  [tick] Short tick band: y=0–{top_band}px "
          f"(ruler height={h}px)")
    return 0, top_band

# based on auto-correlation based spacing followed by peak detection on smoothed projection
def detect_ticks(cropped, y_focus_start, y_focus_end, padded_mask):
    from scipy.ndimage import uniform_filter1d

    focus_band = cropped[y_focus_start:y_focus_end, :]
    mask_band  = padded_mask[y_focus_start:y_focus_end, :]
    gray = to_gray(focus_band)

    if mask_band.shape[:2] != gray.shape[:2]:
        mask_band = cv2.resize(mask_band, (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    mask_band = mask_band.astype(np.uint8)
    gray = cv2.bitwise_and(gray, gray, mask=mask_band)

    filtered = cv2.bilateralFilter(gray, d=3, sigmaColor=25, sigmaSpace=25)

    # ── Binary edge map + Sobel ───────────────────────────────────────
    binary_edges = cv2.adaptiveThreshold(
        filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )
    sobelx   = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=1)
    edge_mag = (binary_edges > 0).astype(np.float32) * np.abs(sobelx)

    # ── Edge band projections (top + bottom 20%) ──────────────────────
    short_band_end       = max(3, int(edge_mag.shape[0] * 0.20))
    profile_short_top    = edge_mag[:short_band_end, :].sum(axis=0).astype(float)
    profile_short_bottom = edge_mag[-short_band_end:, :].sum(axis=0).astype(float)
    profile_short        = profile_short_top + profile_short_bottom

    profile_full       = edge_mag.sum(axis=0).astype(float)
    profile_short_norm = profile_short / (profile_short.max() + 1e-9)
    profile_full_norm  = profile_full  / (profile_full.max()  + 1e-9)

    # Mixed profile for coarse detection and ruler boundary
    profile_raw      = 0.4 * profile_full_norm + 0.6 * profile_short_norm
    # Pure edge-band profile for fine pass — no dilution from full height
    profile_raw_fine = profile_short_norm.copy()

    print(f"  [tick] Short tick band: y=0–{short_band_end}px and "
          f"y={edge_mag.shape[0]-short_band_end}–{edge_mag.shape[0]}px "
          f"(ruler height={edge_mag.shape[0]}px)")

    # ── Ruler boundary ────────────────────────────────────────────────
    energy_envelope = uniform_filter1d(profile_raw,
                                       size=max(5, int(len(profile_raw) * 0.05)))
    energy_envelope /= energy_envelope.max() + 1e-9
    active = np.where(energy_envelope > 0.20)[0]

    ruler_left  = int(active[0])  if len(active) > 10 else 0
    ruler_right = int(active[-1]) if len(active) > 10 else len(profile_raw)

    if ruler_right - ruler_left < 10:
        ruler_left  = 0
        ruler_right = len(profile_raw)
        print(f"  [tick] Boundary too narrow — using full profile width")

    profile_raw[:ruler_left]       = 0
    profile_raw[ruler_right:]      = 0
    profile_raw_fine[:ruler_left]  = 0
    profile_raw_fine[ruler_right:] = 0
    edge_mag[:, :ruler_left]       = 0
    edge_mag[:, ruler_right:]      = 0
    gray[:, :ruler_left]           = 128
    gray[:, ruler_right:]          = 128
    print(f"  [tick] Ruler boundary: x={ruler_left}–{ruler_right}px")

    # ── Step 1: estimate rough spacing ───────────────────────────────
    if len(profile_raw) < 10:
        print(f"  [tick] Profile too short ({len(profile_raw)}px) — skipping")
        empty = np.array([], dtype=float)
        return (empty, np.zeros(1), {}, np.zeros((1,1), dtype=np.float32),
                np.zeros(1), empty, empty, np.zeros(1), 7,
                np.zeros(1), np.zeros(1))

    rough_win    = max(5, int(len(profile_raw) / 300) | 1)
    rough_smooth = savgol_filter(profile_raw, window_length=rough_win, polyorder=2)
    rough_smooth = (rough_smooth - rough_smooth.min()) / \
                   (rough_smooth.max() + 1e-9)
    rough_peaks, _ = find_peaks(rough_smooth, height=0.05,
                                distance=3, prominence=0.01)
    if len(rough_peaks) > 2:
        rough_spacing = float(np.median(np.diff(rough_peaks)))
    else:
        rough_spacing = len(profile_raw) / 100.0
    rough_spacing = min(rough_spacing, len(profile_raw) * 0.05)
    rough_spacing = max(rough_spacing, 3.0)

    if rough_spacing < 6.0 and len(rough_peaks) > 4:
        alt_spacing = float(np.median(np.diff(rough_peaks[::2])))
        if alt_spacing > rough_spacing * 1.5:
            print(f"  [tick] Edge-pair detected (rough={rough_spacing:.1f}px) — "
                  f"correcting to {alt_spacing:.1f}px")
            rough_spacing = alt_spacing

    # ── Step 2: local normalisation ───────────────────────────────────
    local_mean = uniform_filter1d(profile_raw,
                                  size=max(5, int(rough_spacing * 3))) + 1e-9
    norm_raw   = profile_raw / local_mean

    # Separate normalisation for fine pass using edge-band only
    local_mean_fine = uniform_filter1d(profile_raw_fine,
                                       size=max(5, int(rough_spacing * 3))) + 1e-9
    norm_raw_fine   = profile_raw_fine / local_mean_fine

    # ── Step 3: autocorrelation for true tick period ──────────────────
    light_win     = max(3, int(rough_spacing * 0.5) | 1)
    profile_light = savgol_filter(norm_raw, window_length=light_win, polyorder=2)
    profile_light = (profile_light - profile_light.min()) / \
                    (profile_light.max() - profile_light.min() + 1e-9)

    true_spacing = int(rough_spacing)
    autocorr = np.real(np.fft.ifft(np.abs(np.fft.fft(profile_light))**2))
    autocorr = autocorr[:len(autocorr)//2]
    autocorr /= autocorr.max() + 1e-9
    ac_peaks, _ = find_peaks(autocorr[3:], prominence=0.05)
    if len(ac_peaks) > 0:
        true_spacing = int(ac_peaks[0]) + 3
        print(f"  [tick] Autocorr spacing = {true_spacing}px "
              f"(rough = {rough_spacing:.1f}px)")
    else:
        print(f"  [tick] Using rough spacing = {true_spacing}px")

    if true_spacing > rough_spacing * 2.5:
        true_spacing = int(rough_spacing)
        print(f"  [tick] Autocorr locked onto harmonic (too high) — "
              f"reverting to rough spacing = {true_spacing}px")
    elif true_spacing < rough_spacing * 0.6:
        true_spacing = int(rough_spacing)
        print(f"  [tick] Autocorr locked onto sub-harmonic (too low) — "
              f"reverting to rough spacing = {true_spacing}px")

    # ── Step 4: two-pass peak detection ──────────────────────────────
    # Adaptive parameters derived from true_spacing — no hardcoding.
    expected_half_spacing = true_spacing / 2.0

    # Gate: if 0.5mm ticks aren't physically resolvable, skip fine pass entirely
    FINE_PASS_MIN_PX = 5  # minimum resolvable 0.5mm spacing in pixels
    run_fine_pass = expected_half_spacing >= FINE_PASS_MIN_PX

    if run_fine_pass:
        fine_win      = max(3, int(expected_half_spacing * 0.4) | 1)
        min_dist_fine = max(2, int(expected_half_spacing * 0.65))
        print(f"  [tick] Fine pass enabled — expected 0.5mm spacing = "
          f"{expected_half_spacing:.1f}px, "
          f"fine_win={fine_win}, min_dist={min_dist_fine}")
    else:
        fine_win      = None
        min_dist_fine = None
        print(f"  [tick] Fine pass skipped — expected 0.5mm spacing "
            f"({expected_half_spacing:.1f}px) < {FINE_PASS_MIN_PX}px threshold")

    profile_fine = savgol_filter(norm_raw_fine,
                                window_length=fine_win if run_fine_pass else 3,
                                polyorder=2)
    profile_fine = (profile_fine - profile_fine.min()) / \
                (profile_fine.max() - profile_fine.min() + 1e-9)
    
    # ── Diagnostic ───────────────────────────────────────────────────
    from scipy.signal import peak_prominences
    _diag_dist = min_dist_fine if min_dist_fine is not None else max(2, int(expected_half_spacing * 0.65))
    all_peaks_diag, _ = find_peaks(profile_fine, height=0.03, distance=_diag_dist)

    prominences_diag  = peak_prominences(profile_fine, all_peaks_diag)[0]
    print(f"  [tick] norm_raw_fine stats: min={norm_raw_fine.min():.3f}, "
          f"max={norm_raw_fine.max():.3f}, mean={norm_raw_fine.mean():.3f}")
    print(f"  [tick] profile_fine stats: min={profile_fine.min():.3f}, "
          f"max={profile_fine.max():.3f}, mean={profile_fine.mean():.3f}")
    print(f"  [tick] true_spacing={true_spacing}px, fine_win={fine_win}px, "
          f"min_dist_fine={min_dist_fine}px")
    peaks_no_filter, _   = find_peaks(profile_fine)
    peaks_height_only, _ = find_peaks(profile_fine, height=0.05)
    peaks_dist_only, _   = find_peaks(profile_fine, distance=_diag_dist)
    print(f"  [tick] Peaks with zero filters: {len(peaks_no_filter)}")
    print(f"  [tick] Peaks with height>0.05 only: {len(peaks_height_only)}")
    print(f"  [tick] Peaks with distance only: {len(peaks_dist_only)}")
    if len(all_peaks_diag) > 0:
        print(f"  [tick] Fine pass all peaks: n={len(all_peaks_diag)}, "
            f"prominence min={prominences_diag.min():.4f}, "
            f"median={np.median(prominences_diag):.4f}, "
            f"max={prominences_diag.max():.4f}")
        print(f"  [tick] Peaks with prominence < 0.005: "
            f"{np.sum(prominences_diag < 0.005)} / {len(all_peaks_diag)}")
        print(f"  [tick] Peaks with prominence < 0.02: "
            f"{np.sum(prominences_diag < 0.02)} / {len(all_peaks_diag)}")
    else:
        print(f"  [tick] Fine pass all peaks: n=0")

    # ── Fine pass peak detection ──────────────────────────────────────
    if run_fine_pass:
        peaks_fine, _ = find_peaks(
            profile_fine,
            height=0.05,
            distance=min_dist_fine,
            prominence=0.005,
        )
    else:
        peaks_fine = np.array([], dtype=np.float64)

    # ── Coarse pass ───────────────────────────────────────────────────
    coarse_win     = max(3, int(true_spacing * 0.8) | 1)
    profile_coarse = savgol_filter(norm_raw, window_length=coarse_win, polyorder=2)
    profile_coarse = (profile_coarse - profile_coarse.min()) / \
                     (profile_coarse.max() - profile_coarse.min() + 1e-9)
    min_dist_coarse = max(3, int(true_spacing * 0.90))
    peaks_coarse, _ = find_peaks(
        profile_coarse,
        height=0.05,
        distance=min_dist_coarse,
        prominence=0.02,
    )

    # ── Decide which pass to use ──────────────────────────────────────
    if len(peaks_coarse) > 2 and len(peaks_fine) > 2:
        ratio_count    = len(peaks_fine) / len(peaks_coarse)
        spacing_fine   = float(np.median(np.diff(np.sort(peaks_fine))))
        spacing_coarse = float(np.median(np.diff(np.sort(peaks_coarse))))
        spacing_ratio  = spacing_coarse / (spacing_fine + 1e-9)

        print(f"  [tick] Fine={len(peaks_fine)} ticks (spacing={spacing_fine:.1f}px), "
              f"Coarse={len(peaks_coarse)} ticks (spacing={spacing_coarse:.1f}px), "
              f"count_ratio={ratio_count:.2f}, spacing_ratio={spacing_ratio:.2f}")

        if 1.3 < spacing_ratio < 3.0 and 1.3 < ratio_count < 3.0:
            peaks          = peaks_fine
            profile_merged = profile_fine
            original_spacing = true_spacing
            true_spacing   = true_spacing / spacing_ratio
            print(f"  [tick] Using FINE pass — 0.5mm ticks detected, "
                  f"recalibrated true_spacing={true_spacing}px")
        else:
            peaks            = peaks_coarse
            profile_merged   = profile_coarse
            original_spacing = true_spacing
            print(f"  [tick] Defaulting to COARSE pass (insufficient peaks for comparison)")
    else:
        peaks          = peaks_coarse
        profile_merged = profile_coarse
        original_spacing = true_spacing
        print(f"  [tick] Defaulting to COARSE pass (insufficient peaks for comparison)")

     # Force float64 regardless of which branch was taken
    peaks = np.asarray(peaks, dtype=np.float64)

    # Step 5? centroid refinement
    if peaks is peaks_fine:
        half_win = max(1, min(2, int(expected_half_spacing * 0.15)))
        refined_centers = []
        for tc in peaks:
            if half_win == 0:
                refined_centers.append(float(tc))
                continue
            lo  = max(0, int(tc) - half_win)
            hi  = min(len(profile_fine), int(tc) + half_win + 1)
            win = profile_fine[lo:hi]
            if len(win) == 0:
                refined_centers.append(float(tc))
                continue
            positions = np.arange(lo, hi, dtype=float)
            centroid  = float(np.sum(positions * win) / (np.sum(win) + 1e-9))
            refined_centers.append(centroid)
        tick_centers = np.array(sorted(refined_centers), dtype=float)
        print(f"  [tick] Step 5 fine pass centroid refinement: "
            f"{len(tick_centers)} ticks")
    else:
        # Coarse pass centroid refinement on profile_coarse
        half_win = max(1, min(3, int(true_spacing * 0.15)))
        refined_centers = []
        for tc in peaks_coarse:
            lo  = max(0, int(tc) - half_win)
            hi  = min(len(profile_coarse), int(tc) + half_win + 1)
            win = profile_coarse[lo:hi]
            if len(win) == 0:
                refined_centers.append(float(tc))
                continue
            positions = np.arange(lo, hi, dtype=np.float64)
            centroid  = float(np.sum(positions * win) / (np.sum(win) + 1e-9))
            refined_centers.append(centroid)
        tick_centers = np.array(refined_centers, dtype=np.float64)
        original_spacing = true_spacing
        print(f"  [tick] Step 5 coarse pass centroid refinement: "
            f"{len(tick_centers)} ticks")
    
    # Remove duplicates — use original_spacing not recalibrated true_spacing
    if len(tick_centers) > 1:
        spacings = np.diff(tick_centers)
        keep     = np.concatenate([[True], spacings >= original_spacing * 0.20])
        tick_centers = tick_centers[keep].astype(float)
        print(f"  [tick] After duplicate removal: {len(tick_centers)} ticks "
              f"(threshold={original_spacing * 0.20:.2f}px)")
        # diagnostic
        sp = np.diff(tick_centers)
        print(f"  [tick] Post-removal spacings: "
              f"min={sp.min():.2f}, p25={np.percentile(sp,25):.2f}, "
              f"median={np.median(sp):.2f}, p75={np.percentile(sp,75):.2f}, "
              f"max={sp.max():.2f}")
        
    print(f"  [tick] Subpixel check — integer positions: "
          f"{np.sum(np.abs(tick_centers - np.round(tick_centers)) < 0.01)} "
          f"/ {len(tick_centers)}")
        
    # ── Step 6: gap fill ──────────────────────────────────────────────
    if len(tick_centers) > 5:
        spacings = np.diff(tick_centers)
        q25      = float(np.percentile(spacings, 25))
        pre_fill_median = float(np.median(spacings[spacings <= q25 * 2.5]))
        if pre_fill_median < 3.0:
            pre_fill_median = float(np.median(spacings))

        print(f"  [tick] Gap fill reference spacing = {pre_fill_median:.2f}px "
              f"(overall median = {float(np.median(spacings)):.2f}px)")

        filled = list(tick_centers)
        i = 0
        while i < len(filled) - 1:
            gap       = filled[i+1] - filled[i]
            n_missing = round(gap / pre_fill_median) - 1
            if 0 < n_missing <= 3:
                for j in range(1, n_missing + 1):
                    filled.append(filled[i] + j * gap / (n_missing + 1))
            i += 1

        tick_centers = np.array(sorted(filled), dtype=float)
        spacings     = np.diff(tick_centers)
        keep         = np.concatenate([[True],
                                       spacings >= pre_fill_median * 0.35])
        tick_centers = tick_centers[keep].astype(float)
        print(f"  [tick] After Step 6 gap fill: {len(tick_centers)} ticks, "
              f"median spacing={float(np.median(np.diff(tick_centers))):.2f}px")

    print(f"  [tick] Pre-cleanup median spacing = "
          f"{float(np.median(np.diff(tick_centers))):.2f}px, "
          f"n={len(tick_centers)}")

    # ── Step 7: enforce periodicity ───────────────────────────────────
    if len(tick_centers) > 5:
        spacings       = np.diff(tick_centers)
        median_spacing = float(np.median(spacings))

        cleaned = [tick_centers[0]]
        for i in range(1, len(tick_centers)):
            gap = tick_centers[i] - cleaned[-1]
            if gap < median_spacing * 0.6:
                idx_prev = min(int(cleaned[-1]), len(profile_merged)-1)
                idx_curr = min(int(tick_centers[i]), len(profile_merged)-1)
                if profile_merged[idx_curr] > profile_merged[idx_prev]:
                    cleaned[-1] = tick_centers[i]
            else:
                cleaned.append(tick_centers[i])
        tick_centers = np.array(cleaned, dtype=float)

        if len(tick_centers) > 5:
            spacings       = np.diff(tick_centers)
            median_spacing = float(np.median(spacings))
            keep = [True]
            for i, sp in enumerate(spacings):
                ratio = sp / median_spacing
                if 0.4 < ratio < 1.6 or 1.6 < ratio < 3.5:
                    keep.append(True)
                else:
                    keep.append(False)
            tick_centers = tick_centers[np.array(keep)]

        print(f"  [tick] After Step 7 periodicity: {len(tick_centers)} ticks, "
              f"median spacing={float(np.median(np.diff(tick_centers))):.2f}px")
        print(f"  [tick] After Step 7 dtype={tick_centers.dtype}, sample={tick_centers[:5].tolist()}")

    tick_centers = np.asarray(tick_centers, dtype=np.float64)
    print(f"  [tick] tick_centers dtype={tick_centers.dtype}, sample={tick_centers[:5].tolist()}")

    profile_display  = profile_merged
    major_peaks      = tick_centers
    minor_peaks = np.array([], dtype=np.float64)
    profile_residual = profile_display.copy()

    print(f"[5] Detected {len(tick_centers)} total ticks")
    return (tick_centers, profile_display, {}, edge_mag,
            profile_residual, major_peaks, minor_peaks,
            profile_merged, true_spacing,
            profile_full, profile_short)


def debug_ticks(cropped, y_focus_start, y_focus_end,
                tick_positions, profile, edge_mag,
                profile_residual, major_peaks, minor_peaks, output_dir):
    focus_band = cropped[y_focus_start:y_focus_end, :]
    n_cols = focus_band.shape[1]
    x_axis = np.arange(n_cols)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.05)

    # ── Panel 1: ruler image with tick overlays ───────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(bgr2rgb(focus_band), aspect='auto',
               extent=[0, n_cols, focus_band.shape[0], 0])
    for t in tick_positions:
        ax0.axvline(x=t, color='red', lw=0.8, alpha=0.8)
    ax0.set_xlim(0, n_cols)
    ax0.set_title(f"Detected ticks — {len(tick_positions)} total", fontsize=11)
    ax0.set_xticklabels([])
    ax0.set_ylabel("Y (px)", fontsize=9)

    # ── Panel 2: smoothed projection profile ─────────────────────────
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(x_axis, profile, color='darkorange', lw=1.5, label='Smoothed projection')
    ax1.scatter(tick_positions, profile[np.array(tick_positions, dtype=int)],
                c='red', s=30, zorder=5, label='Detected peaks')
    ax1.set_xlim(0, n_cols)
    ax1.set_ylabel("Norm. edge energy", fontsize=9)
    ax1.set_title("Smoothed projection profile", fontsize=11)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])

    # ── Panel 3: vertical edge magnitude map ─────────────────────────
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.imshow(edge_mag, cmap='inferno', aspect='auto',
               extent=[0, n_cols, edge_mag.shape[0], 0])
    for t in tick_positions:
        ax2.axvline(x=t, color='red', lw=0.8, alpha=0.6)
    ax2.set_xlim(0, n_cols)
    ax2.set_xlabel("X position (px)", fontsize=10)
    ax2.set_title("Vertical edge magnitude map", fontsize=11)

    fig.suptitle("Step 5 – Tick Mark Detection", fontsize=15, fontweight="bold")
    save_fig(fig, Path(output_dir), "step5_tick_detection.png")


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 – Spacing clustering & px/mm calculation
# ──────────────────────────────────────────────────────────────────────────────

# scale-invariant classification of tick height using Canny edges
def classify_ticks_by_height(cropped, tick_positions, true_spacing):
    """
    For each detected tick position, measure the vertical extent of the
    Canny edge response. Tall ticks = 1mm, short ticks = 0.5mm.
    Returns: tick_positions_05mm, tick_positions_1mm, height_profile
    """
    gray = to_gray(cropped)
    h, w = gray.shape

    # Auto-tune Canny thresholds from gradient magnitude distribution
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Use Otsu on gradient magnitude for adaptive thresholds
    grad_8u = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(grad_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh  = float(otsu_thresh) * 0.4
    high_thresh = float(otsu_thresh) * 1.0
    
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    
    # For each tick position, measure vertical extent of edge response
    # in a narrow column window around the tick
    col_half_win = max(1, int(true_spacing * 0.25))
    tick_heights = []
    
    for tc in tick_positions:
        col = int(round(tc))
        col_lo = max(0, col - col_half_win)
        col_hi = min(w, col + col_half_win + 1)
        
        # Sum edge response across the column window
        col_edges = edges[:, col_lo:col_hi].sum(axis=1).astype(float)
        
        # Find contiguous rows with edge response
        active_rows = np.where(col_edges > 0)[0]
        if len(active_rows) == 0:
            tick_heights.append(0)
            continue
        
        # Vertical extent = max row - min row in the active region
        tick_heights.append(int(active_rows[-1] - active_rows[0] + 1))
    
    tick_heights = np.array(tick_heights, dtype=float)
    
    if len(tick_heights) < 4:
        # Not enough data — assume all 1mm
        return tick_positions, np.array([]), tick_heights
    
    # Classify by height: bimodal distribution expected
    # Short ticks cluster at ~20-40% of ruler height
    # Tall ticks cluster at ~60-100% of ruler height
    height_threshold = float(np.percentile(tick_heights, 50))
    
    mask_tall  = tick_heights >= height_threshold
    mask_short = tick_heights <  height_threshold
    
    ticks_1mm  = tick_positions[mask_tall]
    ticks_05mm = tick_positions[mask_short]
    
    print(f"  [height] Canny height classification: "
          f"{len(ticks_1mm)} × 1mm, {len(ticks_05mm)} × 0.5mm "
          f"(threshold={height_threshold:.1f}px, "
          f"ruler_h={h}px, col_win={col_half_win}px)")
    print(f"  [height] Height distribution: "
          f"min={tick_heights.min():.0f}, "
          f"p25={np.percentile(tick_heights,25):.0f}, "
          f"median={np.median(tick_heights):.0f}, "
          f"p75={np.percentile(tick_heights,75):.0f}, "
          f"max={tick_heights.max():.0f}")
    
    return ticks_1mm, ticks_05mm, tick_heights

def compute_pixels_per_mm(tick_positions, detected_mm=1.0):
    if len(tick_positions) < 2:
        raise ValueError("Need at least 2 tick positions to compute spacing")

    spacings = np.diff(np.sort(tick_positions)).astype(float)
    print(f"  [px/mm] First 10 spacings: {spacings[:10]}")
    print(f"  [px/mm] Tick positions sample: {tick_positions[:5]}")

    # ── Outlier removal ───────────────────────────────────────────────
    q1, q3 = np.percentile(spacings, [25, 75])
    iqr = q3 - q1
    valid_mask     = (spacings >= q1 - 3*iqr) & (spacings <= q3 + 3*iqr)
    spacings_clean = spacings[valid_mask]
    if len(spacings_clean) < 2:
        spacings_clean = spacings

    median_spacing_px = float(np.median(spacings_clean))
    px_per_mm         = median_spacing_px / detected_mm
    tick_uniformity   = max(0.0, 1.0 - min(float(np.std(spacings_clean) /
                            (np.mean(spacings_clean) + 1e-9)), 1.0))

    print(f"[6] Detected tick type     : {detected_mm}mm")
    print(f"    Median spacing         : {median_spacing_px:.2f}px")
    print(f"    px/mm                  : {px_per_mm:.4f}")
    print(f"    Tick uniformity        : {tick_uniformity:.2f}")

    return px_per_mm, {0.5: median_spacing_px/0.5, 1.0: median_spacing_px/1.0}, \
           spacings_clean, tick_uniformity, detected_mm

def debug_spacing(tick_positions, spacings, spacings_clean,
                  chosen, px_mm_candidates, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Spacing distribution
    axes[0].hist(spacings, bins=30, color="steelblue", alpha=0.7, label="All spacings")
    axes[0].hist(spacings_clean, bins=30, color="orange", alpha=0.5,
                 label="After outlier removal")
    for cm in chosen["cluster_means"]:
        axes[0].axvline(cm, color="red", lw=2, linestyle="--")
    axes[0].set_xlabel("Inter-tick spacing (px)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("Tick spacing distribution", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 2) Clustered spacings colour-coded
    if len(chosen["cluster_means"]) > 1:
        sc = axes[1].scatter(range(len(spacings_clean)), spacings_clean,
                             c=chosen["labels"], cmap="tab10", s=20)
        plt.colorbar(sc, ax=axes[1], label="Cluster")
    else:
        axes[1].scatter(range(len(spacings_clean)), spacings_clean,
                        color="steelblue", s=20)
    for cm in chosen["cluster_means"]:
        axes[1].axhline(cm, color="red", lw=1.5, linestyle="--",
                        label=f"Cluster mean {cm:.1f}px")
    axes[1].set_xlabel("Spacing index", fontsize=10)
    axes[1].set_ylabel("Spacing (px)", fontsize=10)
    axes[1].set_title("Clustered tick spacings", fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # 3) px/mm summary bar chart
    labels = [f"{mm}mm/div" for mm in px_mm_candidates]
    values = list(px_mm_candidates.values())
    bars = axes[2].barh(labels, values, color=["#2196F3", "#FF5722"][:len(values)])
    for bar, val in zip(bars, values):
        axes[2].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f} px/mm", va="center", fontsize=11)
    axes[2].set_xlabel("Pixels per mm", fontsize=10)
    axes[2].set_title("px/mm estimates (by tick assumption)", fontsize=11)
    axes[2].grid(True, alpha=0.3, axis="x")

    fig.suptitle("Step 6 – Spacing Analysis & px/mm Calculation",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, output_dir, "step6_spacing_pxmm.png")


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 – Composite confidence score
# ──────────────────────────────────────────────────────────────────────────────

def compute_confidence(angle_conf, focus_score, tick_uniformity, n_ticks):
    """
    Composite confidence in [0, 1].
    Components:
      - angle_conf    : how well-defined the ruler angle was
      - focus_score   : relative sharpness in focus band
      - tick_uniform  : 1 – CoV of tick spacings (closer to 1 = more regular)
      - tick_count    : saturates at ~50 ticks
    """
    tick_count_score = min(n_ticks / 50.0, 1.0)

    weights = {
        "angle":     0.20,
        "focus":     0.25,
        "uniformity": 0.40,
        "count":     0.15,
    }
    composite = (
        weights["angle"]     * angle_conf      +
        weights["focus"]     * focus_score     +
        weights["uniformity"]* tick_uniformity +
        weights["count"]     * tick_count_score
    )

    breakdown = {
        "angle_confidence":  round(angle_conf, 3),
        "focus_score":       round(focus_score, 3),
        "tick_uniformity":   round(tick_uniformity, 3),
        "tick_count_score":  round(tick_count_score, 3),
        "composite":         round(composite, 3),
    }
    return composite, breakdown


def debug_confidence(breakdown, px_per_mm, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Radar / bar chart of components
    components = ["angle_confidence", "focus_score", "tick_uniformity", "tick_count_score"]
    values = [breakdown[c] for c in components]
    labels = ["Angle\nconfidence", "Focus\nscore", "Tick\nuniformity", "Tick\ncount"]
    colors = ["#4CAF50" if v >= 0.7 else "#FF9800" if v >= 0.4 else "#f44336"
              for v in values]

    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylim(0, 1.05)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    axes[0].axhline(0.7, color="gray", lw=1, linestyle="--", alpha=0.6, label="0.7 threshold")
    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_title("Confidence component breakdown", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Summary panel
    composite = breakdown["composite"]
    colour = "#4CAF50" if composite >= 0.7 else "#FF9800" if composite >= 0.4 else "#f44336"
    rating = "HIGH" if composite >= 0.7 else "MEDIUM" if composite >= 0.4 else "LOW"

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")

    # Big composite circle
    circle = plt.Circle((0.5, 0.62), 0.25, color=colour, alpha=0.85)
    axes[1].add_patch(circle)
    axes[1].text(0.5, 0.62, f"{composite:.2f}", ha="center", va="center",
                 fontsize=28, fontweight="bold", color="white")
    axes[1].text(0.5, 0.33, f"Confidence: {rating}", ha="center",
                 fontsize=16, fontweight="bold", color=colour)
    axes[1].text(0.5, 0.20, f"Result:  {px_per_mm:.3f} px/mm", ha="center",
                 fontsize=13, color="#333333")
    axes[1].text(0.5, 0.09, f"(assuming 1 mm / tick)", ha="center",
                 fontsize=10, color="#777777")
    axes[1].set_title("Final result summary", fontsize=12)

    fig.suptitle("Step 7 – Confidence Score & Final Result",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, output_dir, "step7_confidence.png")


# ──────────────────────────────────────────────────────────────────────────────
# Step 8 – Full-pipeline summary plot
# ──────────────────────────────────────────────────────────────────────────────

def debug_summary(image_path, deskewed, cropped, padded_mask,
                  tick_positions, profile,
                  profile_full, profile_short,
                  y_focus_start, y_focus_end,
                  angle_deg, px_per_mm, confidence, breakdown,
                  roi_coords, output_dir, detected_mm=1.0,
                  ticks_1mm=None, ticks_05mm=None, height_profile=None):
    avi_name = Path(image_path).name
    n_cols   = cropped.shape[1]
    x_axis   = np.arange(n_cols)
    conf_col = "#4CAF50" if confidence >= 0.7 else \
               "#FF9800" if confidence >= 0.4 else "#f44336"

    short_band_end = max(2, int(cropped.shape[0] * 0.20))

    fig = plt.figure(figsize=(20, 22))
    gs = GridSpec(7, 1, figure=fig,
              height_ratios=[1.2, 1.2, 1.2, 1.2, 1.4, 1.2, 1.0],
              hspace=0.06)

    # ── Row 0: deskewed + CLAHE ───────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(bgr2rgb(deskewed))
    roi_x, roi_y, roi_w, roi_h = roi_coords
    rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h,
                              linewidth=2, edgecolor="lime", facecolor="none")
    ax0.add_patch(rect)
    ax0.set_title(f"Deskewed & CLAHE enhanced  (angle corrected: {angle_deg:.2f}°)",
                  fontsize=12, pad=6)
    ax0.axis("off")

    # ── Row 1: cropped ROI ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(bgr2rgb(cropped), aspect="auto",
               extent=[0, n_cols, cropped.shape[0], 0])
    ax1.set_xlim(0, n_cols)
    ax1.set_ylabel("Y (px)", fontsize=8)
    ax1.set_title("Cropped & padded ROI", fontsize=11)
    ax1.set_xticklabels([])
    ax1.tick_params(axis='y', labelsize=7)

    # ── Row 2: focus region overlay with band lines ───────────────────
    # ── Row 2: focus region overlay with selected band ────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.imshow(bgr2rgb(cropped), aspect="auto",
               extent=[0, n_cols, cropped.shape[0], 0])
    ax3.axhspan(y_focus_start, y_focus_end, color="cyan", alpha=0.25)
    ax3.axhline(y_focus_start, color="cyan",        lw=1.5, linestyle="--")
    ax3.axhline(y_focus_end,   color="deepskyblue", lw=1.5, linestyle="--")
    ax3.axhspan(0, short_band_end,
                color="limegreen", alpha=0.35)
    ax3.axhline(short_band_end, color="limegreen", lw=1.5, linestyle=":")
    ax3.axhspan(cropped.shape[0] - short_band_end, cropped.shape[0],
                color="limegreen", alpha=0.35)
    ax3.axhline(cropped.shape[0] - short_band_end,
                color="limegreen", lw=1.5, linestyle=":")
    ax3.set_xlim(0, n_cols)
    ax3.set_ylabel("Y (px)", fontsize=8)
    ax3.set_title(
        f"Focus region overlay  |  "
        f"edge bands y=0–{short_band_end}px and "
        f"y={cropped.shape[0]-short_band_end}–{cropped.shape[0]}px",
        fontsize=11)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="cyan",      lw=1.5, linestyle="--",
               label=f"Focus band  y={y_focus_start}–{y_focus_end}px"),
        Line2D([0], [0], color="limegreen", lw=1.5, linestyle=":",
               label=f"Edge bands  y=0–{short_band_end}px and "
                     f"y={cropped.shape[0]-short_band_end}–{cropped.shape[0]}px"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.6)
    ax3.set_xticklabels([])
    ax3.tick_params(axis='y', labelsize=7)

    # ── Row 3: smoothed projection profile (merged, with tick markers) ─
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(x_axis, profile, color="darkorange", lw=1.2)
    ax4.scatter(tick_positions, profile[np.array(tick_positions, dtype=int)],
                c="red", s=20, zorder=5)
    ax4.set_xlim(0, n_cols)
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("Edge energy", fontsize=8)
    ax4.set_title("Smoothed projection profile", fontsize=11)
    ax4.grid(True, alpha=0.25)
    ax4.set_xticklabels([])
    ax4.tick_params(axis='y', labelsize=7)

    # ── Row 4: all three projection profiles overlaid ─────────────────
    # Normalize profile_short for display (it may have different scale)
    profile_short_norm = profile_short / (profile_short.max() + 1e-9)
    profile_full_norm  = profile_full  / (profile_full.max()  + 1e-9)

    ax4b = fig.add_subplot(gs[4], sharex=ax1)
    ax4b.plot(x_axis, profile_full_norm,
              color="steelblue",  lw=0.9, alpha=0.7, label="Full height")
    ax4b.plot(x_axis, profile_short_norm,
              color="limegreen",  lw=0.9, alpha=0.8, label="Edge bands (top+bottom)")
    ax4b.plot(x_axis, profile,
              color="darkorange", lw=1.2, alpha=0.9, label="Merged (used for detection)")
    norm_raw_display = profile / (profile.max() + 1e-9)  # proxy — or pass norm_raw directly
    ax4b.plot(x_axis, norm_raw_display,
              color="magenta", lw=0.8, alpha=0.6, label="norm_raw (pre-smoothing)")
    ax4b.set_xlim(0, n_cols)
    ax4b.set_ylim(0, 1.05)
    ax4b.set_ylabel("Edge energy", fontsize=8)
    ax4b.set_title(
        f"Projection profiles  |  edge bands = top+bottom 25%  "
        f"({short_band_end}px each, ruler h={cropped.shape[0]}px)",
        fontsize=11)
    ax4b.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax4b.grid(True, alpha=0.25)
    ax4b.set_xticklabels([])
    ax4b.tick_params(axis='y', labelsize=7)

    # ── Row 5: detected ticks overlaid on cropped image ───────────────
    ax5 = fig.add_subplot(gs[5], sharex=ax1)
    ax5.imshow(bgr2rgb(cropped), aspect="auto",
               extent=[0, n_cols, cropped.shape[0], 0])
    for t in tick_positions:
        ax5.axvline(x=t, color="cyan", lw=0.8, alpha=0.85)
    ax5.set_xlim(0, n_cols)
    ax5.set_xlabel("X position (px)", fontsize=9)
    ax5.set_ylabel("Y (px)", fontsize=8)
    ax5.set_title(f"Detected ticks — {len(tick_positions)} total  |  "
                  f"{px_per_mm:.4f} px/mm  |  tick type: {detected_mm}mm",
                  fontsize=11, color=conf_col)
    ax5.tick_params(axis='both', labelsize=7)

    # ── Row 6: tick height classification ────────────────────────────
    ax6 = fig.add_subplot(gs[6], sharex=ax1)

    if height_profile is not None and len(height_profile) > 0:
        # Plot height profile as scatter, coloured by classification
        tick_x = np.array(tick_positions, dtype=float)
        colors = []
        for tc in tick_x:
            # find closest tick in ticks_1mm or ticks_05mm
            if ticks_05mm is not None and len(ticks_05mm) > 0:
                dist_05 = np.min(np.abs(ticks_05mm - tc))
            else:
                dist_05 = np.inf
            if ticks_1mm is not None and len(ticks_1mm) > 0:
                dist_1 = np.min(np.abs(ticks_1mm - tc))
            else:
                dist_1 = np.inf
            colors.append('limegreen' if dist_05 < dist_1 else 'tomato')

        ax6.scatter(tick_x, height_profile, c=colors, s=8, alpha=0.7, zorder=3)

        # Height threshold line
        height_threshold = float(np.median(height_profile))
        ax6.axhline(height_threshold, color='white', lw=1.2, linestyle='--',
                    alpha=0.7, label=f'Median threshold ({height_threshold:.0f}px)')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
                markersize=6, label=f'1mm ({len(ticks_1mm) if ticks_1mm is not None else 0} ticks)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                markersize=6, label=f'0.5mm ({len(ticks_05mm) if ticks_05mm is not None else 0} ticks)'),
            Line2D([0], [0], color='white', lw=1.2, linestyle='--',
                label=f'Median threshold ({height_threshold:.0f}px)'),
        ]
        ax6.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.6)
        ax6.set_ylabel("Tick height (px)", fontsize=8)
        ax6.set_title(
            f"Canny tick height classification  |  "
            f"detected_mm={detected_mm}mm  |  "
            f"px/mm={px_per_mm:.4f}",
            fontsize=11, color=conf_col)
    else:
        ax6.text(0.5, 0.5, 'Height profile not available',
                ha='center', va='center', transform=ax6.transAxes, fontsize=10)
        ax6.set_title("Canny tick height classification", fontsize=11)

    ax6.set_xlim(0, n_cols)
    ax6.set_xlabel("X position (px)", fontsize=9)
    ax6.grid(True, alpha=0.2)
    ax6.tick_params(axis='both', labelsize=7)

    fig.suptitle(f"Ruler Calibration Pipeline  ·  {avi_name}",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    avi_stem = Path(image_path).stem
    save_fig(fig, Path(output_dir), f"{avi_stem}_pipeline_summary.png")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(image_path: str, output_dir: str = "debug_output",
                 pad_fraction: float = 0.05, frame_index: int = 0):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f" Ruler Calibration Pipeline")
    print(f" Image  : {image_path}")
    print(f" Output : {out.resolve()}")
    print(f"{'='*60}\n")

    # 1. Load
    img = load_image(image_path, frame_index=frame_index)
    enhanced = preprocess(img)
    debug_load(img, enhanced, out)

    # ── Vertical ruler check ──────────────────────────────────────────
    VERTICAL_RULERS = {
        "cali_c1_03Nov2025",
        "cali_c1_31Mar2026",
        "cali_c1_31Oct2025",
        "cali_c2_04Oct2025",
        "cali_c2_12Nov2025",
        "cali_c2_31Oct2025",
        "cali_c3_27Sept2025",
        "cali_c4_27Sept2025",
    }
    is_vertical = Path(image_path).stem in VERTICAL_RULERS
    v_pad = 30 if is_vertical else 0

    rough_crop, _, _ = detect_ruler_roi(enhanced, pad_fraction,
                                        edge_exclude=10, output_dir=str(out),
                                        vertical_pad=v_pad)
    if is_vertical:
        rough_crop = cv2.rotate(rough_crop, cv2.ROTATE_90_CLOCKWISE)

    angle, angle_conf, edges, line_list = detect_angle(rough_crop)

    if is_vertical:
        # Get final ROI, rotate, THEN detect angle on the rotated crop
        cropped_raw, roi_coords, padded_mask_raw = detect_ruler_roi(
            enhanced, pad_fraction,
            edge_exclude=10,
            output_dir=str(out),
            vertical_pad=v_pad
        )
        cropped_rot     = cv2.rotate(cropped_raw,     cv2.ROTATE_90_CLOCKWISE)
        padded_mask_rot = cv2.rotate(padded_mask_raw, cv2.ROTATE_90_CLOCKWISE)

        # Detect angle on the rotated crop — ruler edges are now horizontal
        angle, angle_conf, edges, line_list = detect_angle(cropped_rot)
        print(f"  [orient] Angle on rotated crop = {angle:.2f}°")

        cropped     = deskew(cropped_rot,     angle)
        padded_mask = deskew(padded_mask_rot, angle)
        deskewed    = enhanced.copy()

        debug_angle(cropped_rot, angle, edges, line_list, cropped, out)
    else:
        deskewed = deskew(enhanced, angle)
        debug_angle(rough_crop, angle, edges, line_list, deskewed, out)

        cropped, roi_coords, padded_mask = detect_ruler_roi(
            deskewed, pad_fraction,
            edge_exclude=10, output_dir=str(out),
            vertical_pad=0
        )
        if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
                    raise ValueError(f"ROI crop is too small: {cropped.shape}")
    debug_roi(deskewed, roi_coords, cropped, padded_mask, out)

    # 4. Focus
    # scores, y_focus_start, y_focus_end, focus_score, strip_h = assess_focus(cropped)
    h_crop        = cropped.shape[0]
    y_focus_start = 0
    y_focus_end   = h_crop
    focus_score   = 1.0
    scores        = np.ones(40)
    strip_h       = max(h_crop // 40, 1)
    debug_focus(cropped, scores, y_focus_start, y_focus_end, strip_h, out)

    # 5. Tick detection
    ticks, profile, props, edge_mag, profile_residual, \
        major_peaks, minor_peaks, profile_merged, true_spacing, \
        profile_full, profile_short = \
        detect_ticks(cropped, y_focus_start, y_focus_end, padded_mask)
    
    debug_ticks(cropped, y_focus_start, y_focus_end, ticks, profile, edge_mag,
                profile_residual, major_peaks, minor_peaks, output_dir)

    if len(ticks) < 2:
        print("\n[ERROR] Fewer than 2 ticks detected — cannot compute px/mm.")
        print("  Suggestions:")
        print("  • Ensure the image contains a clear ruler with visible tick marks")
        print("  • Try increasing image contrast / resolution")
        sys.exit(1)

    # 6. Spacing & px/mm
    ticks_1mm, ticks_05mm, height_profile = classify_ticks_by_height(
    cropped, ticks, true_spacing)
    if len(ticks_05mm) > len(ticks_1mm) * 0.4:
        print(f"  [classify] Using 0.5mm ticks for px/mm computation")
        ticks_for_pxmm = ticks_05mm
        detected_mm    = 0.5
    else:
        ticks_for_pxmm = ticks_1mm
        detected_mm    = 1.0
    px_per_mm, px_mm_candidates, spacings_clean, tick_uniformity, _ = \
        compute_pixels_per_mm(ticks_for_pxmm, detected_mm=detected_mm)

    # 7. Confidence
    confidence, breakdown = compute_confidence(
        angle_conf, focus_score, tick_uniformity, len(ticks))
    # debug_confidence(breakdown, px_per_mm, out)

    # 8. Summary
    debug_summary(image_path, deskewed, cropped, padded_mask,
                  ticks, profile,
                  profile_full, profile_short,
                  y_focus_start, y_focus_end,
                  angle, px_per_mm, confidence, breakdown,
                  roi_coords, str(out), detected_mm=detected_mm,
                  ticks_1mm=ticks_1mm, ticks_05mm=ticks_05mm, height_profile=height_profile)

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" RESULTS  —  {Path(image_path).name}")
    print(f"{'='*60}")
    print(f"  Detected angle          : {angle:.3f}°")
    print(f"  Ticks detected          : {len(ticks)}")
    print(f"  Pixels per mm (1mm/div) : {px_per_mm:.4f} px/mm")
    print(f"  Pixels per mm (0.5mm)   : {px_mm_candidates.get(0.5, float('nan')):.4f} px/mm")
    
    print(f"  Detected tick type      : {detected_mm}mm")
    print(f"  Pixels per mm           : {px_per_mm:.4f} px/mm")

    print(f"\n  Confidence score        : {confidence:.3f}")
    print(f"{'='*60}\n")
    print(f"  Summary saved to: {out.resolve()}/pipeline_summary.png\n")

    return {
        "px_per_mm": px_per_mm,
        "px_mm_candidates": px_mm_candidates,
        "angle_deg": angle,
        "n_ticks": len(ticks),
        "confidence": confidence,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch pipeline — process all AVIs in a folder
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(calis_dir: str, output_dir: str, pad_fraction: float = 0.05):
    import csv
    calis_path = Path(calis_dir)
    avi_files  = sorted(calis_path.glob("*.avi"))

    if not avi_files:
        print(f"[batch] No AVI files found in {calis_dir}")
        return
    
    # Hardcoded vertical rulers — stems only, no extension
    VERTICAL_RULERS = {
        "cali_c1_03Nov2025",
        "cali_c1_31Mar2026",
        "cali_c1_31Oct2025",
        "cali_c2_04Oct2025",
        "cali_c2_12Nov2025",
        "cali_c2_31Oct2025",
        "cali_c3_27Sept2025",
        "cali_c4_27Sept2025"
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "calibration_results.csv"

    fieldnames = [
        "filename",
        "detected_angle_deg",
        "ruler_width_px",
        "ruler_height_px",
        "focus_score",
        "ticks_detected",
        "px_per_mm_1mm",
        "px_per_mm_0.5mm",
        "detected_mm",
        "confidence",
        "status",
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for avi in avi_files:
            print(f"\n{'='*60}")
            print(f" Processing: {avi.name}")
            print(f"{'='*60}")
            row = {f: "" for f in fieldnames}
            row["filename"] = avi.name

            try:
                img      = load_image(str(avi), frame_index=0)
                enhanced = preprocess(img)

                is_vertical = avi.stem in VERTICAL_RULERS
                v_pad = 30 if is_vertical else 0

                # First pass ROI
                rough_crop, _, _ = detect_ruler_roi(
                    enhanced, pad_fraction,
                    edge_exclude=10,
                    output_dir=str(out),
                    vertical_pad=v_pad
                )

                if is_vertical:
                    rough_crop = cv2.rotate(rough_crop, cv2.ROTATE_90_CLOCKWISE)
                    print(f"  [orient] Rough crop rotated 90° CW")

                # Detect angle on (possibly rotated) rough crop
                angle, angle_conf, edges, line_list = detect_angle(rough_crop)
                row["detected_angle_deg"] = round(angle, 3)

                if is_vertical:
                    # Get final ROI, rotate, THEN detect angle on the rotated crop
                    cropped_raw, roi_coords, padded_mask_raw = detect_ruler_roi(
                        enhanced, pad_fraction,
                        edge_exclude=10,
                        output_dir=str(out),
                        vertical_pad=v_pad
                    )
                    cropped_rot     = cv2.rotate(cropped_raw,     cv2.ROTATE_90_CLOCKWISE)
                    padded_mask_rot = cv2.rotate(padded_mask_raw, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(str(out / "debug_cropped_rot.png"), cropped_rot)

                    # Detect angle on the rotated crop — ruler edges are now horizontal
                    angle, angle_conf, edges, line_list = detect_angle(cropped_rot)
                    print(f"  [orient] Angle on rotated crop = {angle:.2f}°")

                    cropped     = deskew(cropped_rot,     angle)
                    padded_mask = deskew(padded_mask_rot, angle)
                    deskewed    = enhanced.copy()

                    debug_angle(cropped_rot, angle, edges, line_list, cropped, out)
                    print(f"  [orient] Rotated + deskewed crop: "
                          f"{cropped.shape[1]}×{cropped.shape[0]}")
                else:
                    # Normal horizontal ruler: deskew full image, then re-crop
                    deskewed = deskew(enhanced, angle)
                    debug_angle(rough_crop, angle, edges, line_list, deskewed, out)

                    cropped, roi_coords, padded_mask = detect_ruler_roi(
                        deskewed, pad_fraction,
                        edge_exclude=10,
                        output_dir=str(out),
                        vertical_pad=0
                    )
                    if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
                        raise ValueError(f"ROI crop is too small: {cropped.shape}")
                    
                row["ruler_width_px"]  = roi_coords[2]
                row["ruler_height_px"] = roi_coords[3]

                # scores, y_focus_start, y_focus_end, focus_score, strip_h = assess_focus(cropped)
                # # bypass assess_focus
                h_crop        = cropped.shape[0]
                y_focus_start = 0
                y_focus_end   = h_crop
                focus_score   = 1.0
                scores        = np.ones(40)
                strip_h       = max(h_crop // 40, 1)
                row["focus_score"] = round(focus_score, 3)
                
                ticks, profile, props, edge_mag, profile_residual, \
                    major_peaks, minor_peaks, profile_merged, true_spacing, \
                    profile_full, profile_short = \
                    detect_ticks(cropped, y_focus_start, y_focus_end, padded_mask)
                row["ticks_detected"] = len(ticks)

                if len(ticks) < 2:
                    row["status"] = "ERROR: fewer than 2 ticks"
                    writer.writerow(row)
                    print(f"  [batch] SKIPPED — fewer than 2 ticks")
                    continue

                ticks_1mm, ticks_05mm, height_profile = classify_ticks_by_height(
                    cropped, ticks, true_spacing)
                if len(ticks_05mm) > len(ticks_1mm) * 0.4:
                    print(f"  [classify] Using 0.5mm ticks for px/mm computation")
                    ticks_for_pxmm = ticks_05mm
                    detected_mm    = 0.5
                else:
                    ticks_for_pxmm = ticks_1mm
                    detected_mm    = 1.0
                px_per_mm, px_mm_candidates, spacings_clean, tick_uniformity, _ = \
                    compute_pixels_per_mm(ticks_for_pxmm, detected_mm=detected_mm)
                
                row["px_per_mm_1mm"]   = round(px_per_mm, 4)
                row["px_per_mm_0.5mm"] = round(
                    px_mm_candidates.get(0.5, float("nan")), 4)
                
                row["detected_mm"] = detected_mm

                confidence, breakdown = compute_confidence(
                    angle_conf, focus_score, tick_uniformity, len(ticks))
                row["confidence"] = round(confidence, 3)
                row["status"]     = "OK"

                debug_summary(
                    str(avi), deskewed, cropped, padded_mask,
                    ticks, profile,
                    profile_full, profile_short,
                    y_focus_start, y_focus_end,
                    angle, px_per_mm, confidence, breakdown,
                    roi_coords, str(out), detected_mm=detected_mm,
                    ticks_1mm=ticks_1mm, ticks_05mm=ticks_05mm, height_profile=height_profile
                )

                print(f"  [batch] {avi.name}: {px_per_mm:.4f} px/mm "
                      f"({len(ticks)} ticks, conf={confidence:.2f})")

            except Exception as e:
                row["status"] = f"ERROR: {e}"
                print(f"  [batch] ERROR on {avi.name}: {e}")

            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f" Batch complete — {len(avi_files)} files processed")
    print(f" Results saved to: {csv_path}")
    print(f"{'='*60}\n")

# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
 
#     # IMAGE_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_02Nov2025.avi" # 10.0000 px/mm
#     IMAGE_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_15Oct2025.avi" # 6.6552 px/mm
#     # IMAGE_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_12Oct2025.avi"
#     # IMAGE_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_20Sept2025.avi"
#     # IMAGE_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_01Oct2025.avi"
#     # IMAGE_PATH = "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/cali_c1_22Sept2025.avi"

#     OUTPUT_DIR = "/Users/sophiehanson/Desktop/automate_cali_digitization/debug_out/"
#     FRAME      = 0     # single-frame AVI → always 0
#     PAD        = 0.05  # inward pad fraction — usually leave as-is
 
#     result = run_pipeline(IMAGE_PATH, output_dir=OUTPUT_DIR,
#                           pad_fraction=PAD, frame_index=FRAME)
    
if __name__ == "__main__":

    CALIS_DIR  = "/Users/sophiehanson/Desktop/automate_cali_digitization/calis/"
    OUTPUT_DIR = "/Users/sophiehanson/Desktop/automate_cali_digitization/image_check_output/"
    PAD        = 0.05  # inward pad fraction

    run_batch(CALIS_DIR, OUTPUT_DIR, pad_fraction=PAD)
