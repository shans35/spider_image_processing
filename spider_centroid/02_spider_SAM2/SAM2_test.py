# 11/3/26 - Messing with SAM

# ran UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=1 uv pip install torch torchvision --index https://download.pytorch.org/whl/cu126 --reinstall

# TRANSPLANT YOLO STUFF

# python test_yolo.py
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
matplotlib.use("Agg")
from typing import NamedTuple
from pathlib import Path
import numpy as np
import random
import cv2

# invoke yolo on one frame from every video
# display raw image, yolo applied, and sam applied using matplotlib
# save into debug directory

MODEL_PATH = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/02_spider_SAM2/best.pt")
AVI_DIR = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/input/yolo_avi_collection_spiders/additional_input")
OUTPUT_DIR = Path("/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/test_material/SAM2_test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        
        # assumes all images may accomodate 256 x 1024 dimensions
        
        original_height, original_width = img.shape[:2]

        start_x, end_x = 0, original_width
        start_y, end_y = original_height - 256, original_height # complicated, get total y and do total y-256 -> total y
        
        img = img[start_y:end_y, start_x:end_x]
        

        return img

def invoke_yolo(avi_path: str)-> YoloResult:
    """Runs YOLO inference on a single frame path and returns the YoloResult class containing bbox, img, and cropped image."""

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

# THIS CODE CREATES MODEL VARIABLE 

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

sam2_checkpoint = "/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/02_spider_SAM2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=dev)

predictor = SAM2ImagePredictor(sam2_model)

# Segmentation using bounding boxes

def box_segmentor(image, box_coords, model_checkpoint=sam2_checkpoint, device='auto'):
    """
    Perform SAM2 segmentation using multiple box coordinates with optimized processing.
    
    Args:
        img (obj): raw image screencap
        box_coords (list): List of [x_min, y_min, x_max, y_max] coordinates
        model_checkpoint (str): Optional path to SAM2 checkpoint
        device (str): 'cuda', 'cpu' or 'auto' (default) (should be cuda for us)
    
    Returns: 
        tuple: (masks_list, scores_list) containing segmentation results
    """

    # Device configuration
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model (if not already loaded)
    if not hasattr(box_segmentor, 'predictor'):
        checkpoint = model_checkpoint or "../checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        box_segmentor.predictor = SAM2ImagePredictor(sam2_model)
    
    # Load and preprocess image
    #image = np.array(img.convert("RGB"))
    
    # Set image once for all predictions (optimization)
    device = torch.device(device)   # accepts "cpu", "cuda", or an existing torch.device
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
        box_segmentor.predictor.set_image(image)
        
        masks_list = []
        scores_list = []
        
        for box in box_coords:
            box_array = np.array(box)[None, :]  # Add batch dimension
            
            # Correct unpacking: predict returns (masks, scores, logits)
            predicted_masks, predicted_scores, _ = box_segmentor.predictor.predict(
                box=box_array,
                multimask_output=False
            )
            
            masks_list.append(predicted_masks[0])
            scores_list.append(predicted_scores[0])
    
    return masks_list, scores_list


# Visualize result! 

def display_mask(img, mask, alpha=0.5, random_color=True, borders=True, name='blank'):
    """
    Display multiple masks over an image with different colors

    Args:
        image: PIL.Image - Original image
        masks: List of 2D numpy arrays (float32 masks 0-1)
        alpha: float (0-1) - Transparency of mask fills
        random_color: bool - Use random colors for each mask
        borders: bool - Show border contours
        name: name of avi
    """

    img_np = img
    h, w = img_np.shape[:2]
    print(f"height of img is: {h}. width of img is: {w}")
    
    # Create overlay canvas
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    
    
    # Generate colors for each mask
    colors = []
    for _ in mask:
        if random_color:
            colors.append(np.append(np.random.random(3), alpha))
        else:
            colors.append(np.array([30/255, 144/255, 255/255, alpha]))
    
    # Process each mask
    for i, mask in enumerate(mask):
        # Convert mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Create mask overlay
        mask_overlay = np.zeros((h, w, 4))
        print(f"h: {h}. w: {w}")
        
        mask_overlay[binary_mask == 1] = colors[i]
        print("cuold maskoverlay2 be the rpoeblem")
        
        # Add borders
        if borders:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contours = [cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True) for c in contours]
            
            # Draw white borders (80% opacity)
            border_color = (1, 1, 1, 0.8)  # White borders
            cv2.drawContours(mask_overlay, contours, -1, border_color, 2)
            
        
        # Combine overlays
        overlay = np.where(mask_overlay != 0, mask_overlay, overlay)
    
    # Create final composition
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    ax.imshow(overlay)
    ax.axis('off')

    # save fig to output dir
    fig.savefig(OUTPUT_DIR / f"{name}_debug.jpg", bbox_inches="tight")
    plt.close(fig)

def get_mask_centroid_area(mask: np.ndarray):
    # Ensure mask is 2D boolean (SAM predictor outputs shape (N, H, W) or
    if mask.ndim == 3:
        mask = mask[0] # take first mask if batched

    # Find row (y) and column (x) indeces where mask is True
    y_indices, x_indices = np.where(mask > 0)

    # Guard against empty masks
    if len(x_indices) == 0:
        return None
    
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)

    return (float(centroid_x), float(centroid_y))

def get_mask_centroid_moments(mask: np.ndarray):
    # 1. Ensure your binary mask is 2D and uint8 (works for 0/1, 0/255, or boolean arrays)
    binary_mask = mask.astype(np.uint8)
    print("mask success")
    # filtered_data = arr[binary_mask[:256]]

    # 2. Calculate moments with binaryImage=True
    moments = cv2.moments(binary_mask, binaryImage=True)
    print("your mom-ents")

    # 3. Extract area and centroid (x, y)
    area = moments["m00"]
    if area > 0:
        cx = moments["m10"] / area
        cy = moments["m01"] / area
        print(f"Area: {area}, Centroid: ({cx:.2f}, {cy:.2f})")
        return cx, cy
    else:
        print("Mask is completely empty.")
        return 0, 0

def plot_centroid(image, moments, cx, cy):
    # the image is the entire raw image
    print("image length: " + str(image.shape[0]))
    if moments: 
        color = "red"
    else:
        color = "blue"

    fig, ax = plt.subplots()
    ax.imshow(image)  # image is already an array, no imread needed
    ax.scatter(cx, cy, c=color, marker='x')
    return fig

def create_debug_image():

    return
    



# run yolo to prepare boxes 

def pseudo_main():
    """
    Main pipeline function, goes thru avi_dir and extracts bboxes as inputs to SAM2 functions, uses SAM2 functions to save visualizations into output_dir. 

    """

    sorted_avis = sorted(AVI_DIR.glob("*.avi"))
    errors = []
    for avi in sorted_avis:
        if avi.suffix.lower() not in VIDEO_EXTS:
            continue
        else:
            res = invoke_yolo(str(avi))

            # perform segmentation using this! 

            image_path = "george.jpg"
            image = res.img
            name = avi.stem
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_boxes = res.box
            masks, scores = box_segmentor(
                image= img,
                box_coords= [input_boxes]
            )
            print("display mask is culprit")
            display_mask(image, masks, alpha=0.5, random_color=True, borders=False, name=name)
            print("display mask is not culprit")
            print(f"img h, w: {img.shape}, mask dimensions: {[m.shape for m in masks][:3]}")
            for mask in masks:
                cx, cy = get_mask_centroid_moments(mask) 
                fig = plot_centroid(img, 1, cx, cy)
                fig.savefig(f"/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/test_material/SAM2_test_results/sam_debugs/{name}.jpg")

        """
        except Exception as e: 
            errors.append({
                "video": str(avi.stem),
                "error": str(e)
            })
    #        generate_empty_debug(avi.stem, str(e))
            continue
        """
        

    # report at the end
    if errors:
        print(f"\n{len(errors)} of {len(sorted_avis)} videos failed:\n")
        for item in errors:
            print(f"- {item['video']}: {item['error']!r}")
            # print(item["traceback"])  # uncomment for full details

    return errors

pseudo_main()




