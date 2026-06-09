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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Objective: extract a frame from one of the avi's and identify the landing platform. 

#---- Helpers ----#

# Platform image grayitization
def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#---- LOAD ATTEMPT ----#

# def load_image(image_path: str, frame_index: int = 0):

image_path = "/media/peterparker/9BFA-B40E/jumping_spider/image-processing/input/sh154_14_c1_16Mar2026.avi"
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

if __name__ == '__main__':
    INPUT_DIR = Path("/media/peterparker/'9BFA-B40E'/jumping_spider/image-processing/input")
    OUTPUT_DIR = Path("/media/peterparker/'9BFA-B40E'/jumping_spider/image-processing/output")

#---- ACTIONABLE CODE ----#

img = load_image(image_path)
#save_fig(img, output_dir, "your_mom.png")
plt.imshow(img)
plt.axis('off')
plt.show()


    