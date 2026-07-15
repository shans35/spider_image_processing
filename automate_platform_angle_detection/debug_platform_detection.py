import cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import numpy as np

HOME_DIR = "/media/peterparker/9BFA-B40E'/jumping_spider/spider_image_processing"

image_path = "/media/peterparker/9BFA-B40E/jumping_spider/spider_image_processing/automate_platform_angle_detection/input/tw-00811-01_19_c2_06Apr2026.avi"
output_dir = "/media/peterparker/'9BFA-B40E'/jumping_spider/spider_image_processing/automate_platform_angle_detection/output"

cap = cv2.VideoCapture(image_path)
print(cap.read())