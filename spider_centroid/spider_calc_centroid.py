# August 18, 2026
# Ana Curtis

# INTERPRETER VER: 3.12.13
# ENV: env_swag

"""
Q: What data structures do we need to use?

What this script does: 

input: collection of .avi's
output: collection of .csv's

- take in folder of avis and iterate through it
- for every avi in folder: 
    - create csv
    - load in avi
        for every frame:
            - create an instance of AviInstance class to track centroid
            - Use YOLO to place a bounding box around the spider 
            - Crop frame to the ROI
            - Process the image (Grayscale, Gaussian Blur, CannyEdges, Threshold)
            - mask out spider (how)
            - detect centroid, update class instance 
            - use to_csv class method to add another row to csv 
    - return csv

"""

import cv2
from pathlib import Path
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import numpy as np



class AviInstance:
    def __init__ (self, filename, centroid):
        animal_id, jump_number, cali_number, date = parse_filename(filename)

        self.animal_id = animal_id
        self.jump_number = jump_number
        self.cali_number = cali_number
        self.date = date

        self.centroid = centroid
    
    def to_csv():
        # labels stay the same for each csv, only centroid is variable


def parse_filename(filename):
    # must be reduced to filename from path
    # filename contains 3 underscores separating animal id, jump number, cali_number, and the date.

    filename_separated = filename.split('_')

    animal_id = filename_separated[0]
    jump_number = filename_separated[1]
    cali_number = filename_separated[2]
    date = filename_separated[3]

    return animal_id, jump_number, cali_number, date
    

def main(filename: str):
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"No such file: {filepath}")
    
    return

if __name__ == "__main__":
    args = parse_args()
    main()
