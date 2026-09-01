# yeah 

"""
- use spider_calc_central.py on each avi in a folder to compile data 
into an AviInstance class, producing a new csv for each one
> each row is a frame
"""

import pathlib as Path
from pathlib import Path
import sys
import argparse

HOME_DIR = Path("/media/peterparker/SOPHIEF/jumping_spider/high_speed_videos/phidippus_JanFeb2026")
avis = HOME_DIR.iterdir()


for avi in avis:
    filename = avi.name
