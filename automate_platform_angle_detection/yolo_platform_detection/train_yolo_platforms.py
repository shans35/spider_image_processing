# RUN CODE in bash/terminal
# python train_yolo.py

from ultralytics import YOLO

# yolov8n-obb = nano oriented bounding box model
model = YOLO("yolov8n-obb.pt")

model.train(
    data="/media/peterparker/9BFA-B40E/jumping_spider/image_processing_git/spider_image_processing/image_processing/yolo_platform_detection/yolo_dataset/platforms.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    patience=30,
    augment=True,
    project="/media/peterparker/9BFA-B40E/jumping_spider/image_processing_git/spider_image_processing/image_processing/yolo_platform_detection/yolo_runs",
    name="platform_detector_obb_v1",
)

# for 25 labelled avi videos and 100 epochs, it took 16.5 minutes
