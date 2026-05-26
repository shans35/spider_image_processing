# RUN CODE in bash/terminal
# python train_yolo.py

from ultralytics import YOLO

# yolov8n-obb = nano oriented bounding box model
model = YOLO("yolov8n-obb.pt")

model.train(
    data="/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/yolo_dataset/ruler.yaml",
    epochs=100,
    imgsz=640,
    batch=4,
    patience=30,
    augment=True,
    project="/Users/sophiehanson/Desktop/automate_cali_digitization/yolo_object_detection/yolo_runs",
    name="ruler_detector_obb_v1",
)

# for 132 labelled avi videos and 200 epochs, it took 44 minutes
# for 132 labelled avi videos and 100 epochs, it took X minutes
