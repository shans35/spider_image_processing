# RUN CODE in bash/terminal
# python train_yolo.py

from ultralytics import YOLO

# yolov8n.pt = default model name (rectangle). for obb, use 'yolov8n-obb.pt'.
model = YOLO("yolov8n.pt")

model.train(
    data="/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/yolo_object_detection/yolo_dataset/spider.yaml",
    box=10.0,
    epochs=200,
    imgsz=960,
    batch=4,
    # patience=30,
    augment=True,
    project="/media/peterparker/SOPHIEF/jumping_spider/spider_image_processing/spider_centroid/yolo_object_detection/yolo_runs",
    name="spider_detector_bb_v5",
)

# for 30 labelled avi videos and 100 epochs, it took 10 minutes
# for 80 labelled avi videos and 100 epochs, it took 14 minutes


# v5: for 80 labelled avi videos and 200 epochs, it took x minutes 3:16 - 3:47
#    > 100 -> 200 epochs, box 7.5 -> 10.0, imgsz 640 -> 960 (attempt to increase box edge accuracy)
#    > training stopped early as no improvement observed after epoch 40
#    > 70 epochs completed