import os

MODEL_VEHICLE = 'yolov8n.pt'
MODEL_PLATE = 'license_plate_detection.pt' if os.path.exists('license_plate_detection.pt') else 'license_plate_detector.pt'
CONFIDENCE_THRESHOLD = 0.15