import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import numpy as np
from ultralytics import YOLO
from config import MODEL_VEHICLE, MODEL_PLATE, CONFIDENCE_THRESHOLD
from license_plate_detection import LicensePlateDetectorOCR

class VehicleEngine:
    def __init__(self):
        self.vehicle_model = YOLO(MODEL_VEHICLE)
        self.plate_ocr = LicensePlateDetectorOCR(MODEL_PLATE)

    def detect(self, frame):
        if frame is None:
            return [], []
        # Fast single-pass detection across all vehicles and plates on the frame
        v_results = self.vehicle_model(frame, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False)
        p_results = self.plate_ocr.detector(frame, conf=CONFIDENCE_THRESHOLD, iou=0.45, verbose=False)
        return v_results, p_results

    def get_color(self, crop):
        if crop is None or crop.size == 0:
            return "Unknown"
        h, w = crop.shape[:2]
        body = crop[int(h * 0.35) : int(h * 0.75), int(w * 0.15) : int(w * 0.85)]
        if body.size == 0:
            body = crop

        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        valid = hsv[(hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 245)]
        if len(valid) == 0:
            valid = hsv.reshape(-1, 3)

        s_med = np.median(valid[:, 1])
        v_med = np.median(valid[:, 2])
        h_med = np.median(valid[:, 0])

        is_blue_reflection = (85 <= h_med <= 128) and (s_med < 115)
        
        if s_med < 55 or is_blue_reflection:
            if v_med < 70:
                return "Black"
            if v_med > 165:
                return "White"
            return "Silver/Gray"

        if h_med < 10 or h_med > 170:
            return "Red"
        elif 10 <= h_med < 25:
            return "Orange"
        elif 25 <= h_med < 35:
            return "Yellow"
        elif 35 <= h_med < 85:
            return "Green"
        elif 85 <= h_med < 130:
            return "Blue"
        elif 130 <= h_med <= 170:
            return "Purple"
        return "Silver/Gray"

    def read_plate(self, crop):
        text, conf = self.plate_ocr.recognize_plate(crop)
        return text

VehichleVE = VehicleEngine
