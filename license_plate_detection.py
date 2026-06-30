import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from config import MODEL_PLATE, CONFIDENCE_THRESHOLD

class LicensePlateDetectorOCR:
    """
    High-speed License Plate Detection & OCR module optimized for real-time multi-car CPU performance.
    """
    def __init__(self, model_path=MODEL_PLATE, gpu=False):
        self.detector = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=gpu)

    def preprocess_fast(self, crop):
        if crop is None or crop.size == 0:
            return None
        
        h, w = crop.shape[:2]
        if h < 12 or w < 30:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = max(2.5, 70.0 / max(1, h))
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        pad = cv2.copyMakeBorder(upscaled, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

        # Single high-contrast CLAHE pass for ultra-fast CPU recognition
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(pad)
        return clahe

    def correct_ocr_lookalikes(self, text):
        chars = list(text)
        n = len(chars)
        for i, c in enumerate(chars):
            if i > 1 and chars[i-1].isdigit():
                if c == 'O': chars[i] = '0'
                elif c == 'I': chars[i] = '1'
                elif c == 'Z': chars[i] = '2'
                elif c == 'S': chars[i] = '5'
                elif c == 'B' and i == n - 1: chars[i] = '8'
        return "".join(chars)

    def recognize_plate(self, crop):
        view = self.preprocess_fast(crop)
        if view is None:
            return "Unknown", 0.0

        results = self.reader.readtext(
            view, 
            detail=1, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        
        best_text = "Unknown"
        best_conf = 0.0
        for bbox, text, prob in results:
            clean = "".join(c for c in text if c.isalnum()).strip().upper()
            clean = self.correct_ocr_lookalikes(clean)
            if len(clean) >= 5 and prob > best_conf:
                best_text = clean
                best_conf = prob

        return (best_text, best_conf) if (best_conf >= 0.18 and len(best_text) >= 5) else ("Unknown", 0.0)

    def evaluate_frame(self, frame):
        results = self.detector(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                text, conf = self.recognize_plate(crop)
                if text != "Unknown":
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'plate': text,
                        'ocr_conf': conf,
                        'det_conf': float(box.conf[0])
                    })
        return detections
