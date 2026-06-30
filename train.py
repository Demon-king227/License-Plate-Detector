from ultralytics import YOLO
from config import MODEL_VEHICLE

def train_model():
    model = YOLO(MODEL_VEHICLE)
    results = model.train(data='data.yaml', epochs=50, imgsz=640)

if __name__ == '__main__':
    train_model()