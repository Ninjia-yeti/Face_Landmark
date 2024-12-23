from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
model.train(data='data.yaml', epochs=150, imgsz=640)
