from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='fruit.yaml', epochs=100)

model.val()