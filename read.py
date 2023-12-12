from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(show= True, source="0") 