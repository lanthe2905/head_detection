from ultralytics import YOLO

model = YOLO("best.pt")
results = model.track(show= True, source="0") 