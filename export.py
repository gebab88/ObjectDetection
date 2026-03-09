from ultralytics import YOLO

model = YOLO("yolo26m.onnx")

# Einfach ein Testbild nehmen – dauert 2-3 Sekunden
results = model.predict("https://ultralytics.com/images/bus.jpg", conf=0.25)

for r in results:
    for box in r.boxes:
        print(f"{r.names[int(box.cls)]} – {box.conf.item():.2f}")