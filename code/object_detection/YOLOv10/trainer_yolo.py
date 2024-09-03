from ultralytics import YOLOv10

model_name = "./pretrained_base_models/yolov10s.pt"
dataset = "./dataset.yaml"
val_dataset = "./val.yaml"

model = YOLOv10(model_name)
model.train(data=dataset, batch=-1, imgsz=640, epochs=300, verbose=True)

model.val(data=val_dataset, verbose=True)
