from ultralytics import YOLOv10

model_name = "/mnt/beegfs/home/s425500/Thesis/Yolo/yolov10s.pt"
dataset = "/mnt/beegfs/home/s425500/Thesis/Yolo/dataset2.yaml"
val_dataset = "/mnt/beegfs/home/s425500/Thesis/Yolo/val2.yaml"

model = YOLOv10(model_name)
model.train(data=dataset, batch=-1, imgsz=640, epochs=300, verbose=True)

model.val(data=val_dataset, verbose=True)