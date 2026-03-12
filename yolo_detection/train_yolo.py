from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=r"D:\Spring26\TA\robot_vision\yolo_detection\dataset\data.yaml", epochs=10, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model(r"D:\Spring26\TA\robot_vision\yolo_detection\dataset\images\val\0_0_png_camera_noise_photo1.jpg")