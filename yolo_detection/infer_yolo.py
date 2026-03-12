from ultralytics import YOLO

# Load your custom trained model
model = YOLO(r"yolo_detection\runs_50epochs\obb\train\weights\best.pt")

# Run inference on a directory of images or a single file
results = model(source= r"yolo_detection\dataset\images\val\0_0_png_camera_noise_photo1.jpg", conf=0.5) 
# confidence threshold is set to 0.5, you can adjust it as needed, or remove it for all the results 
