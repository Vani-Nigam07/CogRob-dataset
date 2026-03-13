from ultralytics import YOLO
import cv2

# Load model
model = YOLO(r"yolo_detection\runs_50epochs\obb\train\weights\best.pt")

# Run inference
results = model(
    source=r"yolo_detection\dataset\images\val\0_0_png_camera_noise_photo1.jpg",
    
)
print(results[0].obb)

# Get plotted image
annotated_img = results[0].plot()

# Show image
cv2.imshow("Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()