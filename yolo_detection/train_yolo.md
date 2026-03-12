## Command Line Interface (CLI) for quick execution and a Python API for Ultralytics implementation of YOLOv11
## for other versions of the model check https://docs.ultralytics.com/models

create a folder as yolo_detection 
```
mkdir yolo_project && cd yolo_project
```
after activating you venv, install the ultralytics library 
```
pip install ultralytics
```

make a new python file name train_yolo.py or run the following command in the bash cli.
you dont need to download the model needed, mentioning the correct name in the command helps
ultralytics library to do it for you automatically

bash
```
yolo train model=yolo11n.pt data=path_to_data_yaml.yaml epochs=100 imgsz=640
```

python code as in the yolo_detection\train_yolo.py
```
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the example dataset for 100 epochs
results = model.train(data=r"D:\Spring26\TA\robot_vision\yolo_detection\dataset\data.yaml", epochs=10, imgsz=640)
```



## Command Breakdown
yolo train model=yolo11n.pt data=path_to_data_yaml.yaml epochs=100 imgsz=640

yolo: The main entry point for the Ultralytics library.

train: The mode of operation (other modes include predict, val, and export).

model=yolo11n.pt: Specifies the model architecture. Using a .pt file downloads pretrained weights, which speeds up convergence (Transfer Learning).

data=path_to_data.yaml: The path to your configuration file that defines where your images are and what the class names are.

epochs=100: The number of times the model will see the entire dataset.

imgsz=640: The resolution the images are resized to before entering the network.


## how to use the resultant trained model 

When you run a YOLO training command, the library creates a structured output directory (usually named runs/model/train/) to store everything from performance logs to the final model weights.

you can see the already trained model results in https://github.com/Vani-Nigam07/CogRob-dataset/tree/main/yolo_detection/runs_50epochs/obb/train 

## inference
you have to use 'yolo_detection\runs_50epochs\obb\train\weights\best.pt' for your trained model's results on a new image(data point!)

bash
```
from ultralytics import YOLO

# Load your custom trained model
model = YOLO(r"yolo_detection\runs_50epochs\obb\train\weights\best.pt")

# Run inference on a directory of images or a single file
results = model(source= r"yolo_detection\dataset\images\val\0_0_png_camera_noise_photo1.jpg", conf=0.5) 


``` 



