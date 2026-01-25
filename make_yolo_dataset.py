import os
import random
import shutil

# ===== CONFIG =====
SRC_IMAGES = r"D:\Spring26\TA\robot_vision\raw_data\augmented_images\output\images"
SRC_LABELS = r"D:\Spring26\TA\robot_vision\raw_data\augmented_images\output\labels"

DST_ROOT = r"D:\Spring26\TA\robot_vision\yolo_detection\dataset"

TRAIN_RATIO = 0.8
SEED = 42


random.seed(SEED)

images = [f for f in os.listdir(SRC_IMAGES) if f.endswith(".jpg")]
random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def copy_split(img_list, split):
    os.makedirs(os.path.join(DST_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, "labels", split), exist_ok=True)
    for img in img_list:
        lbl = img.replace(".jpg", ".txt")

        shutil.copy(
            os.path.join(SRC_IMAGES, img),
            os.path.join(DST_ROOT, "images", split, img)
        )
        shutil.copy(
            os.path.join(SRC_LABELS, lbl),
            os.path.join(DST_ROOT, "labels", split, lbl)
        )

copy_split(train_imgs, "train")
copy_split(val_imgs, "val")

print(f"Train: {len(train_imgs)} images")
print(f"Val: {len(val_imgs)} images")
