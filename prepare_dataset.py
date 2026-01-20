import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
BASE_DIR = r"D:\Spring26\TA\robot_vision\classification_dataset"

IMG_SRC_DIR = os.path.join(BASE_DIR, "augmented_images_pip")
CSV_SRC = os.path.join(BASE_DIR, "augmented_labels.csv")

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, "images")
VAL_IMG_DIR   = os.path.join(VAL_DIR, "images")
TEST_IMG_DIR = os.path.join(TEST_DIR, "images")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

# ---- Load labels ----
df = pd.read_csv(CSV_SRC)

# Extract base id (e.g. "3_2" from "3_2_aug14.png")
df["base_id"] = df["image_name"].apply(lambda x: x.split("_aug")[0])

unique_ids = df["base_id"].unique()
train_val_ids, test_ids = train_test_split(unique_ids, test_size=0.4, random_state=42)

train_ids, val_ids = train_test_split(
    unique_ids, test_size=0.2, random_state=42
)

train_df = df[df["base_id"].isin(train_ids)]
val_df   = df[df["base_id"].isin(val_ids)]
test_df = df[df["base_id"].isin(test_ids)]
# ---- Copy images ----
def copy_images(sub_df, dst_img_dir):
    for img_name in tqdm(sub_df["image_name"].unique()):
        src = os.path.join(IMG_SRC_DIR, img_name)
        dst = os.path.join(dst_img_dir, img_name)

        if not os.path.exists(src):
            print(f"Missing image: {src}")
            continue

        shutil.copy2(src, dst)

print("Copying train images...")
copy_images(train_df, TRAIN_IMG_DIR)

print("Copying val images...")
copy_images(val_df, VAL_IMG_DIR)

print("Copying test images...")
copy_images(test_df, TEST_IMG_DIR)

# ---- Save CSVs (drop helper column) ----
train_df.drop(columns=["base_id"]).to_csv(
    os.path.join(TRAIN_DIR, "labels.csv"),
    index=False
)

val_df.drop(columns=["base_id"]).to_csv(
    os.path.join(VAL_DIR, "labels.csv"),
    index=False
)
test_df.drop(columns=["base_id"]).to_csv(
    os.path.join(TEST_DIR, "labels.csv"),
    index=False
)

print("dataset prepared successfully.")

