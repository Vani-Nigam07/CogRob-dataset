import os
import cv2
import pandas as pd
from tqdm import tqdm
import albumentations as A

appearance_aug = A.Compose([
    # ---- Background & lighting ----
    A.RandomBrightnessContrast(
        brightness_limit=0.25,
        contrast_limit=0.25,
        p=0.8
    ),

    A.HueSaturationValue(
        hue_shift_limit=5,        # slight tint
        sat_shift_limit=10,
        val_shift_limit=10,
        p=0.5
    ),

    A.RandomGamma(
        gamma_limit=(80, 120),
        p=0.5
    ),

    # ---- Camera effects ----
    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.3
    ),

    A.GaussNoise(
        std_range=(0.02, 0.06),
        p=0.4
    ),

    # # ---- Subtle vignette ----
    # A.Vignette(
    #     intensity=(0.2, 0.4),
    #     p=0.3
    # ),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)
])


INPUT_IMG_DIR = r"D:\Spring26\TA\robot_vision\raw_data\augmented_images\images"
INPUT_CSV = r"D:\Spring26\TA\robot_vision\raw_data\augmented_images\labels_images.csv"

OUT_IMG_DIR = r"D:\Spring26\TA\robot_vision\classification_dataset\augmented_images_pip"
OUT_CSV = r"D:\Spring26\TA\robot_vision\classification_dataset\augmented_labels.csv"

AUGS_PER_IMAGE = 50   # ← increase later (100–500)

os.makedirs(OUT_IMG_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

aug_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row["image_name"]
    L = int(row["left_pips"])
    R = int(row["right_pips"])

    img_path = os.path.join(INPUT_IMG_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    for i in range(AUGS_PER_IMAGE):
        augmented = appearance_aug(image=img)
        aug_img = augmented["image"]

        new_name = f"{img_name.replace('.png','')}_aug{i}.png"
        cv2.imwrite(os.path.join(OUT_IMG_DIR, new_name), aug_img)

        aug_rows.append({
            "image_name": new_name,
            "left_pips": L,
            "right_pips": R
        })

# Save new CSV
aug_df = pd.DataFrame(aug_rows)
aug_df.to_csv(OUT_CSV, index=False)

print("Appearance augmentation complete.")
