from PIL import Image
import os
import numpy as np

img_dir = "classification_dataset/train/images"

ratios = []

for fname in os.listdir(img_dir)[:100]:  # sample 100 images
    img = Image.open(os.path.join(img_dir, fname))
    w, h = img.size
    ratios.append(w / h)

print(f"Mean AR: {np.mean(ratios):.3f}")
print(f"Std  AR: {np.std(ratios):.3f}")
