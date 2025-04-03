import os
from PIL import Image

# Set paths to dataset folders
dataset_folders = [r"D:\Programming\GIthub projects\Few-shot learning\Few-shot-learning\dataset\train\not_porsche",
                   r"D:\Programming\GIthub projects\Few-shot learning\Few-shot-learning\dataset\train\porsche",
                   r"D:\Programming\GIthub projects\Few-shot learning\Few-shot-learning\dataset\val\porsche",
                   r"D:\Programming\GIthub projects\Few-shot learning\Few-shot-learning\dataset\val\not_porsche"]
output_size = (224, 224)  # Resize to 224x224

def resize_images(folder):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.resize(output_size)
                img.save(img_path)  # Overwrite with resized image
        except Exception as e:
            print(f"Skipping {filename}: {e}")

# Resize images in all dataset folders
for folder in dataset_folders:
    resize_images(folder)

print("✅ All images resized successfully!")
