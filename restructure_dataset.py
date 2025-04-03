import os
import shutil
import random

# Set paths
dataset_path = "dataset/train"
output_path = "dataset/structured_train"
os.makedirs(output_path, exist_ok=True)

# Parameters
num_support = 5   # Support images per class
num_query = 3     # Query images per class

# Process each class folder
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    
    if os.path.isdir(class_path):  # Ensure it's a folder
        images = [img for img in os.listdir(class_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)  # Shuffle images
        
        # Create support and query directories
        support_dir = os.path.join(output_path, "support", class_name)
        query_dir = os.path.join(output_path, "query", class_name)
        os.makedirs(support_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)
        
        # Move images
        for i, img in enumerate(images):
            src = os.path.join(class_path, img)
            if i < num_support:
                shutil.move(src, os.path.join(support_dir, img))
            elif i < num_support + num_query:
                shutil.move(src, os.path.join(query_dir, img))

print("Dataset restructuring complete!")
