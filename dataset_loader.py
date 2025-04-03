import os
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Image Transformations (Resizing + Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ResNet normalization
])

class FewShotDataset(Dataset):
    def __init__(self, root_dir, split="support"):
        self.root_dir = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(self.root_dir))  # Ensure consistent class ordering
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}  # Class to index mapping
        
        self.data = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            images = os.listdir(class_path)
            self.data.extend([(cls, img) for img in images])
        
        print(f"Class-to-Index Mapping: {self.class_to_idx}")  # Debugging statement
        
    def __getitem__(self, index):
        class_name, img_name = self.data[index]
        class_idx = self.class_to_idx[class_name]
        img_path = os.path.join(self.root_dir, class_name, img_name)
        
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        
        return image, class_idx
    
    def __len__(self):
        return len(self.data)

# Define dataset paths
data_path = r"D:\Programming\GIthub projects\Few-shot learning\Few-shot-learning\new_dataset\structured_train"

# Create DataLoaders
support_dataset = FewShotDataset(data_path, split="support")
query_dataset = FewShotDataset(data_path, split="query")

support_loader = DataLoader(support_dataset, batch_size=5, shuffle=True)
query_loader = DataLoader(query_dataset, batch_size=5, shuffle=True)

# Test DataLoader
for support_images, support_labels in support_loader:
    print(f"Support Batch - Images: {support_images.shape} Labels: {support_labels}")
    break

for query_images, query_labels in query_loader:
    print(f"Query Batch - Images: {query_images.shape} Labels: {query_labels}")
    break
