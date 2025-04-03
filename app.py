import os
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision import transforms, models
import torch.nn as nn

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load trained model
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProtoNet().to(device)
model.load_state_dict(torch.load("few_shot_model.pth", map_location=device))
model.eval()

# Placeholder prototype (replace with actual computed prototype)
porsche_prototype = torch.randn(128).to(device)  # Ensuring correct size

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image).squeeze(0)
    
    similarity = F.cosine_similarity(embedding, porsche_prototype.unsqueeze(0))
    
    if similarity.item() > 0.5:
        return "Porsche Detected"
    else:
        return "Not a Porsche"

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    result = classify_image(file_path)
    
    img = Image.open(file_path)
    img = img.resize((300, 300))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img
    
    result_label.config(text=result)

# GUI Setup
root = tk.Tk()
root.title("Few-Shot Learning Classifier")

canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()

root.mainloop()