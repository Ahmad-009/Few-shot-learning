import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset Class
class FewShotDataset(Dataset):
    def __init__(self, root_dir, split="support"):
        self.root_dir = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.data = [(cls, img) for cls in self.classes for img in os.listdir(os.path.join(self.root_dir, cls))]
        print(f"Loaded {split} set with {len(self.data)} images across {len(self.classes)} classes.")

    def __getitem__(self, index):
        class_name, img_name = self.data[index]
        class_idx = self.class_to_idx[class_name]
        img_path = os.path.join(self.root_dir, class_name, img_name)
        image = Image.open(img_path).convert("RGB")
        return transform(image), class_idx, img_name  # Returning image name for identification

    def __len__(self):
        return len(self.data)

# Model Definition
class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc2(F.relu(self.fc1(x)))

# Compute Prototypes
def compute_prototypes(embeddings, labels):
    unique_labels = torch.unique(labels)
    prototypes = torch.stack([embeddings[labels == lbl].mean(dim=0) for lbl in unique_labels])
    return prototypes, unique_labels

# Cosine Distance
def cosine_distance(x, y):
    return 1 - F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)

# Prototype Loss
def prototype_loss(prototypes, query_embeddings, query_labels, unique_labels):
    dists = cosine_distance(query_embeddings, prototypes)
    target_idxs = torch.tensor([torch.where(unique_labels == lbl)[0].item() for lbl in query_labels if lbl in unique_labels], device=query_labels.device)
    return F.cross_entropy(-dists, target_idxs) if len(target_idxs) > 0 else torch.tensor(0.0, requires_grad=True, device=query_labels.device)

# Load Data
data_path = r"D:\\Programming\\Github projects\\Few-shot learning\\Few-shot-learning\\new_dataset\\structured_train"
support_loader = DataLoader(FewShotDataset(data_path, "support"), batch_size=16, shuffle=True)
query_loader = DataLoader(FewShotDataset(data_path, "query"), batch_size=1, shuffle=False)

# Training Setup
model = ProtoNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
num_epochs = 20

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for support_images, support_labels, _ in support_loader:
        support_images, support_labels = support_images.to(device), support_labels.to(device)
        query_images, query_labels, _ = next(iter(query_loader))
        query_images, query_labels = query_images.to(device), query_labels.to(device)
        
        support_embeddings = model(support_images)
        query_embeddings = model(query_images)
        prototypes, unique_labels = compute_prototypes(support_embeddings, support_labels)
        
        loss = prototype_loss(prototypes, query_embeddings, query_labels, unique_labels)
        if loss.item() > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(support_loader):.4f}")

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for query_images, query_labels, img_name in query_loader:
        query_images, query_labels = query_images.to(device), query_labels.to(device)
        query_embeddings = model(query_images)
        dists = cosine_distance(query_embeddings, prototypes)
        preds = unique_labels[torch.argmin(dists, dim=1)]
        
        result = "Porsche" if preds == query_labels else "Not a Porsche"
        print(f"Image: {img_name[0]} -> Prediction: {result}")
        
        correct += (preds == query_labels).sum().item()
        total += query_labels.size(0)

accuracy = 100 * correct / total if total > 0 else 0
print(f"Test Accuracy: {accuracy:.2f}%")

# Save Model
torch.save(model.state_dict(), "few_shot_model.pth")
print("✅ Model saved as 'few_shot_model.pth'")
