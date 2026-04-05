import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from model_engine import SilentSightModel

# 1. Setup Custom Dataset Class
class AptosDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.png')
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# 2. Training Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # New: 50% chance to flip
    transforms.RandomVerticalFlip(p=0.5),        # New: 50% chance to flip vertically
    transforms.RandomRotation(degrees=15),       # New: Rotate +/- 15 degrees
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Load Data
dataset = AptosDataset(csv_file='data/train.csv', img_dir='data/train_images/', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Initialize and Train
model = SilentSightModel(num_classes=5).to(device)
class_weights = torch.tensor([1.0,2.5,2.0,4.0,4.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("Starting training... this will take time.")
for epoch in range(5): # Start with 5 epochs for the hackathon
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")

# 5. SAVE THE BRAIN
torch.save(model.state_dict(), 'best_model.pth')
print("Saved best_model.pth! Now you can run your app.py")