import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import CropModel
from dataset import CropDataset

# ================= SETTINGS =================
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.001

# ================= DATA =================
dataset = CropDataset(
    "Crop_recommendation.csv",
    "ndvi-2010-2020.csv"
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL =================
model = CropModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= TRAIN =================
for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for weather_ndvi, soil, labels in loader:

        optimizer.zero_grad()

        # Forward pass (ONLY 2 inputs)
        outputs = model(weather_ndvi, soil)

        # Loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), "dl_crop_model.pth")

print("✅ Deep Learning Model Trained & Saved Successfully!")