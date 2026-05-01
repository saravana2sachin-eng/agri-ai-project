import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import joblib

from model import CropModel
from dataset import CropDataset

# ================= SETTINGS =================
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.0005

CROP_CSV = "Crop_recommendation.csv"
NDVI_CSV = "ndvi-2010-2020.csv"

# ================= SAVE LABELS =================
df = pd.read_csv(CROP_CSV)
labels = df['label'].astype('category')
label_list = labels.cat.categories.tolist()

joblib.dump(label_list, "labels.pkl")
print("✅ Labels saved!")

# ================= DATA =================
dataset = CropDataset(CROP_CSV, NDVI_CSV)

# Train / Validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= MODEL =================
model = CropModel(num_classes=len(label_list)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= TRAIN =================
best_val_acc = 0

for epoch in range(EPOCHS):

    # ---------- TRAIN ----------
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for weather_ndvi, soil, labels_batch in train_loader:

        weather_ndvi = weather_ndvi.to(device)
        soil = soil.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = model(weather_ndvi, soil)

        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels_batch).sum().item()
        train_total += labels_batch.size(0)

    train_acc = train_correct / train_total

    # ---------- VALIDATION ----------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for weather_ndvi, soil, labels_batch in val_loader:

            weather_ndvi = weather_ndvi.to(device)
            soil = soil.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(weather_ndvi, soil)

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels_batch).sum().item()
            val_total += labels_batch.size(0)

    val_acc = val_correct / val_total

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# ================= SAVE FINAL MODEL =================
torch.save(model.state_dict(), "dl_crop_model.pth")

print("🎉 Training complete!")
print(f"✅ Best Validation Accuracy: {best_val_acc:.4f}")
print("✅ Best model saved: best_model.pth")
print("✅ Final model saved: dl_crop_model.pth")
print("✅ Labels saved: labels.pkl")