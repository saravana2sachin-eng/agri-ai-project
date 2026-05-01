import torch
import torch.nn as nn

# ================= LSTM =================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


# ================= SOIL =================
class SoilModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

    def forward(self, x):
        return self.fc(x)


# ================= FINAL MODEL =================
class CropModel(nn.Module):
    def __init__(self, num_classes=22):
        super().__init__()

        self.lstm = LSTMModel()
        self.soil = SoilModel()

        self.fc = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),   # regularization
            nn.Linear(128, num_classes)
        )

    def forward(self, weather_ndvi, soil):
        # weather_ndvi: (batch, 7, 3)
        # soil: (batch, 3)

        lstm_out = self.lstm(weather_ndvi)   # (batch, 64)
        soil_out = self.soil(soil)           # (batch, 32)

        combined = torch.cat((lstm_out, soil_out), dim=1)

        output = self.fc(combined)
        return output