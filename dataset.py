import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CropDataset(Dataset):
    def __init__(self, crop_csv, ndvi_csv):
        self.crop_data = pd.read_csv(crop_csv)
        self.ndvi_data = pd.read_csv(ndvi_csv)

        # Encode labels
        self.labels = self.crop_data['label'].astype('category').cat.codes

    def __len__(self):
        return len(self.crop_data)

    def __getitem__(self, idx):
        row = self.crop_data.iloc[idx]

        # ========================
        # SOIL (NPK)
        # ========================
        soil = np.array([row['N'], row['P'], row['K']], dtype=np.float32)
        soil = soil / 100.0  # normalize

        # ========================
        # WEATHER (REAL FROM DATASET)
        # ========================
        weather = np.array([
            [row['temperature'], row['humidity']]
        ] * 7, dtype=np.float32)

        weather = weather / 100.0  # normalize

        # ========================
        # NDVI TIME SERIES (FROM CSV)
        # ========================
        ndvi_values = self.ndvi_data.iloc[idx % len(self.ndvi_data)].values.astype(np.float32)

        # Take first 7 timesteps
        ndvi_series = ndvi_values[:7].reshape(7, 1)

        # Normalize NDVI (optional but safe)
        ndvi_series = ndvi_series / 1.0

        # ========================
        # COMBINE WEATHER + NDVI
        # ========================
        combined = np.concatenate((weather, ndvi_series), axis=1)  # shape (7,3)

        label = self.labels[idx]

        return (
            torch.tensor(combined, dtype=torch.float32),  # (7,3)
            torch.tensor(soil, dtype=torch.float32),      # (3,)
            torch.tensor(label, dtype=torch.long)
        )