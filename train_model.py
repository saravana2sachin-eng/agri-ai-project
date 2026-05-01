import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

print("Starting improved training...")

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Add NDVI
df['ndvi'] = 0.5

# Features
X = df[['N','P','K','temperature','humidity','ph','rainfall','ndvi']]
y = df['label']

# 🔥 Encode labels (IMPORTANT)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Save EVERYTHING
joblib.dump((model, scaler, encoder), "crop_model.pkl")

print("Model saved successfully!")