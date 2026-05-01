import streamlit as st
import torch
import numpy as np
import joblib
from model import CropModel

st.set_page_config(page_title="Crop AI", layout="wide")

@st.cache_resource
def load_all():
    labels = joblib.load("labels.pkl")
    model = CropModel(num_classes=len(labels))
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model, labels

model, labels = load_all()

st.title("🌱 AI Crop Recommendation System (Deep Learning)")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Soil Data")
    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 0, 140, 40)
    K = st.slider("Potassium (K)", 0, 200, 40)

with c2:
    st.subheader("Weather Data")
    temperature = st.slider("Temperature (°C)", 0, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 60)

    lat = st.number_input("Latitude", value=23.5)
    lon = st.number_input("Longitude", value=77.5)
    st.map({"lat": [lat], "lon": [lon]})

if st.button("🚀 Predict Crop", use_container_width=True):
    soil = torch.tensor([[N, P, K]], dtype=torch.float32) / 100.0

    weather = np.array([[temperature, humidity]] * 7, dtype=np.float32) / 100.0
    ndvi = np.random.rand(7, 1).astype(np.float32)  # placeholder
    combined = np.concatenate((weather, ndvi), axis=1)
    combined = torch.tensor([combined], dtype=torch.float32)

    with torch.no_grad():
        out = model(combined, soil)
        pred = int(torch.argmax(out, dim=1).item())

    st.success(f"🌾 Recommended Crop: **{labels[pred].upper()}**")