import streamlit as st
import torch
import numpy as np
import joblib

from model import CropModel

# ================= LOAD =================
@st.cache_resource
def load_model():
    model = CropModel()
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

labels = joblib.load("labels.pkl")

# ================= UI =================
st.set_page_config(page_title="Crop AI", layout="wide")

st.title("🌱 AI Crop Recommendation System")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Soil Data")
    N = st.slider("Nitrogen", 0, 140, 50)
    P = st.slider("Phosphorus", 0, 140, 40)
    K = st.slider("Potassium", 0, 200, 40)

with col2:
    st.subheader("Weather")
    temp = st.slider("Temperature", 0, 50, 25)
    hum = st.slider("Humidity", 0, 100, 60)

    lat = st.number_input("Latitude", value=23.5)
    lon = st.number_input("Longitude", value=77.5)
    st.map({"lat": [lat], "lon": [lon]})

# ================= PREDICT =================
if st.button("Predict Crop"):

    soil = torch.tensor([[N, P, K]], dtype=torch.float32) / 100.0

    weather = np.array([[temp, hum]] * 7, dtype=np.float32) / 100.0
    ndvi = np.random.rand(7, 1).astype(np.float32)

    combined = np.concatenate((weather, ndvi), axis=1)
    combined = torch.tensor([combined], dtype=torch.float32)

    with torch.no_grad():
        output = model(combined, soil)
        pred = torch.argmax(output, dim=1).item()

    st.success(f"🌾 Recommended Crop: **{labels[pred].upper()}**")