import streamlit as st
import numpy as np
import joblib

from weather import get_weather
from ndvi import get_ndvi

# Cache for speed
@st.cache_data
def fetch_weather(lat, lon):
    return get_weather(lat, lon)

@st.cache_data
def fetch_ndvi(lat, lon):
    return get_ndvi(lat, lon)

# Load model
model, scaler, encoder = joblib.load("crop_model.pkl")

st.title("🌱 Smart Crop Recommendation (Advanced AI)")

# Inputs
N = st.number_input("Nitrogen", 0, 140)
P = st.number_input("Phosphorus", 0, 140)
K = st.number_input("Potassium", 0, 200)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)
ph = st.number_input("Soil pH", 0.0, 14.0)

lat = st.number_input("Latitude", value=23.5)
lon = st.number_input("Longitude", value=77.5)

# Map
st.map({"lat": [lat], "lon": [lon]})

if st.button("Predict"):

    # Weather
    try:
        temperature, rainfall = fetch_weather(lat, lon)
        st.write(f"🌡 Temperature: {temperature:.2f}")
        st.write(f"🌧 Rainfall: {rainfall:.2f}")
    except Exception as e:
        st.error(f"Weather error: {e}")
        st.stop()

    # NDVI
    try:
        ndvi = fetch_ndvi(lat, lon)
        st.write(f"🌿 NDVI: {ndvi:.3f}")
    except Exception as e:
        st.error(f"NDVI error: {e}")
        st.stop()

    # Prediction (8 FEATURES)
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall, ndvi]])
    data = scaler.transform(data)

    pred = model.predict(data)

    # 🔥 Convert back to crop name
    result = encoder.inverse_transform(pred)

    st.success(f"🌾 Recommended Crop: {result[0]}")