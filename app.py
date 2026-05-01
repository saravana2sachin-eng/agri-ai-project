import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Smart Crop AI", layout="wide")

# Load model
model = joblib.load("crop_model.pkl")

# Title
st.title("🌱 Smart Crop Recommendation System")
st.markdown("### AI-powered agriculture assistant 🌾")

st.markdown("---")

# Layout (2 columns)
col1, col2 = st.columns([1, 1])

# =========================
# LEFT SIDE (INPUTS)
# =========================
with col1:
    st.subheader("🧪 Soil Parameters")

    N = st.slider("Nitrogen (N)", 0, 140, 50)
    P = st.slider("Phosphorus (P)", 0, 140, 40)
    K = st.slider("Potassium (K)", 0, 200, 40)

    st.subheader("🌦 Environmental Conditions")

    temperature = st.slider("Temperature (°C)", 0, 50, 25)
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 100)

    st.subheader("📍 Location")

    lat = st.number_input("Latitude", value=23.5)
    lon = st.number_input("Longitude", value=77.5)

# =========================
# RIGHT SIDE (OUTPUT + MAP)
# =========================
with col2:
    st.subheader("🗺 Farm Location")

    st.map({"lat": [lat], "lon": [lon]})

    st.markdown("---")

    st.subheader("📊 Prediction")

    if st.button("🚀 Predict Crop", use_container_width=True):

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        result = model.predict(data)

        st.success(f"🌾 Recommended Crop: **{result[0].upper()}**")

        st.markdown("---")

        st.subheader("📌 Input Summary")

        st.write(f"""
        **Soil:**
        - Nitrogen: {N}
        - Phosphorus: {P}
        - Potassium: {K}

        **Environment:**
        - Temperature: {temperature} °C
        - Humidity: {humidity} %
        - pH: {ph}
        - Rainfall: {rainfall} mm

        **Location:**
        - Latitude: {lat}
        - Longitude: {lon}
        """)

# Footer
st.markdown("---")
st.caption("🚀 Built using Machine Learning + Streamlit | Smart Agriculture AI")