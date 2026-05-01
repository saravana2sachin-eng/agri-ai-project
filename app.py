import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Smart Crop AI", layout="wide")

# Load model
model = joblib.load("crop_model.pkl")

# Title
st.title("🌱 Smart Crop Recommendation System")
st.markdown("AI-powered agriculture assistant")

# Layout
col1, col2 = st.columns(2)

# LEFT SIDE — Inputs
with col1:
    st.header("🧪 Soil Inputs")

    N = st.number_input("Nitrogen (N)", 0, 140)
    P = st.number_input("Phosphorus (P)", 0, 140)
    K = st.number_input("Potassium (K)", 0, 200)

    st.header("🌦 Environment")

    temperature = st.number_input("Temperature (°C)")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    rainfall = st.number_input("Rainfall (mm)")

# RIGHT SIDE — Output
with col2:
    st.header("📊 Prediction Result")

    if st.button("🚀 Predict Crop"):

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        result = model.predict(data)

        st.success(f"🌾 Recommended Crop: **{result[0].upper()}**")

        st.markdown("---")
        st.subheader("📌 Summary")

        st.write(f"""
        ✔ Nitrogen: {N}  
        ✔ Phosphorus: {P}  
        ✔ Potassium: {K}  
        ✔ Temperature: {temperature}°C  
        ✔ Humidity: {humidity}%  
        ✔ pH: {ph}  
        ✔ Rainfall: {rainfall} mm  
        """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning + Streamlit")