import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_model():
    data = joblib.load("crop_model.joblib")
    return data["model"], data["features"]

model, features = load_model()
def create_database():
    conn = sqlite3.connect("crop_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            temperature REAL,
            humidity REAL,
            ph REAL,
            rainfall REAL,
            predicted_crop TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


create_database()

st.title("🌱 AI-Based Crop Recommendation System")
st.write(
    "Discover a suitable crop based on soil nutrients "
    "and weather conditions."
)

st.divider()
st.subheader("Enter Soil & Weather Details")

with st.form("crop_form"):
    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0)
        P = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0)
        K = st.number_input("Potassium (K)", 0.0, 250.0, 43.0)
        temperature = st.number_input(
            "Temperature (°C)", -10.0, 60.0, 20.8
        )

    with col2:
        humidity = st.number_input(
            "Humidity (%)", 0.0, 100.0, 82.0
        )
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input(
            "Rainfall (mm)", 0.0, 3000.0, 203.0
        )

    submitted = st.form_submit_button(
        "🌾 Predict Suitable Crop",
        use_container_width=True
    )


if submitted:
    values = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    sample = pd.DataFrame([values])[features]
    prediction = model.predict(sample)[0]
    confidence = model.predict_proba(sample).max() * 100

    conn = sqlite3.connect("crop_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            nitrogen, phosphorus, potassium,
            temperature, humidity, ph, rainfall,
            predicted_crop, confidence, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        N, P, K, temperature, humidity, ph, rainfall,
        str(prediction), float(confidence),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    st.success(f"Recommended Crop: {str(prediction).title()}")
    st.metric("Model Confidence", f"{confidence:.2f}%")
    st.caption(
        "This is a model prediction based on the supplied inputs, "
        "not a guarantee of crop yield or suitability in the field."
    )
