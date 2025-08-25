import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

with open("genre_classifier.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎵 Music Genre Classifier")
st.write("Upload pre-extracted feature CSV (single row) to predict genre.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    prediction = model.predict(X_scaled)
    st.success(f"Predicted Genre: {prediction[0]}")
