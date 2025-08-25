import streamlit as st
import pandas as pd
import joblib

st.title("🎵 Music Genre Classification (Lightweight Model)")

model, scaler, pca, le = joblib.load("music_genre.pkl")

uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "filename" in df.columns:
        df = df.drop("filename", axis=1)

    X_scaled = scaler.transform(df)
    X_pca = pca.transform(X_scaled)
    predictions = model.predict(X_pca)
    predicted_labels = le.inverse_transform(predictions)

    result = pd.DataFrame({"Prediction": predicted_labels})
    st.write(result)
