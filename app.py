import streamlit as st
import pandas as pd
import joblib

clf = joblib.load("music_genre_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🎵 Music Genre Classification App")
st.write("Upload your song features (CSV row) to predict the genre.")

uploaded_file = st.file_uploader("Upload CSV file with features", type=["csv"])

if uploaded_file is not None: 
    data = pd.read_csv(uploaded_file)
    if "label" in data.columns:
        data = data.drop(columns=["label"])
    if "filename" in data.columns:
        data = data.drop(columns=["filename"])

    st.write("Uploaded Features:", data)

    X = scaler.transform(data)
    preds = clf.predict(X)
    pred_labels = le.inverse_transform(preds)

    st.subheader("Predicted Genre(s):")
    for i, label in enumerate(pred_labels):
        st.write(f"Sample {i+1}: {label}")

