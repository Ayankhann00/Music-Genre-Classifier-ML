# app.py
import streamlit as st
import numpy as np
import librosa
import joblib
import os


model = joblib.load("music_genre_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs.flatten()
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

if app_mode == "Home":
    st.title("🎶 Music Genre Classification (ML Version)")
    st.markdown("Upload an audio file and let the ML model predict its genre.")

elif app_mode == "About Project":
    st.header("About Project")
    st.write("""
    This project classifies music into genres using **MFCC features** extracted 
    with Librosa and a **RandomForest classifier** trained in Scikit-learn.
    
    **Genres Covered:** Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock
    """)

elif app_mode == "Prediction":
    st.header("Upload Audio File for Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if test_mp3 is not None:
        filepath = "uploaded_" + test_mp3.name
        with open(filepath, "wb") as f:
            f.write(test_mp3.getbuffer())

        st.audio(test_mp3)

        if st.button("Predict"):
            features = extract_features(filepath)
            if features is not None:
                features = scaler.transform([features])
                prediction = model.predict(features)[0]
                genre = encoder.inverse_transform([prediction])[0]
                st.success(f"🎵 Predicted Genre: **{genre}**")
