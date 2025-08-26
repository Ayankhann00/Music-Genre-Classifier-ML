🎵 Music Genre Classifier
A deep learning project that classifies music tracks into genres using Mel spectrograms extracted from audio files. This project demonstrates the power of audio processing combined with Convolutional Neural Networks (CNN) for music genre classification.
📂 Table of Contents
Project Overview
Features
Dataset
Installation
Usage
Model Architecture
Screenshots
Future Work
License
📌 Project Overview
This system processes audio files and converts them into Mel spectrograms, which are images representing audio frequency content over time. These spectrograms are then fed into a CNN model to classify music into 10 genres:
Genres: Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock
Key Steps:
Load .wav audio files
Chunk audio into 4-second segments with 2-second overlap
Generate Mel spectrograms for each chunk
Preprocess and resize spectrograms for CNN input
Train a CNN to classify genres
⚡ Features
Processes .wav audio files
Audio chunking with overlap for better feature extraction
Mel spectrogram visualization
CNN-based classification
Supports 10 music genres
🗃 Dataset
Uses the GTZAN Dataset, structured as:
Data/genres_original/
    blues/
    classical/
    country/
    disco/
    hiphop/
    jazz/
    metal/
    pop/
    reggae/
    rock/
Each folder contains .wav files corresponding to the genre.
⚙ Installation
Clone the repo:
git clone https://github.com/Ayankhann00/Music-Genre-Classifier-ML.git
cd Music-Genre-Classifier-ML
Install dependencies:
pip install librosa matplotlib tensorflow numpy
Place your dataset in Data/genres_original.
🖥 Usage
Visualize Waveform
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("Data/genres_original/blues/blues.00000.wav", sr=44100)
plt.figure(figsize=(14,5))
librosa.display.waveshow(y, sr=sr)
plt.show()
Example Waveform:
Plot Mel Spectrogram
from utils import plot_melspectrogram
plot_melspectrogram(y, sr)
Example Spectrogram:
Preprocess Data
from preprocess import load_and_preprocess_data

classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
data, labels = load_and_preprocess_data("Data/genres_original", classes)
🏗 Model Architecture
The CNN model contains:
Conv2D layers for feature extraction
MaxPooling layers for downsampling
Flatten layer
Dense layers for classification
Dropout for regularization
Adam optimizer
The model learns patterns in spectrograms to classify genres accurately.
Training Accuracy & Loss:
📸 Screenshots
Waveform of Blues Track
Mel Spectrogram of Chunked Audio
Training Accuracy & Loss Plots
Sample Predictions
(Add your images in the images/ folder and replace links above)
🚀 Future Work
Real-time genre prediction via web interface
Explore additional audio features: MFCCs, Chroma
Data augmentation to improve model accuracy
Support more audio formats like .mp3
📄 License
MIT License © 2025 Ayan Khan
