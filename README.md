Music Genre Classifier
A deep learning project that classifies music tracks into genres using Mel spectrogram features extracted from audio files.
Table of Contents
Project Overview
Features
Dataset
Installation
Usage
Model Architecture
Screenshots
Future Work
License
Project Overview
This project implements a machine learning pipeline for music genre classification. The system uses audio processing techniques to convert songs into Mel spectrograms, which are then fed into a Convolutional Neural Network (CNN) for classification.
Key steps include:
Audio chunking with overlap
Mel spectrogram extraction
Data preprocessing and resizing
CNN-based classification
Features
Processes audio files in .wav format
Splits songs into 4-second overlapping chunks (2-second overlap)
Extracts Mel spectrograms for each chunk
Uses CNN for genre classification
Supports 10 music genres: Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock
Dataset
The project uses the GTZAN Dataset for training and testing. The dataset is organized by genre:
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
Installation
Clone this repository:
git clone https://github.com/Ayankhann00/Music-Genre-Classifier-ML.git
cd Music-Genre-Classifier-ML
Install dependencies:
pip install librosa matplotlib tensorflow numpy
Ensure your dataset is placed in Data/genres_original.
Usage
Visualize Waveform
import librosa
import matplotlib.pyplot as plt
y, sr = librosa.load("Data/genres_original/blues/blues.00000.wav", sr=44100)
plt.figure(figsize=(14,5))
librosa.display.waveshow(y, sr=sr)
plt.show()
Plot Mel Spectrogram
from utils import plot_melspectrogram
plot_melspectrogram(y, sr)
Preprocess Data
from preprocess import load_and_preprocess_data
data, labels = load_and_preprocess_data("Data/genres_original", classes)
Model Architecture
The CNN model used for classification consists of:
2D Convolutional layers
MaxPooling layers
Flatten layer
Fully connected Dense layers
Dropout for regularization
Adam optimizer
This architecture is capable of learning patterns in Mel spectrograms to classify the genre effectively.
Screenshots
Include screenshots such as:
Waveform of a song
Mel spectrograms
Training loss and accuracy plots
Sample predictions
Future Work
Integrate with a web interface for real-time genre prediction
Experiment with more advanced audio features (MFCCs, Chroma)
Improve model accuracy with data augmentation
Support additional audio formats like .mp3
License
This project is licensed under the MIT License.
