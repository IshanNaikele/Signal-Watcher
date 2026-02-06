import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from collections import deque
import csv

# --- 1. Load the AI and Labels ---
print("Loading YAMNet AI Model...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_labels():
    labels = []
    with open('yamnet_class_map.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row['display_name'])
    return labels

class_names = load_labels()

# --- 2. Audio Settings ---
FS = 44100
TARGET_FS = 16000
CHUNK_SIZE = 4410  # 0.1 seconds
BUFFER_LENGTH = 10 # 1.0 second total
THRESHOLD = 0.05
DEVICE_ID = 1

audio_memory = deque(maxlen=BUFFER_LENGTH)

# --- 3. The Logic ---
def audio_callback(indata, frames, time, status):
    audio_memory.append(indata.copy())
    peak = np.max(np.abs(indata))
    
    if peak > THRESHOLD:
        # Consolidate memory
        full_buffer = np.concatenate(list(audio_memory)).flatten()
        
        # Resample to 16kHz
        resampled = librosa.resample(full_buffer, orig_sr=FS, target_sr=TARGET_FS)
        
        # AI Inference
        scores, embeddings, spectrogram = model(resampled)
        
        # Get Top Result
        mean_scores = np.mean(scores, axis=0) # Average scores across the 1s window
        top_class = np.argmax(mean_scores)
        label = class_names[top_class]
        confidence = mean_scores[top_class]
        
        print(f"[{label}] detected with {confidence:.2%} confidence")

# --- 4. Start Listening ---
print("--- Project 3: Live Inference Active ---")
with sd.InputStream(device=DEVICE_ID, channels=1, samplerate=FS, 
                   blocksize=CHUNK_SIZE, callback=audio_callback):
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("\nAI Ear Stopped.")