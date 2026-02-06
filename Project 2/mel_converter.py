import librosa
import numpy as np
def process_to_mel(audio_data, sr=44100):
    # Ensure data is 1D for Librosa
    audio_1d = audio_data.flatten()
    
    # STEP 3: Generate the Mel-Spectrogram
    # n_mels=128 means we divide the vertical 'pitch' into 128 rows
    mel_spec = librosa.feature.melspectrogram(y=audio_1d, sr=sr, n_mels=128)
    
    # STEP 4: Convert to Decibels (Log Scale)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    print(f"Spectrogram Matrix Created! Shape: {mel_db.shape}")
    return mel_db