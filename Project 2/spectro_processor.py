import librosa
import numpy as np

def create_spectrogram_matrix(audio_data, sample_rate=44100):
    """
    Converts raw audio samples into a Log-Mel Spectrogram matrix.
    """
    # Ensure data is 'flat' (1D) for librosa processing
    audio_flat = audio_data.flatten()

    # Step 3: Compute the Mel-Spectrogram
    # n_fft: The window size (how many samples to look at for each 'pitch' calculation)
    # hop_length: How far to slide the window (smaller = higher time resolution)
    # n_mels: How many frequency 'bins' (rows) to create
    mel_signal = librosa.feature.melspectrogram(
        y=audio_flat, 
        sr=sample_rate, 
        n_fft=2048, 
        hop_length=512, 
        n_mels=128
    )

    # Step 4: Power to Decibels (Log Scaling)
    # This makes the quiet frequencies visible and the loud ones non-blinding.
    mel_db = librosa.power_to_db(mel_signal, ref=np.max)

    return mel_db

if __name__ == "__main__":
    # Test with dummy data to verify math works
    test_data = np.random.uniform(-1, 1, 44100)
    matrix = create_spectrogram_matrix(test_data)
    print(f"Success! Spectrogram Matrix Shape: {matrix.shape}")
    print("Interpretation: 128 rows (Frequency) by ~87 columns (Time)")