import sounddevice as sd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import librosa.display
import time
import matplotlib
matplotlib.use('Agg') # This prevents the 'Main Thread' error by disabling the GUI
import matplotlib.pyplot as plt
# Import the math logic you just wrote
from spectro_processor import create_spectrogram_matrix

# --- Settings ---
FS = 44100
CHUNK_SIZE = 4410  
BUFFER_LENGTH = 10 
THRESHOLD = 0.1
DEVICE_ID = 1

audio_memory = deque(maxlen=BUFFER_LENGTH)

def save_visual(matrix, filename):
    """Turns the matrix into a colorful PNG image."""
    plt.figure(figsize=(10, 4))
    
    # librosa.display handles the axes automatically
    librosa.display.specshow(matrix, sr=FS, hop_length=512, x_axis='time', y_axis='mel')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectro-Vision Capture')
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.close() # Important to free up memory
    print(f"--- SUCCESS: {filename} saved! ---")

def audio_callback(indata, frames, time_info, status):
    audio_memory.append(indata.copy())
    peak = np.max(np.abs(indata))
    
    if peak > THRESHOLD:
        print(f"Noise Triggered! (Peak: {peak:.2f})")
        
        # Combine memory into 1 second
        full_second = np.concatenate(list(audio_memory))
        
        # Step 3 & 4: Process to Matrix
        matrix = create_spectrogram_matrix(full_second)
        
        # Step 5: Save to Image
        timestamp = int(time.time())
        save_visual(matrix, f"capture_{timestamp}.png")

print("--- Project 2: Spectro-Vision Active ---")
print("Make a loud sound (whistle vs clap) to see the difference...")

try:
    with sd.InputStream(device=DEVICE_ID, channels=1, samplerate=FS, 
                       blocksize=CHUNK_SIZE, callback=audio_callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nVision Stopped.")