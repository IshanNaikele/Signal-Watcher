import sounddevice as sd
import numpy as np

# Configuration
FS = 44100        # Sample Rate (CD Quality)
CHANNELS = 1      # Mono (we don't need Stereo for a clap)
CHUNK_SIZE = 22050 # 0.5 seconds of audio per buffer (44100 / 2)
DEVICE_ID = 1     # From your discovery script

print(f"Opening stream on device {DEVICE_ID}...")

def audio_callback(indata, frames, time, status):
    """
    This function is called automatically for every 'chunk' of sound.
    """
    if status:
        print(status)
    
    # indata is the raw audio buffer
    print(f"Captured a chunk! Shape: {indata.shape}")

# Start the stream
with sd.InputStream(device=DEVICE_ID, 
                   channels=CHANNELS, 
                   samplerate=FS, 
                   blocksize=CHUNK_SIZE, 
                   callback=audio_callback):
    print("Stream is live. Press Ctrl+C to stop.")
    while True:
        sd.sleep(1000) # Keep the script alive