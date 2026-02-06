import sounddevice as sd
import numpy as np
from collections import deque

# --- Settings ---
FS = 44100
CHUNK_SIZE = 4410  # 0.1 seconds per chunk
# We want to keep 10 chunks to make a full 1.0 second of memory
BUFFER_LENGTH = 10 
DEVICE_ID = 1
THRESHOLD = 0.1

# This is our 'Circular Buffer'
audio_memory = deque(maxlen=BUFFER_LENGTH)

print("--- Step 2: Buffer Preservation Active ---")
print("Listening... I will capture 1.0s of audio when a sound is detected.")

def audio_callback(indata, frames, time, status):
    # 1. Add the new chunk to our memory pipe
    # We use .copy() because 'indata' gets overwritten by the hardware
    audio_memory.append(indata.copy())
    
    # 2. Check for the 'Loud Noise' (Project 1 logic)
    peak = np.max(np.abs(indata))
    
    if peak > THRESHOLD:
        print(f"\n[!] Triggered! Peak: {peak:.4f}")
        
        # 3. CONSOLIDATE: Combine all chunks in the pipe into one long 1s array
        full_second = np.concatenate(list(audio_memory))
        
        print(f"Memory Captured! Shape of full buffer: {full_second.shape}")
        # In Step 3, we will send 'full_second' to the Mel-Converter
        # For now, we just prove we caught it.

try:
    with sd.InputStream(device=DEVICE_ID, channels=1, samplerate=FS, 
                       blocksize=CHUNK_SIZE, callback=audio_callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nWatcher Stopped.")