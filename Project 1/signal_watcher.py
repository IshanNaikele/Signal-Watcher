import sounddevice as sd
import numpy as np

# --- Settings ---
FS = 44100
CHUNK_SIZE = 2048   # Small chunk for fast reaction
THRESHOLD = 0.5    # Adjust this: 0.05 is usually a loud word or a clap
DEVICE_ID = 1

print(f"--- Signal Watcher Active (Device {DEVICE_ID}) ---")
print(f"Listening... (Only showing sounds above {THRESHOLD})")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    
    # Use 'Peak' instead of 'RMS' to catch sudden sharp sounds
    peak = np.max(np.abs(indata))
    
    # Logic: Only interact if sound is LARGE
    if peak > THRESHOLD:
        # Create a visual bar that grows with volume
        bar_length = int(peak * 40)
        bar = "█" * bar_length
        print(f"LOUD NOISE! | Level: {peak:.4f} | {bar}")

try:
    with sd.InputStream(device=DEVICE_ID, 
                       channels=1, 
                       samplerate=FS, 
                       blocksize=CHUNK_SIZE, 
                       callback=audio_callback):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nWatcher Stopped.")