import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from collections import deque
import csv
import time
import threading
from fastapi import FastAPI
import uvicorn

# --- 1. SETUP: AI & LABELS ---
print("Initializing Security AI (YAMNet)...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_labels():
    labels = []
    # Ensure this CSV file is in your D:\Ishan\Signal-Watcher folder
    with open('yamnet_class_map.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row['display_name'])
    return labels

class_names = load_labels()

# --- 2. GLOBAL STATE ---
audio_memory = deque(maxlen=10)   # Last 1.0s of raw audio
event_history = deque(maxlen=15)  # Last 15 identified sounds
current_status = {"level": "NORMAL", "detail": "System Monitoring...", "timestamp": ""}

# Define patterns to watch for
THREAT_PATTERNS = [
    ("Glass", "Footsteps", 10),
    ("Glass", "Whispering", 10),
    ("Screaming", "Crying", 5)
]

# --- 3. THE LOGIC BRAIN ---
def check_patterns():
    global current_status
    if len(event_history) == 0: return
    
    latest = event_history[-1]
    
    # A. IMMEDIATE THREAT DETECTION (Single Sound)
    # If the AI hears these, we don't wait for a sequence!
    emergency_keywords = ["Siren", "Alarm", "Explosion", "Gunshot", "Screaming"]
    if any(k.lower() in latest['label'].lower() for k in emergency_keywords):
        current_status = {
            "level": "CRITICAL",
            "detail": f"IMMEDIATE THREAT: {latest['label']} detected!",
            "timestamp": time.ctime()
        }
        return

    # B. SEQUENCE DETECTION (Two sounds in a row)
    events = list(event_history)
    for i in range(len(events)-1):
        for j in range(i+1, len(events)):
            a, b = events[i], events[j]
            gap = b['time'] - a['time']
            
            for p1, p2, max_t in THREAT_PATTERNS:
                if p1.lower() in a['label'].lower() and p2.lower() in b['label'].lower():
                    if 0 < gap <= max_t:
                        current_status = {
                            "level": "CRITICAL",
                            "detail": f"Pattern Match: {a['label']} followed by {b['label']}",
                            "timestamp": time.ctime()
                        }
                        return

# --- 4. THE AI EAR (Background Thread) ---
def run_ai_ear():
    FS, TARGET_FS = 44100, 16000
    
    def callback(indata, frames, time_info, status):
        audio_memory.append(indata.copy())
        
        # Only process if volume is high enough
        if np.max(np.abs(indata)) > 0.03: 
            full_buffer = np.concatenate(list(audio_memory)).flatten()
            resampled = librosa.resample(full_buffer, orig_sr=FS, target_sr=TARGET_FS)
            
            # AI Inference
            scores, _, _ = model(resampled)
            mean_scores = np.mean(scores, axis=0)
            top_class = np.argmax(mean_scores)
            
            label = class_names[top_class]
            conf = mean_scores[top_class]
            
            # Gate: Only record sounds the AI is >25% sure about
            if conf > 0.25 and label not in ['Silence', 'Background noise']:
                event_history.append({"label": label, "time": time.time()})
                print(f"[AI] Heard: {label} ({conf:.1%})")
                check_patterns()

    with sd.InputStream(device=1, channels=1, samplerate=FS, blocksize=4410, callback=callback):
        print("--- Ear is Listening ---")
        while True: sd.sleep(1000)

# --- 5. THE WEB INTERFACE (FastAPI) ---
app = FastAPI()

@app.get("/status")
async def get_status():
    return current_status

@app.get("/history")
async def get_history():
    return list(event_history)

@app.get("/reset")
async def reset_status():
    global current_status
    current_status = {"level": "NORMAL", "detail": "System Reset", "timestamp": time.ctime()}
    event_history.clear()
    return {"message": "System status cleared."}

if __name__ == "__main__":
    # Start AI in background
    threading.Thread(target=run_ai_ear, daemon=True).start()
    
    # Start Web Server
    uvicorn.run(app, host="127.0.0.1", port=8000)