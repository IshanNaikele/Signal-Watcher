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
from fastapi.responses import HTMLResponse
import uvicorn
import requests

# --- CONFIGURATION ---
# Paste your verified credentials here
TELEGRAM_TOKEN = "8020215956:AAG08ViuPxg2xan5q2DsBr2ocPrOrWEPtg0"
CHAT_ID = "6157888322" 
LOG_FILE = "security_log.csv"

# --- 1. SETUP: AI & LABELS ---
print("Initializing Security AI (YAMNet)...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_labels():
    labels = []
    try:
        with open('yamnet_class_map.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row['display_name'])
    except FileNotFoundError:
        print("Error: yamnet_class_map.csv not found!")
    return labels

class_names = load_labels()

# --- 2. GLOBAL STATE ---
audio_memory = deque(maxlen=10)   # Last 1.0s of raw audio
event_history = deque(maxlen=15)  # Last 15 identified sounds
current_status = {"level": "NORMAL", "detail": "System Monitoring...", "timestamp": ""}
last_alert_sent = 0               # To prevent notification spam

# --- 3. NOTIFICATION & LOGGING ---
def notify_and_log(detail):
    global last_alert_sent
    # A. Send Telegram Alert (Max once per 60 seconds)
    if time.time() - last_alert_sent > 60:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID, 
            "text": f"🚨 SIGNAL WATCHER ALERT 🚨\n{detail}\nTime: {time.ctime()}"
        }
        try:
            requests.post(url, json=payload, timeout=5)
            print("[System] Telegram notification sent.")
            last_alert_sent = time.time()
        except Exception as e:
            print(f"[Error] Telegram failed: {e}")

    # B. Step 5: Save to CSV Logger
    try:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.ctime(), detail])
    except Exception as e:
        print(f"[Error] Logging failed: {e}")

# --- 4. THE LOGIC BRAIN ---
def check_patterns():
    global current_status
    if not event_history: return
    
    latest = event_history[-1]
    
    # IMMEDIATE THREAT DETECTION
    emergency_keywords = ["Siren", "Alarm", "Explosion", "Gunshot", "Screaming"]
    if any(k.lower() in latest['label'].lower() for k in emergency_keywords):
        current_status = {
            "level": "CRITICAL",
            "detail": f"IMMEDIATE EMERGENCY: {latest['label']} detected!",
            "timestamp": time.ctime()
        }
        notify_and_log(current_status['detail'])

# --- 5. THE AI EAR (Background Thread) ---
def run_ai_ear():
    FS, TARGET_FS = 44100, 16000
    
    def callback(indata, frames, time_info, status):
        audio_memory.append(indata.copy())
        
        # Only process if volume is above 3%
        if np.max(np.abs(indata)) > 0.03: 
            full_buffer = np.concatenate(list(audio_memory)).flatten()
            resampled = librosa.resample(full_buffer, orig_sr=FS, target_sr=TARGET_FS)
            
            # AI Inference
            scores, _, _ = model(resampled)
            mean_scores = np.mean(scores, axis=0)
            top_class = np.argmax(mean_scores)
            
            label = class_names[top_class]
            conf = mean_scores[top_class]
            
            # Gate: Record sounds with >25% confidence
            if conf > 0.25 and label not in ['Silence', 'Background noise']:
                event_history.append({"label": label, "time": time.time()})
                print(f"[AI] Heard: {label} ({conf:.1%})")
                check_patterns()

    with sd.InputStream(device=1, channels=1, samplerate=FS, blocksize=4410, callback=callback):
        print("--- AI Ear is Listening (Real-Time) ---")
        while True: sd.sleep(1000)

# --- 6. THE WEB INTERFACE (FastAPI) ---
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    try:
        with open("templates/dashboard.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: templates/dashboard.html not found!</h1>"

@app.get("/status")
async def get_status():
    return current_status

@app.get("/reset")
async def reset_status():
    global current_status
    current_status = {"level": "NORMAL", "detail": "System Reset", "timestamp": time.ctime()}
    return {"message": "System status cleared."}

if __name__ == "__main__":
    # Start the AI Ear in a background thread
    threading.Thread(target=run_ai_ear, daemon=True).start()
    
    # Start the Web Server
    print("Dashboard live at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)