import sounddevice as sd

print("Checking available audio devices...\n")
devices = sd.query_devices()

for i, device in enumerate(devices):
    # We only care about devices that can 'input' sound
    if device['max_input_channels'] > 0:
        print(f"Index {i}: {device['name']}")

print("\n--- Summary ---")
default_input = sd.default.device[0]
print(f"Your system's default input index is: {default_input}")