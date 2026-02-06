import tensorflow as tf
import tensorflow_hub as hub

print(f"TensorFlow Version: {tf.__version__}")
try:
    # Attempt to access the model URL
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("AI SUCCESS: YAMNet model is accessible and ready!")
except Exception as e:
    print(f"AI ERROR: Could not connect to the model. Error: {e}")