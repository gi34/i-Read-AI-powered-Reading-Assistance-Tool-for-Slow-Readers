import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import sys

model_path = "vosk"  # Adjust the path
model = Model(model_path)

recognizer = KaldiRecognizer(model, 16000)  # 16 kHz sampling rate

def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr)
    
    # Convert indata (cffi buffer) to NumPy array and then to raw PCM data
    raw_data = np.frombuffer(indata, dtype=np.int16).tobytes()
    
    # Feed raw PCM data to Vosk
    if recognizer.AcceptWaveform(raw_data):
        print("Final result:", recognizer.Result())
    else:
        print("Partial result:", recognizer.PartialResult())

# Start recording
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopped.")
