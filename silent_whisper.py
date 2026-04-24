import sounddevice as sd
import numpy as np
import whisper
from pynput import keyboard
import asyncio

model = whisper.load_model("base.en")

sample_rate = 16000
duration = 3  # 2-second audio
silence_threshold = 0.00007
chunk_size = int(sample_rate * duration)
buffer = []  # Buffer to store audio samples
stop_transcription = False  


def on_press(key):
    global stop_transcription
    try:
        if key.char == 'k':
            print("Key 'k' pressed. Stopping transcription.")
            stop_transcription = True
            return False 
    except AttributeError:
        pass
    return True


async def process_audio():
    global buffer, stop_transcription

    while not stop_transcription:
        if len(buffer) >= chunk_size:  # ensure 2 sec of audio
            audio_chunk = np.array(buffer[:chunk_size], dtype="float32")
            buffer = buffer[chunk_size:]  # Remove processed samples

            rms = np.sqrt(np.mean(audio_chunk ** 2))
            if rms < silence_threshold:
                print("Pause detected. RMS: "+str(rms))
            else:
                result = model.transcribe(audio_chunk,temperature=0.5)
                transcript = result['text'].strip().lower()
                print(rms)
                print(f"Transcription: {transcript}")
        else:
            await asyncio.sleep(0.1)  # Wait briefly for buffer to fill


def audio(indata, frames, time, status):
    global buffer
    buffer.extend(indata.flatten())  # Append audio samples to the buffer


async def main():
    global stop_transcription
    
    #keyboard
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    #audio
    stream = sd.InputStream(callback=audio, channels=1, samplerate=sample_rate)

    print("Recording... Press 'k' to stop.")
    with stream:
        await process_audio()  # Run the audio processing loop

    listener.join()  # Ensure listener stops


asyncio.run(main())
