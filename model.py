#model+ sentence + pause detection + hyphen + TTS
#!!! pytorch version 2.5.1, weight_only == false. in the future if torch update break model loading, set weight=true manually or update whisper. else downgrade torch
# pip install torch==2.0.1

import asyncio, pyphen, pyttsx3,string
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json, Levenshtein
import whisper
import numpy as np
import time, threading
from threading import Lock
from nltk.tokenize import sent_tokenize
import socketio, re
from nltk.corpus import wordnet as wn
import webrtcvad

vad = webrtcvad.Vad(3)  # Mode 3 is the most aggressive for speech detection

# Global variables to track speech timing
startTime = None
total_speaking_time = 0


tts_active = False
sio = socketio.Client()

vosk_model = Model("vosk")
recognizer = KaldiRecognizer(vosk_model, 16000)
whisper_model = whisper.load_model("base.en")
sample_rate = 16000
chunk_size = int(sample_rate * 1)  
audio_queue = asyncio.Queue(maxsize=100)
stop_transcription = False
whisper_text=""
vosk_text=""
volume = 100  # Initial volume for pause detection
count = 0 # count 3 seconds and then trigger whisper
hyphen_count=0
CHUNK = 16000
pause = 5 #5 sec of pause
is_paused = False
target_word = "get"
volume_lock = Lock()  
start_time=time.time()
idx = 0
chunk_index = 0
engine = pyttsx3.init()
engine.setProperty("rate", 100)
total_word=1
absolute_word = 0
previous_partial_sentence = ""
flag = []
cwpm = 1
wpm = 1
total_speaking_time = 1
calibration_done = asyncio.Event()

#from stress_logger import StressLogger
#logger = StressLogger()
#from datetime import datetime

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2)  


#-------> calculate the parameter
async def start_speaking():
    """Start the timer when speech is detected."""
    global startTime, target_word
    await calibration_done.wait() 

    if startTime is None:
        startTime = time.time()
        #print("Speech started!", target_word)

def stop_speaking():  # 0.2 seconds = 200 ms
    """Stop the timer when speech stops and accumulate speaking time."""
    global startTime, total_speaking_time
    if startTime is not None:
        elapsed_time = time.time() - startTime
        #sentence_changed = partial_sentence != previous_partial_sentence

        if elapsed_time <= 4 :
            total_speaking_time += elapsed_time
            #print(f"Speech stopped! Duration: {elapsed_time:.2f} seconds")
        
        #previous_partial_sentence = partial_sentence
        startTime = None



def is_speech(audio_chunk, sample_rate=16000, frame_duration=30):
    """
    Returns True if speech is detected in the audio chunk.
    :param audio_chunk: A raw audio chunk (byte data).
    :param sample_rate: The sample rate of the audio, typically 16000 for VAD.
    :param frame_duration: The duration of each frame (in milliseconds). Default is 30 ms.
    :return: True if speech is detected, False otherwise.
    """
    # Convert the audio chunk into 16-bit PCM format (VAD requires this)
    audio_samples = np.frombuffer(audio_chunk, dtype=np.int16)
    
    # Calculate the number of frames per chunk (based on sample rate and frame duration)
    frame_size = int(sample_rate * frame_duration / 1000)  # Frame size in samples
    num_frames = len(audio_samples) // frame_size
    
    for i in range(num_frames):
        # Extract the current frame
        frame = audio_samples[i * frame_size:(i + 1) * frame_size]
        
        # Check if speech is present in this frame
        if vad.is_speech(frame.tobytes(), sample_rate):
            return True  # Speech detected in this frame
    
    return False  # No speech detected in any frame

def conclusion():
    global total_word, total_speaking_time, flag, wpm, cwpm
    stop_speaking()

    #logger.plot_resource_usage()

    if total_speaking_time ==0:
        total_speaking_time = 60

    elif total_speaking_time > 5:
        print("raw: ",total_speaking_time)
        total_speaking_time-=5

    if total_word == 0:
        total_word = 1

    print("total time taken:", total_speaking_time)
    print("total word: ", total_word)

    wpm = round((total_word * 60) / total_speaking_time, 2)
    print(f"Words Per Minute (WPM): {wpm:.2f}")

    cwpm = round((total_word -len(flag))*60 / total_speaking_time,2)
    print(len(flag))
    print(f"C Words Per Minute (CWPM): {cwpm:.2f}")




def on_stop_signal():
    #print("🛑 Stop signal received! Stopping transcription...")
    global stop_transcription, chunk_index, absolute_word, is_paused,total_word, total_speaking_time, wpm, cwpm, flag
    stop_transcription = True
    is_paused = False
    chunk_index=0
    absolute_word=0
    total_speaking_time=0
    total_word=0
    wpm = 1
    cwpm = 1
    flag=[]
    
    
def pause_transcription():
    global is_paused
    if is_paused:
        #print('Spacebar is pressed. Resumed transcription.')
        is_paused=False
    else:
        #print('🛑 Pause transcription')
        is_paused = True




#this function is being called from app.py to connect to flask
def connect_to_flask():
    while True:
        try:
            print("🔄 Attempting to connect to Flask WebSocket...")
            sio.connect("http://127.0.0.1:5000")
            print("✅ Successfully connected!")
            break  # Exit loop once connected
        except socketio.exceptions.ConnectionError:
            print("🔴 Connection failed, retrying in 2 seconds...")
            time.sleep(2)  # Wait before retrying

    sio.wait()  # Keep listening for messages




#----------->definition----------------------->
def get_definition(word):
    synsets = wn.synsets(word) # get the meaning
    
    if synsets:
        definition = synsets[0].definition() # get the first definition
        sio.emit("show_definition",{
            "definition": definition,
            "word": word
        })
        #print("Original definition:", definition)

    else:
        sio.emit("show_definition",{
            "definition": "Definition not found",
            "word": word
        }) 



#-----------> engine-------------------------->
#split the text into chunks for every one full sentence after removing punctuation
def split_sentences(filename):
    with open(filename, "r", encoding = "utf-8") as file:
        text = file.read()

    text = text.replace(" - ", " ")
    text = re.sub(r"[\u002D\u2013\u2014]", " ", text)

    sentences = sent_tokenize(text)

    #remove punctuation

    # Define punctuation to remove (after handling hyphens)
    punctuation_to_remove = string.punctuation + "“”‘’—"

    # Remove punctuation from each sentence
    cleaned_sentences = [
        sentence.translate(str.maketrans("", "", punctuation_to_remove)).strip()
        for sentence in sentences
    ]

    return cleaned_sentences  # Returns cleaned sentences with spaces instead of hyphens





filename = f'stories/The lottery.txt'
chunks = "get it"
partial_sentence = chunks


import wave
chunk = 1
def save_audio(audio_bytes, filename, sample_rate=16000):
    global chunk
    filename = f"chunk_{chunk}.wav"
    chunk+=1

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit audio = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)


def set_story (story_name):
    global filename
    #print(story_name)
    filename = f'stories/{story_name}.txt'

def get_story():
    return filename
    
def audio_callback(indata, frames, time, status):
    global volume, is_paused, tts_active
    if tts_active:
        return

    if is_paused:
        print('stop receiving audio')
    else:
        current_volume = np.abs(np.frombuffer(indata, dtype=np.int16)).mean()
        with volume_lock:
            volume = current_volume
        audio_queue.put_nowait(bytes(indata))
        #logger.mark("audio received")
        #logger.measure("audio received")
        #logger.log_cpu_memory()

def calculate_accuracy(reference, text, threshold=1):
    global start_time
    distance = Levenshtein.distance(text.lower(), reference.lower())
    max_length = max(len(text), len(reference))
    if distance<=threshold:
        #print(distance, max_length)
        accuracy=1.0
        start_time=time.time()
        #print(f"Time taken: {start_time:.2f} seconds")
    else:
        #max_length= max(len(text),len(reference))
        accuracy = (1 - distance / max_length)

    return accuracy

#remove the correctly spoken word
def remaining(vosk_text, whisper_text, partial_sentence):
    global target_word, hyphen_count,start_time, idx, absolute_word, total_word
    vosk_words = vosk_text.split()
    whisper_words = whisper_text.strip().split()
    sentence_words = partial_sentence.split()
 
    while idx < len(sentence_words):
        target_word = sentence_words[idx]
        vosk_word = vosk_words[idx] if idx < len(vosk_words) else ''
        whisper_word = whisper_words[idx] if idx < len(whisper_words) else ''

        if vosk_word.lower() == target_word or whisper_word.lower() == target_word:
            #logger.request_highlight(absolute_word, target_word)


            idx += 1
            hyphen_count=0
            start_time=time.time()
            absolute_word +=1
            total_word+=1
            
            
            #elapsed_time = int(time.time() - start_time)
            #print(f"{elapsed_time}s...", end="\r")


        else:
            vosk_accuracy = calculate_accuracy(vosk_word, target_word,threshold=1)
            whisper_accuracy = calculate_accuracy(whisper_word, target_word,threshold=1)
            max_accuracy = max(vosk_accuracy, whisper_accuracy)

            if max_accuracy >= 0.6:
                #print(f"read successfully: '{target_word}'  Accuracy: {max_accuracy}")
                #logger.request_highlight(absolute_word, target_word)

                idx += 1
                absolute_word+=1
                total_word+=1
                hyphen_count=0
                start_time=time.time()
                #elapsed_time = int(time.time() - start_time)
               # print(f"{elapsed_time}s...", end="\r")

            else:
                #print(f"read: '{target_word}' (Vosk={vosk_accuracy:.2f}, Whisper={whisper_accuracy:.2f})")
                break


    partial_sentence = " ".join(sentence_words[idx:])
    idx = 0
    return partial_sentence

async def process_vosk(data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, sync_process_vosk, data)

def sync_process_vosk(data):
    if recognizer.AcceptWaveform(data):
        vosk_result = json.loads(recognizer.Result())
        return vosk_result.get("text", "").strip()
    return ""

def process_whisper(audio_chunk):
    audio_chunk_32 = np.frombuffer(audio_chunk, dtype='int16').astype('float32') / 32768.0
    result = whisper_model.transcribe(audio_chunk_32, temperature=0.5)
    return result['text'].strip()


async def transcription_loop():
    buffer = b''
    global chunk_index # Index of the current chunk
    global stop_transcription, count, whisper_text, vosk_text, partial_sentence, idx, target_word,absolute_word, total_word

   

    if chunk_index < len(chunks):
        reference = chunks[chunk_index]
        partial_sentence = reference
        #print(f"Current chunk: {reference}")

    while not stop_transcription:
        if chunk_index >= len(chunks):
            #print("All words read!")
            stop_transcription = True
            break

        while tts_active:
            await asyncio.sleep(1)

        if not audio_queue.empty():
            audio_chunk = await audio_queue.get()
            buffer += audio_chunk
            count += 1
            
            if is_speech(audio_chunk):
                #save_audio(audio_chunk, "test_chunk.wav")
                await start_speaking()  # Start the timer when speech is detected
            else:
                stop_speaking()
            
            #logger.send_highlight_event(chunk_index,absolute_word,target_word)
           # '''
            #sent_time = int(datetime.now().timestamp() * 1000)
            emit_thread = threading.Thread(target=lambda: sio.emit("new_target_word", {
                "chunk_index": chunk_index,
                "word_index": absolute_word,
                "target_word": target_word
                #"sent_time": sent_time
            }), daemon=True)
            emit_thread.start()
            #'''

            
            vosk_task = asyncio.create_task(process_vosk(audio_chunk))
            vosk_text = await vosk_task

            #print(f"Vosk: {vosk_text}")
            partial_sentence = remaining(vosk_text, whisper_text, partial_sentence)
            #print('target: ', target_word)
            #print('Sentence to read:', partial_sentence, "\n")

            if count == 3:  # After 3 times (1 sec)
                whisper_task = asyncio.get_event_loop().run_in_executor(executor, process_whisper, buffer)
                whisper_text = await whisper_task
                #print(f"Whisper: {whisper_text}")
                count = 0

                partial_sentence = remaining(vosk_text, whisper_text, partial_sentence)
                #print('Sentence to read:', partial_sentence, "\n")

                buffer = b''

                #check if the sentence is finished reading
                if not partial_sentence:
                    chunk_index += 1
                    #print("total word in partial:",total_word)
                    absolute_word=0
                    if chunk_index < len(chunks):
                        reference = chunks[chunk_index]
                        partial_sentence = reference
                        #print(f"Next chunk: {reference}")

                        #'''
                        #logger.send_highlight_event(chunk_index,absolute_word,target_word)

                        #sent_time = int(datetime.now().timestamp() * 1000)
                        emit_thread = threading.Thread(target=lambda: sio.emit("new_target_word", {
                            "chunk_index": chunk_index,
                            "word_index": absolute_word,
                            "target_word": target_word
                            #"sent_time": sent_time
                        }), daemon=True)
                        emit_thread.start()
                        #'''

                    else:
                        print("All words read!")
                        stop_transcription = True
        else:
            await asyncio.sleep(0.1)




async def calibrate_noise():
    #print("Calibrating... Please stay silent for 5 seconds.")
    global volume
    noise_levels = []
    calibration_duration = 5  # Desired calibration duration in seconds
    iterations = int(sample_rate / CHUNK * calibration_duration)  # Total iterations 1 sec per chunk, 16000 of sample rate and Chunk, total 5 sec of audio collected.
    sleep_duration = calibration_duration / iterations  # Sleep duration per iteration

    for _ in range(iterations):
        with volume_lock:
            noise_levels.append(volume)
        await asyncio.sleep(sleep_duration)

    dynamic_threshold = np.mean(noise_levels)+300
    #print(f"Dynamic Threshold Set: {dynamic_threshold}")

    # pass the current word index to flask

    #'''
    #logger.send_highlight_event(chunk_index,absolute_word,target_word)
    #sent_time = int(datetime.now().timestamp() * 1000)
    emit_thread = threading.Thread(target=lambda: sio.emit("new_target_word", {
        "chunk_index": chunk_index,
        "word_index": absolute_word,
        "target_word": target_word
        #"sent_time": sent_time
    }), daemon=True)
    emit_thread.start()
    #'''

    sio.emit("calibration_done")

    calibration_done.set()  # Let others know calibration is done

    return dynamic_threshold


async def hyphen():
    global hyphen_count
    pyphen_en = pyphen.Pyphen(lang='en')

    word = target_word
    hyphenated_word = pyphen_en.inserted(word).replace("-", "·")


    sio.emit("hyphenated_word",{
        "chunk_index": chunk_index,
        "word_index": absolute_word,
        "hyphenated_word": hyphenated_word,
        "hyphen_count": hyphen_count
    })
    print ("hyphen",hyphenated_word)





#-------->TTS
def read_text(text):
    engine.say(text)
    engine.runAndWait()
    engine.endLoop()


tts_lock = threading.Lock()

def speak_text(text):
    global tts_active
    with tts_lock:
        try:
            engine.say(text)
            engine.runAndWait()
        except RuntimeError as e:
            print(f"TTS Error: {e}")
        finally:
            try:
                engine.endLoop()
            except Exception as e:
                print(f"TTS endLoop warning: {e}")
            tts_active = False
            print(f"TTS: Finished speaking '{text}'")



def TTS():
    global target_word, chunk_index, idx, absolute_word, total_word, tts_active
    if tts_active:
        print("TTS is already active. Skipping this call.")
        return
    
    tts_active = True
    tts_thread = threading.Thread(target=speak_text, args=(target_word,))
    tts_thread.start()
    tts_thread.join()
    
    flag.append(target_word) # add the target word to dictionary
    idx +=1
    absolute_word+=1
    total_word+=1

   # '''
    #logger.send_highlight_event(chunk_index,absolute_word,target_word)

    #sent_time = int(datetime.now().timestamp() * 1000) 
    emit_thread = threading.Thread(target=lambda: sio.emit("new_target_word", {
        "chunk_index": chunk_index,
        "word_index": absolute_word,
        "target_word": target_word
        #"sent_time": sent_time
    }), daemon=True)
    emit_thread.start()
   # '''


    

async def dynamic_pause_detection():
    global volume, hyphen_count, start_time, is_paused, startTime
    threshold = await calibrate_noise()
    startTime = start_time = time.time()
    
    while not stop_transcription:
        if is_paused:
            print(' pause detected')

        else:
            elapsed_time = int(time.time() - start_time)
            print(f"{elapsed_time}s...", end="\r")  

            if volume >= threshold:
                print(f"Attempt: {hyphen_count}")

                if hyphen_count == 5:
                    TTS()
                    hyphen_count=0

                elif hyphen_count == 3 :
                    await hyphen()
                    hyphen_count+=1
                else:
                    hyphen_count+=1

                start_time = time.time()
                print("\nSpeech detected! Timer reset.")
                
            
        
            elif elapsed_time >= pause:
                print(f"\nPause Detected (Dynamic)")
                TTS()
                hyphen_count=0
                start_time = time.time()  
                print("Timer reset!")

        await asyncio.sleep(1)



'''
def on_press(key):
    global is_paused
    try:
        if key.char == 'k':
            if is_paused:
                print('Key spacebar is pressed. Paused transcription.')
                is_paused=False
            else:
                print('Resumend transcription')
                is_paused = True
            
    except AttributeError:
        return True




listener = keyboard.Listener(on_press=on_press)
listener.start()
'''
# Main function
# sentences are splitted here
async def main():
    global stop_transcription, chunks
    # block size =8000, chunk = 8000/16000 = 0.5 sec
    try: 
        with sd.RawInputStream(samplerate=sample_rate, blocksize=8000, dtype="int16",
                               channels=1, callback=audio_callback):
            #print("Listening... Press 'k' to stop/resume.")
            stop_transcription = False
            chunks = split_sentences(get_story()) # need this in transcription loop
            await asyncio.gather(transcription_loop(), dynamic_pause_detection())  # Run both tasks concurrently
    except KeyboardInterrupt:
        print("\nTranscription stopped.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_transcription = True
        #listener.stop()

#asyncio.run(main())