
from flask import Flask, render_template, jsonify
import os, time
from nltk.tokenize import sent_tokenize
import threading, asyncio
from flask_socketio import SocketIO
import suggestion
model = None



app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins='*', ping_interval=0.05, ping_timeout=5, async_mode='eventlet') #enable websocket


# Function to split sentences from the selected story
def split_sentences(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()

    sentences = sent_tokenize(text)
    return sentences

import suggestion


@app.route('/report')
def conclude():
    
    model.conclusion()
    wpm = model.wpm
    cwpm = model.cwpm
    flaglist = model.flag
    
    generate = suggestion.get_suggestion(flaglist)
    suggestion_data = suggestion.clean_groq_output(generate)

    model.on_stop_signal()
    socketio.emit("new_target_word", {
        "chunk_index": 0,
        "word_index": 0,
        "target_word": "RESET"
    })

    return render_template('conclude.html', wpm = wpm, cwpm = cwpm, flag = flaglist, suggestion = suggestion_data)


@app.route('/')
def home():
    return render_template('index.html',wpm=0, cwpm=0)

#----------------> this is a function to stop the system
@app.route('/stop_transcription')
def stop():
    print("🛑 Stopping transcription & resetting word index")
    
    model.on_stop_signal()
    socketio.emit("new_target_word", {
        "chunk_index": 0,
        "word_index": 0,
        "target_word": "RESET"
    })

    return jsonify({'status': 'Transcription stopped & reset'})


@socketio.on("hyphenated_word")
def hyphenated_word(data):
    chunk_index = data['chunk_index']
    word_index = data["word_index"]
    hyphenated_word = data["hyphenated_word"]
    hyphen_count = data["hyphen_count"]

    print("hyphenated word", hyphenated_word)
    socketio.emit("hyphenated_word", {
        "chunk_index": chunk_index,
        "word_index": word_index,
        "hyphenated_word": hyphenated_word,
        "hyphen_count": hyphen_count
    })


@app.route('/read/<story_name>')
def read_story(story_name):
    global chunks
    for _ in range(10):  # Try for 10 seconds
        if model is not None:
            break
        print("⏳ Waiting for model to initialize...")
        time.sleep(1)

    if model is None:
        print("❌ Model failed to initialize!")
        return "Error: Model not initialized", 500  

    model.set_story(story_name)
    story_path = f'stories/{story_name}.txt'

    if not os.path.exists(story_path):
        return "Story not found", 404

    chunks = split_sentences(story_path)
    return render_template('text.html', story_name=story_name, chunks=chunks)




@socketio.on("new_target_word")
def handle_update_chunk(data):
    # Use the data from model.py directly, use data cuz need to update the value dynamically from model
    chunk_index = data["chunk_index"]
    idx = data["word_index"]
    target_word = data["target_word"]
    #sent_time = data["sent_time"]

    # Just emit the exact data to frontend
    socketio.emit("new_target_word", {
        "chunk_index": chunk_index,
        "word_index": idx,
        "target_word": target_word
        #"sent_time": sent_time
    })


transcription_thread = None

#---------------> to start the system
@app.route('/start_transcription')
def start_transcription():
    global transcription_thread


    def run_model():
        loop = asyncio.new_event_loop()  # ✅ Create a new event loop
        asyncio.set_event_loop(loop)  # ✅ Set it for this thread
        loop.run_until_complete(model.main())  # ✅ Run model.main() in this loop

    if transcription_thread is None or not transcription_thread.is_alive():
        transcription_thread = threading.Thread(target=run_model, daemon=True)
        transcription_thread.start()
        return jsonify({'status': 'Transcription started'})

    return jsonify({'status': 'Already running'})

def start_model():
    time.sleep(2)  # Ensure Flask has started before connecting
    global model
    import model
    model= model
    print("✅ Imported model.py!")
    model.connect_to_flask()  # Connect WebSocket client from model.py


@socketio.on('calibration_done')
def handle_calibration_done():
    print("Calibration complete")
    socketio.emit('calibration_done')  # Forward the event to frontend


@socketio.on('pause_transcription')
def pause_transcription():
    model.pause_transcription()



@socketio.on('get_definition')
def get_definition(data):
    word = data.get('word')
    model.get_definition(word)


@socketio.on('show_definition')
def show_definition(data):
    socketio.emit("show_definition",{
        'definition': data["definition"],
        'word': data['word']
    })
 

# this is for the TTS in report part
@socketio.on('trigger_tts')
def handle_tts(data):
   word = data['word']
   model.read_text(word)


from flask import request
@app.route('/log_highlight_latency', methods=['POST'])
def log_highlight_latency():
    data = request.get_json()
    latency = data.get('latency', 0.0)
    word = data.get('word')

    with open("highlight_latency_log.txt", "a") as f:
        f.write(f"{word}: {latency} s\n")
    return '', 204





if __name__ == '__main__':
    # Start the model in a separate thread
    model_thread = threading.Thread(target=start_model, daemon=True)
    model_thread.start()

    # Start Flask in the main thread
    socketio.run(app, port=5000, debug=True)



