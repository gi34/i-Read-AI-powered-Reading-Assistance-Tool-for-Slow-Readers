# i-Read: AI-Powered Reading Assistance Tool for Slow Readers
An AI-powered reading assistance tool for slow readers to improve reading skills by capturing pronunciation, providing real time visual cues and personalised feedback.


## Project overview
Many slow readers experience difficulty to read fluently. This project aims to aid slow readers in enhancing their reading and pronunciation skills via real time feedback and visual cues. The system capture real time pronunciation and validate the accuracy before proceeding to the next word, allowing slow readers to get instant feedback without coach or guidance.


## Features
- AI-powered pronunciation analysis
- Live pronunciation capture
- Real time feedback for mispronounced words
- Personalised improvement suggestions
- low latency speech processing

## Project Structure

```
ML
└──  app.py        # main file for running in web
└──  model_only    # include whisper and vosk only
└──  model.py
|   # consist of the main engine of the system
|   # connect with app.py as the backend
|
└──  showGraph.py
|   # print the latency and other graphs
|
└──  silent_whisper.py
|   # try on whisper model
|
└──  stress_logger.py
|   # log the latency during the reading
|
└──  suggestion.py
|   # provide suggestion after reading session
|   # connect with app.py and model.py
|
└──  v.py
|   # try on vosk model
|
└──  templates
|   # contains the html of each page
|
└──  stories
|   # contains the raw text of th stories
|
└──  static
|   # contains the css and png 
```

## Demo
[Watch demo video] [https://youtu.be/UKykSsWITZ0]

## Techonolgies Used
- Python
- NLTK
- Whisper
- Vosk
- llm
- socketIO
- flask
- asyncio
