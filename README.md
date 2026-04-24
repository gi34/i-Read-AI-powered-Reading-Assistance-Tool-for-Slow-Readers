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
