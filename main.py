from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense

app = FastAPI()

# Определение ввода данных для генерации текста
class TextGenerationRequest(BaseModel):
    start_text: str
    length: int

# Загрузка модели и необходимых данных
filename = "/content/sample_data/weights-improvement-17-1.8368-bigger.hdf5"
model = Sequential()
model.add(LSTM(256, input_shape=(100, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(59, activation='softmax'))  # Adjust the number of neurons according to your model
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Загрузка текста и подготовка символов
text_file_path = "/content/sample_data/crimeandpunishment.txt"
raw_text = open(text_file_path, 'r', encoding='utf-8').read().lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_vocab = len(chars)

@app.post("/generate/")
def generate_text(request: TextGenerationRequest):
    start_text = request.start_text.lower()
    length = request.length
    
    if len(start_text) < 100:
        raise HTTPException(status_code=400, detail="Start text must be at least 100 characters long.")
    
    # Подготовка начального паттерна
    pattern = [char_to_int[char] for char in start_text[:100]]
    
    generated_text = start_text
    
    for i in range(length):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        
        generated_text += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)