from flask import *
import numpy as np
from keras.models import load_model
import fasttext
import re

model=load_model('./Youtube Project/model.h5')

en_model=fasttext.load_model('./Youtube Project/cc.en.300.bin')


app=Flask(__name__,template_folder='templates')

def preprocess(text):
   process_text=re.compile('[^A-Za-z0-9]')
   text=re.sub(preprocess,' ',process_text.lower())
   return text

def vectorize_text(text):
    vector = en_model.get_sentence_vector(text)
    return vector

@app.route('/')
def home():
    return render_template('index.html')    

def predict():
    if request.method == 'POST':
        input_text = request.form['text']

        processed_text = preprocess(input_text)

        vectorized_text = vectorize_text(processed_text)
        vectorized_text = np.reshape(vectorized_text, (-1,1,300))

        prediction = model.predict(vectorized_text)

        predicted_class = np.argmax(prediction)

        classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7']
        result = classes[predicted_class]

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)