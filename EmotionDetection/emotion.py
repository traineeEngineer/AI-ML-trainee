import streamlit as st
import streamlit.components.v1 as components
import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

st.title('Emotion Detection 😊😐😕😡')


st.subheader("User Input Emotion Analysis")
st.text("")
userText = st.text_input('User Input', placeholder='Input text HERE')
st.text("")

import torch


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('Tuned_data')



# Assuming you have already loaded your model and tokenizer
def generate_reply(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    num_labels = len(model.config.id2label)
    summary = predicted_class_id
    return summary


def res(text):
    if text == 0:
        return "A N G R Y"
    elif text == 1:
        return "F E A R"
    elif text == 2:
        return "J O Y"
    elif text == 3:
        return "L O V E"
    elif text == 4:
        return "S A D N E S S"
    elif text == 5:
        return "S U R P R I S E"
    else:        
        pass


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_streamlit_joy = load_lottiefile("Animation - 1713783050333.json")
lottie_streamlit_head = load_lottiefile("Animation - 1713785099300.json")
lottie_streamlit_angry = load_lottiefile("lottie.json")
lottie_streamlit_fear = load_lottiefile("lottie (1).json")
lottie_streamlit_love = load_lottiefile("lottie (2).json")
lottie_streamlit_sad = load_lottiefile("lottie (3).json")
lottie_streamlit_surprise = load_lottiefile("lottie (4).json")

  

if st.button("Find Out"): 
    response = generate_reply(userText)
    result = res(response)
    if response == 0: # Angry
        st.success(result)
        st_lottie(lottie_streamlit_angry, speed=1, height=200, key="lottie_animation1")
        
    if response == 1: # fear
        st.success(result)
        st_lottie(lottie_streamlit_fear, speed=1, height=200, key="lottie_animation2")
        
    if response == 2: # joy
        st.success(result)
        st_lottie(lottie_streamlit_joy, speed=1, height=200, key="lottie_animation3")
        
    if response == 3: # love
        st.success(result)
        st_lottie(lottie_streamlit_love, speed=1, height=200, key="lottie_animation4")
        
    if response == 4: # sadness
        st.success(result)
        st_lottie(lottie_streamlit_sad, speed=1, height=200, key="lottie_animation5")
        
    if response == 5: # surprise
        st.success(result)
        st_lottie(lottie_streamlit_surprise, speed=1, height=200, key="lottie_animation6")
