import streamlit as st
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Function to load the model
@st.cache_resource
def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        torch_dtype=torch.bfloat16
    )
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, quantization_config=quantization_config, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
    model.eval()
    return model, tokenizer

# Sidebar for file upload
with st.sidebar:
    st.markdown('**Upload Image**')
    uploaded_file = st.file_uploader("", type=['jpeg', 'png', 'jpg'])

# Display uploaded image
with st.container():
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=350, use_column_width=False)

# Chat input
user_input = st.chat_input("Chat here")

# Load model
model, tokenizer = load_model()

# Process and generate response
if uploaded_file is not None and user_input:
    with st.spinner('Generating response...'):
        image = Image.open(uploaded_file).convert('RGB')
        question = user_input
        msgs = [{'role': 'user', 'content': question}]

        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.8,
            stream=True
        )

        generated_text = ""
        for new_text in res:
            generated_text += new_text
        with st.container(height=200,border=True):
            st.write(generated_text)
