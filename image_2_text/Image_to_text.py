import streamlit as st
from PIL import Image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch

@st.cache_resource
def load_model():
    # Configuration for quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        dtype=torch.float16
    )
    # Load the model and tokenizer
    model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True, quantization_config=quantization_config, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI
st.title("Image Q/A with MiniCPM-Llama3")

# Image upload in the sidebar
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    
    # Add an expander for the image
    with st.sidebar.expander("Uploaded Image"):
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Initialize session state to store questions and answers
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []

    if 'question' not in st.session_state:
        st.session_state.question = ""

    # Form for question input
    with st.form(key='question_form', clear_on_submit=True):
        question = st.text_input("Enter your question", value=st.session_state.question, key="question_input")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if question:
            # Display a spinner while processing
            with st.spinner('Thinking...'):
                # Create msgs variable
                msgs = [{'role': 'user', 'content': question}]
                
                # Call the model
                res = model.chat(
                    image=image,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.7,
                    stream=True
                )
                
                # Collect the generated text
                generated_text = ""
                for new_text in res:
                    generated_text += new_text
                
                # Store the question and answer in session state
                st.session_state.qa_pairs.append((question, generated_text))
                
                # Clear the input box
                st.session_state.question = ""
                # st.experimental_rerun()

    # Display all questions and answers
    for idx, (q, a) in enumerate(st.session_state.qa_pairs):
        st.write(f"**Question {idx + 1}:** {q}")
        st.write(f"**Answer:** {a}")
        st.markdown("---")