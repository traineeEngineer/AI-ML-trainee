import streamlit as st
from dotenv import load_dotenv
from utils import query_agent

load_dotenv()


st.title("Let's do some analysis on your csv")
st.header('please upload your csv file here')

data=st.file_uploader("upload csv file",type='csv')

query=st.text_area("Enter your query")
button=st.button('Generate Response')

if button:
    answer= query_agent(data,query)
    st.write(answer)
