from langchain_google_genai import ChatGoogleGenerativeAI
# to create Chatbot require this template
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser


import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import os

os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")

# prompt template
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",'you are helpful assistant. Please response to the queries '),
        ("human","Question:{question}")
    ]
)


st.title("LangchainDemo with GOOGLE API")
input_text=st.text_input("search topic you want")

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
output_parser=StrOutputParser()
chain=prompt_template|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))

