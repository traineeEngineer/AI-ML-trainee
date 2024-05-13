import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import crea_retrieval_chain
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()

## load Groq-api
groq_api=os.environ["GROQ_API_KEY"]="apikey"

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader('https://docs.smith.langchain.com/')
    st.session_state.docs=st.session_state.loader.load()
    
    st.session_state.text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_spliter.split_documents(st.session_state.docs[:20])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    
st.title("ChatGroq Demo")
llm=ChatGroq(groq_api=groq_api,
             model='Gemma-7b-It')

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on provided context only.
    please provide the most accurate reponse based on the question
    <context>
    {context}
    </context>
    
    Question:{input}
    """
)

doc_chain=create_stuff_documents_chain(llm,prompt)
retriver=st.session_state.vectors.as_retriever()

prompt=st.text_input("input your prompt here")

if prompt:
    response= retriver.invoke({"input":prompt})
    st.write(response['answer'])  
    
    with st.expander('Document similarity search'):
       for i , doc in enumerate(response['context']):
          st.write(doc.page_content)
          st.write("----------------------------")
    
    