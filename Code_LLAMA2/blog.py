import streamlit as st

from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

#Function to get Response LLAMA2

def get_llama_response(input_txt,no_words,blog_style):
    llm=CTransformers(model='./model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={"max_new_tokens":256,
                              "temperature":0.01})
    
    #Prompt Template
    tempalate="""
    write a blog for {blog_style} job profile for  topic{input_txt} 
    with in {no_words} words.
    """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_txt","no_words"],
                          template=tempalate)
    
    # Generate Response for LLama model
    response=llm(prompt.format(blog_style=blog_style,input_txt=input_txt,no_words=no_words))
    return response

print('Generate My Blog')

input_txt = input("Enter Blog Topic: ")

no_words = input('No of Words: ')
blog_style = input('Writing the blog for (Researchers/Datascientist/Common People): ')

# Final Response
response_text = get_llama_response(input_txt, no_words, blog_style)
print(response_text)
