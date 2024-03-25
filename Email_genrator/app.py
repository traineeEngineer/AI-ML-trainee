import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers


#function get response abck
def getLLMResponse(form_input,email_sender,email_recipient,email_style):
    
    llm=CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={
                          'max_new_token':256,
                          'temperature':0.01
                      })

    template="""
    Write an email with{style} and includes topic:{email_topic}.\n\nSender:{sender}\nRecipient:{recipient}
    \n\n EmailText:
    """
    prompt=PromptTemplate(input_variables=["style",'email_topic','sender','recipient'],template=template)
    
    response=llm(prompt.format(email_topic=form_input,sender=email_sender,recipient=email_recipient,style=email_style))
    return response

st.set_page_config(page_title='Generate Email',
                   page_icon=':email',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header('Generate Emails')

form_input=st.text_area("Enter email topic",height=275)

col1,col2,col3=st.columns([10,10,5])
with col1:
    email_sender=st.text_input("sender name")
with col2:
    email_recipient=st.text_input('Recipient name')
with col3:
    email_style=st.selectbox('Writing style',('Formal','Appreciating','Not Satisfied','Neutral'),index=0)    
    
submit=st.button("Generate")

if submit:
    st.write(getLLMResponse(form_input,email_sender,email_recipient,email_style))    
