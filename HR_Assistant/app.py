import streamlit as st
import uuid
from utils import *
from dotenv import load_dotenv


# creating session variable
if 'unique_id' not in st.session_state:
    st.session_state['unique_id']=''
    
def main():
    load_dotenv()
    st.set_page_config(page_title='Resume Screening Assistant')
    st.title('HR-Resum Screening Assistant')
    st.subheader('I can help you in resume screening process')
    
    job_description=st.text_area('please paste Job Description here',key=1)
    document_count=st.text_input('No of resumes',key=2)
    pdf=st.file_uploader('upload Resume here only PDF file allowed here',type=['pdf'],accept_multiple_files=True)
    
    submit=st.button("Help me with analysis")
    
    if submit:
        with st.spinner("wait for it..."):
            #st.write('our process')
            
            #creating a unique ID 
            st.session_state['unique_id']=uuid.uuid4().hex    
            #st.write(st.session_state['unique_id'])
            
            #Creating a doc list out all upload pdf files
            docs=create_docs(pdf,st.session_state['unique_id'])
            st.write(docs)
            
            # display the count of resumes
            st.write(len(docs))
            
            # create embeddings instance
            embeddings=create_embedding_load_data()
            
            #push to pine cone
            push_to_pinecone("api_key",'pincone_env',"mcq-app",docs,embeddings)
            
            #fetch relevant docs from pinecone
            relevant_docs=similar_docs(job_description,document_count,'pineconeapi_key','pincone_env',docs,embeddings)
            #st.write(relevant_docs)
            
            #Introduce line seperator
            st.write(":heavy_minus_sign:"*30)
            
            #for each item in relevant docs-we are displaying same info of it on UI
            for item in range(len(relevant_docs)):
                st.subheader(''+str(item))
                st.write("**File** :"+relevant_docs[item][0].metadata['name'])
                
                with st.expander("Show me"):
                    st.info('Match score'+str(relevant_docs[item][1]))
                    summary=get_summary(relevant_docs[item][0])
                    st.write("**summary**"+summary)
        st.success('Hope I was able to save our time')  
        
if __name__ =="__main__":
    main()
    
