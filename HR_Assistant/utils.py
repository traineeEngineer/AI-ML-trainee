import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub

#Ectract info from files
def get_pdf(pdf_doc):
    text=""
    pdf_reader=PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text+=page.extract_text
    return text

#iterate over files in that user upload files one by one
def create_docs(user_pdf_list,uniqu_id):
    docs=[]
    
    for filename in user_pdf_list:
        chunks=get_pdf(filename)
        
        docs.append(Document(page_content=chunks,metadata={
            "name":filename.name,
            "id":filename.id,
            "type":filename.type,
            "size":filename.size,
            "unique_id":uniqu_id
        }))   
    return docs 

# create embedding instance
def create_embedding_load_data():
    embeding=SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    return embeding

# push to pinecone 
def push_to_pinecone(pinecone_apikey,pinecone_env,index_name,embeddins,docs):
    pinecone.init(
        api_key=pinecone_apikey,
        environment=pinecone_env,
    )   
    print('Done here')
    Pinecone.from_documents(docs,embeddins,index_name)

# pull relevant docs from pinecone
def pull_to_pinecone(pinecone_apikey,pinecone_env,index_name,embeddins):
    pinecone.init(
        api_key=pinecone_apikey,
        environment=pinecone_env,
    )   
    index_name=index_name
    index=Pinecone.from_existing_index(index_name,embeddins)
    return index

# function to get similar docs form pinecone
def similar_docs(query,k,pinecone_apikey,pinecone_env,index_name,embeddins,unique_id):
    pinecone.init(
        api_key=pinecone_apikey,
        environment=pinecone_env,
    ) 
    index_name=index_name
    index=pull_to_pinecone(pinecone_apikey,pinecone_env,index_name,embeddins)
    similar_docs=index.similarity_search_with_score(query,int(k),{'unique_id':unique_id})
    return similar_docs

# helps us get summary docs
def get_summary(current_docs):
    llm=OpenAI(temperature=0)
    chain=load_summarize_chain(llm,chain_type='map_reduce')
    summary=chain.run([current_docs])
    return summary
    
    
    
