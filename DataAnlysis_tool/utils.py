from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain.llms import OpenAI

def query_agent(data,query):
    df=pd.read_csv(data)
    llm=OpenAI()
    agent=create_pandas_dataframe_agent(llm,df,verbose=True)
    
    return agent.run(query)