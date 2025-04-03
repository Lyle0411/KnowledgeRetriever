import pandas as pd
import numpy as np
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model=os.getenv('DEFAULT_LARGE_MODEL'), temperature=0.8)
chain = (PromptTemplate(template="""{query}""",
                        input_variables=["query"])
         | llm)
embeddings = OpenAIEmbeddings(model = os.getenv('DEFAULT_EMBEDDING_MODEL'),
                              disallowed_special=())
def qa(query:str) -> str:  return chain.invoke({'query': query}).content

if __name__ == '__main__':
    # res = qa("你好？")
    # print(res)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(root_dir)
    print(project_root)